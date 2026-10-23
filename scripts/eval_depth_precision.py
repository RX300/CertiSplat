"""Analyse sparse-view priors against available all-view COLMAP depth references."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewshot.camera import Camera, sample_image
from fewshot.data import SCENES, dtu_background_mask, load_scene
from fewshot.prior_io import nearest, read_array, write_json

REFERENCE_ROOT = Path("/tmp/claude-42048/-workspace-ubuntu2004-cuda12-1-projects-3dgs-fewshot/"
                      "a4c09ad5-1f1c-46f1-a864-c077c0772d2a/scratchpad/colmap")
SOURCES = ["stereo", "mvs", "da3", "stereogs_mvsanywhere"]


def reference_ready(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            char = stream.read(1)
            if not char:
                return False
            header.extend(char)
        w, h, c = map(int, header[:-1].split(b"&"))
    return path.stat().st_size == len(header) + 4 * w * h * c


@torch.no_grad()
def covisible_masks(views, depths):
    cameras = [Camera.from_view(v, with_image=False, device="cpu") for v in views]
    tensors = [torch.from_numpy(d) for d in depths]
    masks = []
    for i, (camera, depth) in enumerate(zip(cameras, tensors)):
        points = camera.backproject(depth)
        count = torch.zeros_like(depth, dtype=torch.int32)
        for j, other in enumerate(cameras):
            if i == j:
                continue
            u, v, z = other.project(points)
            inside = (u >= 0) & (u <= other.width - 1) & (v >= 0) & (v <= other.height - 1) & (z > 0)
            seen = sample_image(tensors[j][None], u, v, mode="nearest")[0]
            count += inside & (seen > 0) & ((z - seen).abs() <= 0.02 * seen)
        masks.append(((depth > 0) & torch.isfinite(depth) & (count >= 1)).numpy())
    return masks


def load_source(scene, source, root, reference_root):
    views = scene.train
    if source not in ("da3", "stereogs_mvsanywhere"):
        paths = [root / scene.dataset / "3v" / scene.name / source / f"{v.name}.npz" for v in views]
        if not all(p.is_file() for p in paths):
            return None
        depths, masks = [], []
        for path in paths:
            with np.load(path) as data:
                depths.append(data["depth"])
                masks.append(data["valid"])
    else:
        path = (reference_root / f"LLFF_{scene.name}_da3train.npz" if source == "da3" else
                Path("datasets/nerf_llff_data") / scene.name /
                "MVSAnywhere_3_views_H_1440_W_1920_img_dist_thresh_3.0_depth_thresh_0.08_vthresh_2_init/mvs_depths_masks.npz")
        if not path.is_file():
            return None
        with np.load(path) as data:
            names = data["names" if source == "da3" else "image_names"].tolist()
            assert names == [v.name for v in views], (source, scene.name, names)
            raw = data["depth" if source == "da3" else "depths"]
            raw_masks = np.isfinite(raw) & (raw > 0) if source == "da3" else data["masks"]
            depths = [nearest(d, v.height, v.width) for d, v in zip(raw, views)]
            masks = [nearest(m, v.height, v.width) for m, v in zip(raw_masks, views)]
    valid = [m & np.isfinite(d) & (d > 0) for d, m in zip(depths, masks)]
    return depths, valid


def measurements(depths, valid, references, covisible, regions=None):
    if regions is not None:
        valid = [m & region for m, region in zip(valid, regions)]
        covisible = [c & region for c, region in zip(covisible, regions)]
    errors = [np.abs(d[m & c] - r[m & c]) / r[m & c]
              for d, m, r, c in zip(depths, valid, references, covisible)]
    return {"pixels": (sum(d.size for d in depths) if regions is None
                       else sum(int(region.sum()) for region in regions)),
            "valid": sum(int(m.sum()) for m in valid),
            "covisible": sum(int(c.sum()) for c in covisible),
            "valid_covisible": sum(int((m & c).sum()) for m, c in zip(valid, covisible)),
            "errors": np.concatenate(errors)}


def summarize(measurement):
    errors = measurement["errors"]
    result = {k: v for k, v in measurement.items() if k != "errors"}
    result.update(coverage_all=measurement["valid"] / measurement["pixels"],
                  coverage_covisible=(measurement["valid_covisible"] / measurement["covisible"]
                                      if measurement["covisible"] else None),
                  median_relative_error=float(np.median(errors)) if errors.size else None,
                  mean_relative_error=float(np.mean(errors, dtype=np.float64)) if errors.size else None)
    for threshold in (0.005, 0.01, 0.02, 0.05):
        result[f"fraction_below_{threshold}"] = float((errors < threshold).mean()) if errors.size else None
    return result


def table_row(scene, source, stats):
    keys = ["coverage_all", "coverage_covisible", "median_relative_error", "mean_relative_error",
            "fraction_below_0.005", "fraction_below_0.01", "fraction_below_0.02", "fraction_below_0.05"]
    cells = ["—" if stats[k] is None else f"{100 * stats[k]:.2f}" for k in keys]
    return "| " + " | ".join([scene, source, *cells]) + " |"


def percent(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/priors"))
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--report", type=Path, default=Path("docs/depth_precision.md"))
    parser.add_argument("--dataset", choices=SCENES, default="LLFF")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--exclude-background", action="store_true",
                        help="Restrict DTU coverage and errors to non-background training-image pixels")
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    args = parser.parse_args()
    if args.exclude_background and args.dataset != "DTU":
        parser.error("--exclude-background requires --dataset DTU")
    if args.scene and any(name not in SCENES[args.dataset] for name in args.scene):
        parser.error("--scene must belong to --dataset")
    started = time.perf_counter()
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(),
              "reference_root": str(args.reference_root), "dataset": args.dataset,
              "exclude_background": args.exclude_background, "scenes": {}, "skipped": {}, "pooled": {}}
    pooled = {source: [] for source in args.sources}
    rows = []
    for name in args.scene or SCENES[args.dataset]:
        scene = load_scene(args.dataset, name)
        paths = [args.reference_root / name / "dense/stereo/depth_maps" /
                 (v.path.name + ".geometric.bin") for v in scene.train]
        missing = [p.name for p in paths if not reference_ready(p)]
        if missing:
            report["skipped"][name] = "Missing/incomplete training-view references: " + ", ".join(missing)
            print(f"{name}: reference not ready", flush=True)
            continue
        references = [nearest(read_array(p), v.height, v.width) for p, v in zip(paths, scene.train)]
        covisible = covisible_masks(scene.train, references)
        regions = ([~dtu_background_mask(v.load(), name) for v in scene.train]
                   if args.exclude_background else None)
        if regions is not None:
            covisible = [c & region for c, region in zip(covisible, regions)]
        report["scenes"][name] = {"covisible_pixels": sum(int(c.sum()) for c in covisible),
                                  "pixels": (sum(c.size for c in covisible) if regions is None
                                             else sum(int(r.sum()) for r in regions)),
                                  "image_pixels": sum(c.size for c in covisible), "sources": {}}
        for source in args.sources:
            loaded = load_source(scene, source, args.output_root, args.reference_root)
            if loaded is None:
                report["skipped"][f"{name}/{source}"] = "Prior files not present"
                continue
            measure = measurements(*loaded, references, covisible, regions)
            stats = summarize(measure)
            report["scenes"][name]["sources"][source] = stats
            pooled[source].append(measure)
            rows.append(table_row(name, source, stats))
        print(f"{name}: evaluated; co-visible {sum(c.sum() for c in covisible):,} pixels", flush=True)
    for source, measures in pooled.items():
        if not measures:
            continue
        combined = {k: sum(m[k] for m in measures) for k in ("pixels", "valid", "covisible", "valid_covisible")}
        combined["errors"] = np.concatenate([m["errors"] for m in measures])
        stats = summarize(combined)
        stats["scene_count"] = len(measures)
        report["pooled"][source] = stats
        rows.append(table_row("pooled", source, stats))
    report["seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Depth precision of three-view priors", "", f"Evaluation time (UTC): {report['evaluated_at']}.", "",
             "All depths are camera-z in the supplied COLMAP coordinate system. Priors use only the three "
             "training views. All-view COLMAP geometric maps are analysis references only; no scale alignment is applied.", "",
             "Reference and comparison-map resizing uses nearest source pixel centres. For each training pixel with "
             "positive reference depth, co-visibility requires projection inside at least one other training view and "
             "agreement with its nearest reference depth within 2%. Projection uses `Camera.project` pixel-index coordinates.", "",
             "Coverage is reported over all benchmark pixels and over co-visible pixels. Errors and threshold fractions "
             "use co-visible pixels with a valid prior. All table entries are percentages. Pooled statistics concatenate "
             "pixels across the available scenes, rather than averaging scene statistics.", "",
             ("DTU background is excluded using `fewshot.data.dtu_background_mask` on each benchmark-resolution "
              "training image. Coverage all uses all non-background pixels as its denominator; co-visibility is "
              "computed from the original reference maps before intersecting the target foreground mask."
              if args.exclude_background else "No background mask is applied."), "",
             "| Scene | Source | Coverage all | Coverage co-visible | Median error | Mean error | <0.5% | <1% | <2% | <5% |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
             "Co-visible reference pixels:", ""]
    for name, stats in report["scenes"].items():
        lines.append(f"- {name}: {stats['covisible_pixels']:,} / {stats['pixels']:,} "
                     f"({100 * stats['covisible_pixels'] / stats['pixels']:.2f}%).")
    lines.extend(["", "Factual summary:", ""])
    for source, stats in report["pooled"].items():
        lines.append(f"- {source}: {stats['scene_count']} scenes, {100 * stats['coverage_all']:.2f}% overall coverage, "
                     f"{percent(stats['coverage_covisible'])} co-visible coverage; "
                     f"median/mean relative error {percent(stats['median_relative_error'])}/"
                     f"{percent(stats['mean_relative_error'])} on {stats['valid_covisible']:,} evaluated pixels.")
    if report["skipped"]:
        lines.extend(["", "Unavailable at evaluation time:", ""])
        lines.extend(f"- {name}: {reason}." for name, reason in report["skipped"].items())
    lines.extend(["", "Per-scene generation settings, coverage and timings are in each prior directory's `meta.json`. "
                  f"Machine-readable evaluation counts are in `{args.report.with_suffix('.json').name}`.", ""])
    args.report.write_text("\n".join(lines))
    write_json(args.report.with_suffix(".json"), report)
    print(f"Wrote {args.report} ({report['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
