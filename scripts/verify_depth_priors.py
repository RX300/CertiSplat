"""Read generated priors back and measure camera and cross-view reprojection consistency."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewshot.camera import Camera, sample_image
from fewshot.data import SCENES, load_scene
from fewshot.prior_io import write_json


@torch.no_grad()
def verify_scene(scene, source, root):
    directory = root / scene.dataset / f"{len(scene.train)}v" / scene.name / source
    meta = json.loads((directory / "meta.json").read_text())
    cameras = [Camera.from_view(v, with_image=False, device="cpu") for v in scene.train]
    depths, masks, view_stats = [], [], {}
    for view, camera in zip(scene.train, cameras):
        with np.load(directory / f"{view.name}.npz") as data:
            depth, valid, conf = data["depth"], data["valid"], data["conf"]
        assert depth.shape == valid.shape == conf.shape == (view.height, view.width)
        assert depth.dtype == conf.dtype == np.float32 and valid.dtype == bool
        assert np.isfinite(depth).all() and np.isfinite(conf).all()
        assert np.all(depth[~valid] == 0) and np.all(depth[valid] > 0)
        assert np.all((conf >= 0) & (conf <= 1)) and np.all(conf[~valid] == 0)
        assert np.isin(conf[valid], [0.5, 1] if source == "stereo" else [1]).all()
        tensor = torch.from_numpy(depth)
        mask = torch.from_numpy(valid)
        u, v, z = camera.project(camera.backproject(tensor))
        y, x = np.indices(depth.shape)
        pixel_error = np.hypot(u.numpy()[valid] - x[valid], v.numpy()[valid] - y[valid])
        max_error = float(pixel_error.max()) if pixel_error.size else None
        if pixel_error.size:
            assert max_error < 0.01, (scene.name, view.name, max_error)
        np.testing.assert_allclose(z.numpy()[valid], depth[valid], rtol=1e-4, atol=1e-5)
        depths.append(tensor)
        masks.append(mask)
        view_stats[view.name] = {"coverage": float(valid.mean()),
                                 "depth_min": float(depth[valid].min()) if valid.any() else None,
                                 "depth_max": float(depth[valid].max()) if valid.any() else None,
                                 "max_camera_roundtrip_pixels": max_error}
    crossview = []
    for i, j in permutations(range(len(scene.train)), 2):
        target = cameras[j]
        u, v, z = target.project(cameras[i].backproject(depths[i]))
        inside = (u >= 0) & (u <= target.width - 1) & (v >= 0) & (v <= target.height - 1) & (z > 0)
        seen = sample_image(depths[j][None], u, v, mode="nearest")[0]
        overlap = masks[i] & inside & (seen > 0)
        errors = ((z[overlap] - seen[overlap]).abs() / seen[overlap]).numpy()
        crossview.append({"from": scene.train[i].name, "to": scene.train[j].name,
                          "overlap_pixels": int(overlap.sum()),
                          "median_relative_error": float(np.median(errors)) if errors.size else None,
                          "fraction_below_0.02": float((errors < 0.02).mean()) if errors.size else None})
    rectification = []
    if source == "stereo":
        view_by_name = {v.name: v for v in scene.train}
        depth_by_name = {v.name: d.numpy() for v, d in zip(scene.train, depths)}
        for pair in meta["pairs"]:
            if pair["status"] == "skipped":
                rectification.append({"names": pair["names"], "skipped": pair["reason"]})
                continue
            left, right = [view_by_name[n] for n in pair["names"]]
            d = depth_by_name[left.name]
            y, x = np.nonzero(d > 0)
            y, x = y[::max(1, len(y) // 2048)], x[::max(1, len(x) // 2048)]
            if not len(y):
                continue
            rays = np.stack([x + 0.5, y + 0.5, np.ones_like(x)], -1) @ np.linalg.inv(left.K.astype(float)).T
            local = rays * d[y, x, None]
            R1, R2 = map(np.asarray, pair["rotations"])
            P1, P2 = map(np.asarray, pair["projections"])
            rectified = local @ R1.T
            homogeneous = np.column_stack([rectified, np.ones(len(rectified))])
            p1, p2 = homogeneous @ P1.T, homogeneous @ P2.T
            uv1, uv2 = p1[:, :2] / p1[:, 2:], p2[:, :2] / p2[:, 2:]
            recovered = cv2.triangulatePoints(P1, P2, uv1.T, uv2.T)
            recovered = (recovered[:3] / recovered[3]).T @ R1
            rel = np.abs(recovered[:, 2] - local[:, 2]) / local[:, 2]
            # Compare the rectified target projection against its original camera.
            relative = right.w2c.astype(float) @ np.linalg.inv(left.w2c.astype(float))
            right_local = local @ relative[:3, :3].T + relative[:3, 3]
            expected = (right_local @ R2.T) @ P2[:, :3].T
            expected = expected[:, :2] / expected[:, 2:]
            projection_error = float(np.max(np.linalg.norm(uv2 - expected, axis=1)))
            assert projection_error < 0.01
            assert float(rel.max()) < 1e-6
            rectification.append({"names": pair["names"],
                                  "max_target_projection_pixels": projection_error,
                                  "max_triangulation_relative_error": float(rel.max()),
                                  "max_epipolar_pixels": float(np.abs(uv1[:, 1] - uv2[:, 1]).max())})
    return {"seconds": meta["timings_seconds"]["total"],
            "long_side": meta["long_side"] if source == "stereo" else None,
            "views": view_stats,
            "crossview": crossview, "rectification": rectification}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/priors"))
    parser.add_argument("--report", type=Path, default=Path("docs/depth_prior_validation.json"))
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly skip unfinished scenes during a running batch")
    args = parser.parse_args()
    report = {"scenes": {}, "missing": [], "npz_count": 0}
    for dataset, names in SCENES.items():
        for name in names:
            scene = load_scene(dataset, name, args.n_views)
            for source in ("stereo", "mvs"):
                key = f"{dataset}/{args.n_views}v/{name}/{source}"
                if args.allow_incomplete and not (args.output_root / key / "meta.json").is_file():
                    report["missing"].append(key)
                    continue
                report["scenes"][key] = verify_scene(scene, source, args.output_root)
                report["npz_count"] += len(scene.train)
                print(f"Verified {key}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    rows = ["# Depth prior validation", "", f"Read back and checked {report['npz_count']} NPZ files.", "",
            "Checks: exact benchmark shape and dtypes; finite positive camera-z where valid; zero depth and confidence "
            "elsewhere; confidence range; camera backprojection/projection round trip below 0.01 pixels. "
            "Stereo rectification also checks target-camera projection and triangulation against actual prior samples.", "",
            "Cross-view measurements below pool directed view pairs. They use positive priors in both views and "
            "nearest target sampling, without a reference occlusion mask. They include occlusions and disagreements; "
            "the <2% column is a measured consistency rate, not a correctness guarantee.", "",
            "| Dataset/scene | Source | Seconds | Mean coverage (%) | Cross-view overlap pixels | Within 2% (%) |",
            "|---|---|---:|---:|---:|---:|"]
    skipped = []
    for key, stats in report["scenes"].items():
        dataset, count, name, source = key.split("/")
        coverage = np.mean([s["coverage"] for s in stats["views"].values()])
        count = sum(s["overlap_pixels"] for s in stats["crossview"])
        agreed = sum(s["overlap_pixels"] * s["fraction_below_0.02"] for s in stats["crossview"] if s["overlap_pixels"])
        fraction = f"{100 * agreed / count:.2f}" if count else "—"
        rows.append(f"| {dataset}/{count}/{name} | {source} | {stats['seconds']:.1f} | {100 * coverage:.2f} | {count:,} | {fraction} |")
        skipped.extend(f"- {key}: {' / '.join(p['names'])}: {p['skipped']}."
                       for p in stats["rectification"] if "skipped" in p)
    rows.extend(["", "Timings are wall time per scene, including image loading, inference/COLMAP, and output writing. "
                 "Stereo model loading is excluded here and recorded separately in `meta.json`. "
                 "Concurrent batches share GPU 0, so summed times are not elapsed batch time.", ""])
    for dataset in SCENES:
        for source in ("stereo", "mvs"):
            times = [s["seconds"] for key, s in report["scenes"].items()
                     if key.startswith(dataset + "/") and key.endswith("/" + source)]
            rows.append(f"- {dataset}/{source}: {len(times)} scenes; summed scene time {sum(times):.1f}s "
                        f"({sum(times) / 60:.2f} min).")
    if skipped:
        rows.extend(["Skipped degenerate stereo pairs:", "", *skipped, ""])
    if report["missing"]:
        rows.extend(["Unfinished scenes:", "", *[f"- {s}" for s in report["missing"]], ""])
    args.report.with_suffix(".md").write_text("\n".join(rows))
    print(f"Verified {report['npz_count']} NPZs; wrote {args.report}")


if __name__ == "__main__":
    main()
