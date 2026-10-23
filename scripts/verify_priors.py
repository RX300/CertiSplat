"""Check every depth prior used by the method and measure cross-view consistency of certified depth.

Certified = confidence 1 in fused_np, the same fusion at 3/6/9 views on LLFF and DTU.

    bash scripts/run.sh tools scripts/verify_priors.py --json docs/priors_check.json > docs/PRIORS_CHECK.md
Integrity: benchmark shape, finite values, positive depth exactly where valid. Consistency: certified
pixels (confidence 1) of each training view are projected into every other training view; the
relative camera-z difference to that view's certified depth (nearest pixel) is pooled per setting.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewshot.data import SCENES, load_scene, prior_dir  # noqa: E402

SOURCES = ("stereo", "mvs", "da3", "fused_np")


def load(path, view):
    with np.load(path) as data:
        depth, valid = data["depth"], data["valid"]
        conf = data["conf"] if "conf" in data.files else np.ones_like(depth)
    assert depth.shape == valid.shape == conf.shape == (view.height, view.width), path
    assert np.isfinite(depth).all() and np.isfinite(conf).all(), path
    if path.parent.name == "fused_np":
        assert np.isin(conf, np.array([0.2, 0.7, 1.0], dtype=conf.dtype)).all(), path
    assert np.array_equal(valid, depth > 0), path
    return depth, conf


def consistency(scene, maps):
    errors = []
    for i, (vi, (di, ci)) in enumerate(zip(scene.train, maps)):
        y, x = np.nonzero(ci == 1)
        rays = np.column_stack([x + .5, y + .5, np.ones(len(x))]) @ np.linalg.inv(vi.K.astype(float)).T
        c2w = np.linalg.inv(vi.w2c.astype(float))
        points = (rays * di[y, x, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        for j, (vj, (dj, cj)) in enumerate(zip(scene.train, maps)):
            if i == j:
                continue
            q = points @ vj.w2c[:3, :3].astype(float).T + vj.w2c[:3, 3]
            z = q[:, 2]
            p = q @ vj.K.astype(float).T
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = np.floor(p[:, 0] / z).astype(np.int64), np.floor(p[:, 1] / z).astype(np.int64)
            inside = (z > 0) & (u >= 0) & (v >= 0) & (u < vj.width) & (v < vj.height)
            u, v, z = u[inside], v[inside], z[inside]
            both = cj[v, u] == 1
            errors.append(np.abs(z[both] - dj[v, u][both]) / dj[v, u][both])
    e = np.concatenate(errors) if errors else np.zeros(0)
    return float(np.median(e)) if len(e) else float("nan"), float((e < .01).mean()) if len(e) else float("nan"), len(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the exact measurements used by the paper tables")
    args = parser.parse_args()
    rows = []
    print("# Depth-prior check\n")
    print("All 3/6/9-view settings use fused_np without cross-view propagation; raw stereo, MVS and DA3 are also checked.\n")
    print("| Dataset | views | scenes | files checked | certified pixels | weighted scene-median rel. diff | < 1% |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in ("LLFF", "DTU"):
        for n in (3, 6, 9):
            files, certified, med, below, weights = 0, [], [], [], []
            for name in SCENES[dataset]:
                scene = load_scene(dataset, name, n)
                for source in SOURCES:
                    folder = prior_dir(dataset, name, n, source)
                    maps = [load(folder / f"{v.name}.npz", v) for v in scene.train]
                    files += len(maps)
                    if source == "fused_np":
                        certified.append(np.mean([(c == 1).mean() for _, c in maps]))
                        m, b, k = consistency(scene, maps)
                        med.append(m); below.append(b); weights.append(k)
            w = np.array(weights, float)
            rows.append({"dataset": dataset, "views": n, "scenes": len(SCENES[dataset]), "files_checked": files,
                         "certified_fraction": float(np.mean(certified)),
                         "cross_view_relative_difference": float(np.average(med, weights=w)),
                         "cross_view_below_one_percent": float(np.average(below, weights=w))})
            print(f"| {dataset} | {n} | {len(SCENES[dataset])} | {files} | {100 * np.mean(certified):.1f}% | "
                  f"{100 * np.average(med, weights=w):.2f}% | {100 * np.average(below, weights=w):.1f}% |", flush=True)
    print("\nCertified fractions average equally over scenes and training views. Cross-view statistics compare "
          "projected certified source depths with certified depths at the nearest target pixel; the reported "
          "difference is a count-weighted mean of per-scene medians, and the < 1% rate uses the same projected "
          "pixel-pair counts. These are internal consistency measurements, not errors against reference geometry.")
    if args.json:
        args.json.write_text(json.dumps({"source": "fused_np", "aggregation": "scene/view-mean certified fractions; "
            "projected-pixel-pair-count-weighted scene medians and below-threshold fractions", "settings": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
