"""Check that completed baseline runs used CertiSplat's split and test GT (CPU only).

For every completed job of a benchmark (StableGS excluded unless --methods names it):
* cameras.json lists the test cameras, then the training cameras; their names, in order, must equal
  fewshot.data.load_scene(dataset, scene, n).test / .train, and their centres and rotations must match
  CertiSplat's poses;
* every test GT PNG the official evaluator scored (evaluation/<method>/<dataset>_<n>v/<scene>/test/ours_<it>/gt)
  must equal CertiSplat's GT (View.load(), quantised to 8 bits) pixel for pixel.

Usage: CUDA_VISIBLE_DEVICES= bash scripts/run.sh tools scripts/verify_baselines.py --bench views_69
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fewshot.data import load_scene  # noqa: E402


def folder(dataset, views):
    return f"{dataset}_{views}v"


def check(bench, job):
    scene = load_scene(job["dataset"], job["scene"], job["views"])
    problems = []
    cameras = json.loads((Path(job["output"]) / "cameras.json").read_text())
    names = [c["img_name"] for c in cameras]
    n_test = len(scene.test)
    if names[:n_test] != [v.name for v in scene.test]:
        problems.append(f"test cameras {names[:n_test]} != {[v.name for v in scene.test]}")
    if names[n_test:] != [v.name for v in scene.train]:
        problems.append(f"train cameras {names[n_test:]} != {[v.name for v in scene.train]}")
    pose_error = 0.0
    by_name = {v.name: v for v in scene.views}
    for camera in cameras:
        view = by_name.get(camera["img_name"])
        if view is None:
            problems.append(f"unknown camera {camera['img_name']}")
            continue
        c2w = np.linalg.inv(view.w2c.astype(np.float64))
        pose_error = max(pose_error, float(np.abs(np.array(camera["position"]) - c2w[:3, 3]).max()),
                         float(np.abs(np.array(camera["rotation"]) - c2w[:3, :3]).max()))
    if pose_error > 1e-4:
        problems.append(f"pose mismatch up to {pose_error:.2e}")

    gt_dir = (bench / "evaluation" / job["method"] / folder(job["dataset"], job["views"]) / job["scene"]
              / "test" / f"ours_{job['iteration']}" / "gt")
    files = sorted(gt_dir.glob("*.png"))
    if len(files) != n_test:
        problems.append(f"{len(files)} GT files, expected {n_test}")
    max_diff = 0
    for path, view in zip(files, scene.test):
        with Image.open(path) as image:
            theirs = np.asarray(image.convert("RGB"), dtype=np.int16)
        ours = np.round(view.load() * 255).astype(np.int16)
        if theirs.shape != ours.shape:
            problems.append(f"{path.name}: shape {theirs.shape} != {ours.shape}")
            continue
        max_diff = max(max_diff, int(np.abs(theirs - ours).max()))
    if max_diff:
        problems.append(f"GT max pixel difference {max_diff}")
    return {"method": job["method"], "dataset": job["dataset"], "views": job["views"], "scene": job["scene"],
            "train": [v.name for v in scene.train], "test_views": len(files), "pose_error": pose_error,
            "gt_max_diff": max_diff, "ok": not problems, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True)
    parser.add_argument("--methods", nargs="+", default=["TWINGS", "StereoGS"])
    args = parser.parse_args()
    bench = ROOT / "benchmarks" / args.bench
    jobs = json.loads((bench / "manifest.json").read_text())["jobs"]
    states = json.loads((bench / "status.json").read_text())["jobs"]
    results = [check(bench, job) for job, state in zip(jobs, states)
               if job["method"] in args.methods and state["status"] == "completed"]
    for r in results:
        status = "ok" if r["ok"] else "FAIL " + "; ".join(r["problems"])
        print(f"{r['method']:9s} {folder(r['dataset'], r['views']):8s} {r['scene']:9s} "
              f"pose {r['pose_error']:.1e} gt {r['gt_max_diff']}  {status}")
    failed = [r for r in results if not r["ok"]]
    print(f"{len(results)} runs checked, {len(failed)} failed")
    (bench / "split_gt_check.json").write_text(json.dumps(results, indent=2) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
