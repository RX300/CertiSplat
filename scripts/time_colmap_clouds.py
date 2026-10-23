"""Wall-clock time of the sparse-view COLMAP dense clouds that StableGS and TWINGS start from
(<scene>/<n>_views/dense/fused.ply; downloaded for LLFF 3-view and DTU, so absent from their benchmark timings).

Runs the unmodified pipeline() of StableGS/tools/colmap_llff.py or StableGS/scripts/colmap_dtu.py on a scratch
copy of each scene (the upstream code starts with `rm -r <n>_views`). Needs the COLMAP shim on PATH
(runtimes/colmap_shim), as for scripts/colmap_llff_views.py.
"""
import argparse
import json
import shutil
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DATA = Path(os.environ.get("CERTISPLAT_DATA", str(ROOT / "datasets")))
DATA = {"LLFF": _DATA / "nerf_llff_data", "DTU": _DATA / "DTU/dtu_corgs"}
SOURCE = {"LLFF": ROOT / "StableGS/tools/colmap_llff.py", "DTU": ROOT / "StableGS/scripts/colmap_dtu.py"}


def pipeline(dataset):
    code = SOURCE[dataset].read_text()
    code = code[:code.rindex("\nfor scene in")]
    namespace = {"__name__": "colmap_" + dataset}
    exec(compile(code, str(SOURCE[dataset]), "exec"), namespace)
    return namespace["pipeline"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("targets", nargs="+", help="DATASET:scene")
    args = parser.parse_args()
    results = []
    for target in args.targets:
        dataset, scene = target.split(":")
        base = args.work / dataset
        copy = base / scene
        if not copy.exists():
            copy.mkdir(parents=True)
            shutil.copytree(DATA[dataset] / scene / "images", copy / "images")
            shutil.copytree(DATA[dataset] / scene / "sparse", copy / "sparse")
        for n in args.views:
            started = time.time()
            pipeline(dataset)(scene, base_path=str(base) + "/", n_views=n)
            seconds = time.time() - started
            ply = copy / f"{n}_views/dense/fused.ply"
            header = ply.read_bytes()[:400].decode("latin1") if ply.exists() else ""
            points = int(header.split("element vertex ")[1].split()[0]) if "element vertex" in header else 0
            results.append({"dataset": dataset, "scene": scene, "views": n, "seconds": seconds, "points": points})
            print(json.dumps(results[-1]), flush=True)
    (args.work / "colmap_timings.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
