"""Wall-clock timing of CertiSplat's prior construction (stereo, MVS, DA3, fusion), one process per stage
as in normal use (model loading included), written to a scratch prior root so experiments/priors is untouched.

    CUDA_VISIBLE_DEVICES=2 python scripts/time_priors.py --out /tmp/priors_timing LLFF:fern DTU:scan8
"""
import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ["bash", str(ROOT / "scripts/run.sh"), "tools"]
DA3_PYTHON = os.environ.get("DA3_PYTHON", sys.executable)
# Redirect the prior root (data.PRIORS, also imported by mvs_prior) before running a stage's main().
PATCH = ("import sys, runpy, pathlib; sys.path.insert(0, {root!r}); import fewshot.data as d; "
         "d.PRIORS = pathlib.Path({out!r}); import fewshot.mvs_prior as m; m.PRIORS = d.PRIORS; "
         "sys.argv = [{module!r}] + {argv!r}; runpy.run_module({module!r}, run_name='__main__')")


def stage(python, module, argv, out, env):
    code = PATCH.format(root=str(ROOT), out=str(out), module=module, argv=argv)
    started = time.time()
    subprocess.run([*python, "-c", code], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("targets", nargs="+", help="DATASET:scene")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    da3_env = env
    tools = ["bash", str(ROOT / "scripts/run.sh"), "tools"]
    results = []
    for target in args.targets:
        dataset, scene = target.split(":")
        for n in args.views:
            common = ["--dataset", dataset, "--n-views", str(n)]
            row = {"dataset": dataset, "scene": scene, "views": n,
                   "stereo": stage(tools, "fewshot.stereo_prior", [*common, "--scene", scene], args.out, env),
                   "mvs": stage(tools, "fewshot.mvs_prior", [*common, "--scene", scene], args.out, env),
                   "da3": stage([DA3_PYTHON], "fewshot.da3_prior", [*common, "--scenes", scene], args.out, da3_env),
                   "fuse": stage(tools, "fewshot.fuse", [*common, "--scenes", scene, "--name", "timing"], args.out, env)}
            row["total"] = sum(row[k] for k in ("stereo", "mvs", "da3", "fuse"))
            results.append(row)
            print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    (args.out / "timings.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
