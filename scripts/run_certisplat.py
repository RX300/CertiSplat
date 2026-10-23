"""CertiSplat end to end for whole settings: priors (stereo, MVS, DA3), fusion, training + evaluation.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_certisplat.py LLFF:3 DTU:3 --replace
    CUDA_VISIBLE_DEVICES=0 python scripts/run_certisplat.py LLFF:3 --scenes fern --replace

Same recipe as LLFF/DTU at 6/9 views: fusion without propagation (fused_np), training defaults.
Existing priors are reused. --replace retrains in pending directories and replaces each old run
only after training, evaluation and artifact checks succeed. Failed jobs stop the sequence.
Results: experiments/dev/certisplat/<dataset>[_<n>v]/<scene>/seed0/ (no suffix at 3 views);
logs: experiments/dev/certisplat/logs/.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fewshot.data import SCENES, load_scene, prior_dir  # noqa: E402

TOOLS = ["bash", str(ROOT / "scripts/run.sh"), "tools"]
DA3 = [os.environ.get("DA3_PYTHON", sys.executable)]   # Python of the Depth Anything 3 environment
LOGS = ROOT / "experiments/dev/certisplat/logs"


def done(dataset, scene, n, source):
    folder = prior_dir(dataset, scene, n, source)
    return all((folder / f"{v.name}.npz").is_file() for v in load_scene(dataset, scene, n).train)


def call(argv, log, env):
    print(time.strftime("%H:%M:%S"), " ".join(argv[-8:]), flush=True)
    with open(log, "a") as stream:
        stream.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(argv) + "\n")
        stream.flush()
        subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", nargs="+", help="DATASET:VIEWS")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--replace", action="store_true", help="replace completed training results after validating new runs")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    gpu = env.get("CUDA_VISIBLE_DEVICES", "0")
    for setting in args.settings:
        dataset, n = setting.split(":")
        scenes = args.scenes or SCENES[dataset]
        log = LOGS / f"{dataset}_{n}v.log"
        for scene in scenes:
            common = ["--dataset", dataset, "--n-views", n]
            if not done(dataset, scene, int(n), "stereo"):
                call(TOOLS + ["-m", "fewshot.stereo_prior", *common, "--scene", scene], log, env)
            if not done(dataset, scene, int(n), "mvs"):
                call(TOOLS + ["-m", "fewshot.mvs_prior", *common, "--scene", scene], log, env)
            if not done(dataset, scene, int(n), "da3"):
                call(DA3 + ["-m", "fewshot.da3_prior", *common, "--scenes", scene], log, env)
            if not (prior_dir(dataset, scene, int(n), "fused_np") / "init_points.npz").is_file():
                call(TOOLS + ["-m", "fewshot.fuse", *common, "--scenes", scene, "--name", "fused_np"], log, env)
        call([sys.executable, str(ROOT / "scripts/sweep.py"), "run", "--name", "certisplat", "--dataset", dataset,
              "--gpu", gpu, "--n-views", n, "--scenes", *scenes, *(["--replace"] if args.replace else []), "--",
              "--init", "npz:experiments/priors/" + dataset + "/{views}v/{scene}/fused_np/init_points.npz"], log, env)


if __name__ == "__main__":
    main()
