"""Run one training configuration over benchmark scenes and tabulate it against the baselines.

    python scripts/sweep.py run  --name base --dataset LLFF --gpu 0 -- --init stereogs --dropout 0.3
    python scripts/sweep.py run  --name x --dataset DTU --gpu 0 --n-views 6 -- \
        --init npz:experiments/priors/DTU/{views}v/{scene}/fused_np/init_points.npz
("{scene}" and "{views}" in training arguments are substituted per run.)
    python scripts/sweep.py table --root experiments/dev --dataset LLFF base other
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fewshot.data import SCENES  # noqa: E402

BASELINES = ROOT / "benchmarks/first_pass/per_scene.csv"


def run(args, train_args):
    scenes = args.scenes or SCENES[args.dataset]
    for seed in args.seeds:
        for scene in scenes:
            out = Path(args.root) / args.name / args.dataset / scene / f"seed{seed}"
            if args.n_views != 3:
                out = Path(args.root) / args.name / f"{args.dataset}_{args.n_views}v" / scene / f"seed{seed}"
            if not args.replace and (out / "metrics.json").exists():
                continue
            work = out.with_name("." + out.name + ".pending") if args.replace else out
            if work.exists():
                raise FileExistsError(f"Incomplete or running output: {work}")
            cmd = ["bash", str(ROOT / "scripts/run.sh"), "tools", "-m", "fewshot.train",
                   "--dataset", args.dataset, "--scene", scene, "--seed", str(seed),
                   "--n-views", str(args.n_views), "--output", str(work),
                   *[a.replace("{scene}", scene).replace("{views}", str(args.n_views)) for a in train_args]]
            work.mkdir(parents=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            with open(work / "stdout.log", "w") as log:
                log.write("command: " + " ".join(cmd) + "\n")
                log.flush()
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            metrics = json.loads((work / "metrics.json").read_text())
            views = json.loads((work / "per_view.json").read_text())
            config = json.loads((work / "config.json").read_text())
            last = json.loads((work / "train_log.jsonl").read_text().splitlines()[-1])
            assert last["step"] == config["iterations"], work
            assert (work / "point_cloud.ply").stat().st_size > 0, work
            for key in ("PSNR", "SSIM", "LPIPS"):
                assert math.isfinite(metrics[key]), (work, key)
                assert abs(sum(v[key] for v in views) / len(views) - metrics[key]) < 1e-7, (work, key)
            assert all((work / "test/renders" / f"{v['view']}.png").is_file() for v in views), work
            if args.replace:
                config["output"] = str(out)
                (work / "config.json").write_text(json.dumps(config, indent=2))
                if out.exists():
                    shutil.rmtree(out)
                work.rename(out)
            print(f"{args.name} {args.dataset}/{scene} seed{seed}: ok", flush=True)


def baseline_scores(dataset):
    scores = {}
    for row in csv.DictReader(open(BASELINES, encoding="utf-8-sig")):
        if row["dataset"] == dataset and row["status"] == "completed":
            scores.setdefault(row["method"], {})[row["scene"]] = (
                float(row["PSNR"]), float(row["SSIM"]), float(row["LPIPS"]))
    return scores


def collect(root, name, dataset):
    scores = {}
    for path in sorted((Path(root) / name / dataset).glob("*/seed*/metrics.json")):
        m = json.loads(path.read_text())
        scores.setdefault(m["scene"], []).append((m["PSNR"], m["SSIM"], m["LPIPS"]))
    return {scene: tuple(sum(v[i] for v in values) / len(values) for i in range(3))
            for scene, values in scores.items()}


def table(args):
    scenes = SCENES[args.dataset]
    columns = {k: v for k, v in baseline_scores(args.dataset).items()}
    for name in args.names:
        columns[name] = collect(args.root, name, args.dataset)
    print(f"{'scene':10s}" + "".join(f"{k[:18]:>20s}" for k in columns))
    for scene in scenes:
        cells = []
        for values in columns.values():
            cells.append(f"{values[scene][0]:20.2f}" if scene in values else f"{'-':>20s}")
        print(f"{scene:10s}" + "".join(cells))
    line = []
    for values in columns.values():
        done = [s for s in scenes if s in values]
        if len(done) == len(scenes):
            mean = [sum(values[s][i] for s in scenes) / len(scenes) for i in range(3)]
            line.append(f"{mean[0]:7.2f}/{mean[1]:.3f}/{mean[2]:.3f}".rjust(20))
        else:
            line.append(f"({len(done)}/{len(scenes)})".rjust(20))
    print(f"{'MEAN':10s}" + "".join(line))


def main():
    argv = sys.argv[1:]
    train_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, train_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--name", required=True)
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--gpu", type=int, required=True)
    p_run.add_argument("--scenes", nargs="*")
    p_run.add_argument("--seeds", type=int, nargs="+", default=[0])
    p_run.add_argument("--n-views", type=int, default=3)
    p_run.add_argument("--replace", action="store_true", help="train in a pending directory; replace the old run only after successful evaluation")
    p_run.add_argument("--root", default=str(ROOT / "experiments/dev"))
    p_table = sub.add_parser("table")
    p_table.add_argument("names", nargs="*")
    p_table.add_argument("--dataset", required=True)
    p_table.add_argument("--root", default=str(ROOT / "experiments/dev"))
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args, train_args)
    else:
        table(args)


if __name__ == "__main__":
    main()
