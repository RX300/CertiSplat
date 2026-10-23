"""Benchmark table: CertiSplat runs against the baselines' reported (paper) numbers.

    python scripts/report.py > docs/RESULTS.md
Baselines are compared by their paper numbers (decided 2026-10-21; local runs stopped). CertiSplat runs are read
from experiments/dev/<run>/<dataset>[_<n>v]/<scene>/seed0/metrics.json. The local baseline runs that completed
(benchmarks/{first_pass,views_69}/per_scene.csv) are listed only as a reproduction check of the paper numbers.
A setting is reported only when every scene of the dataset is present.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fewshot.data import SCENES  # noqa: E402

KEYS = ("PSNR", "SSIM", "LPIPS")
# Final runs: one recipe (training defaults) for every dataset and view count, all under experiments/dev/certisplat
# (the LLFF 6/9-view runs were trained as "llff_rec" with the same settings; see experiments/dev/README.md).
VIEWS = {"LLFF": (3, 6, 9), "DTU": (3, 6, 9), "Mip-NeRF360": (12, 24), "Blender": (8,)}
OURS = {(dataset, n): "certisplat" for dataset, counts in VIEWS.items() for n in counts}
# Reported by the papers, re-read from the arXiv HTML (docs/BASELINE_AUDIT.md): StereoGS v2 Tables 1-4 ("Ours*",
# fixed dropout 0.3), TWINGS Tables 1-3, ICO-GS Tables 1-3 (mean of three seeds).
REPORTED = {
    "StereoGS + dropout (ECCV'26)": {("LLFF", 3): (22.05, .783, .147), ("LLFF", 6): (25.40, .856, .102),
                                     ("LLFF", 9): (26.44, .883, .082), ("DTU", 3): (22.00, .890, .095),
                                     ("DTU", 6): (25.41, .938, .063), ("DTU", 9): (27.39, .955, .047),
                                     ("Mip-NeRF360", 12): (20.51, .625, .309),
                                     ("Mip-NeRF360", 24): (24.25, .783, .206), ("Blender", 8): (25.04, .899, .078)},
    "TWINGS (CVPR'26)": {("LLFF", 3): (21.49, .754, .167), ("LLFF", 6): (24.97, .844, .106),
                         ("LLFF", 9): (26.23, .876, .085), ("DTU", 3): (21.52, .880, .107),
                         ("DTU", 6): (25.81, .930, .063), ("DTU", 9): (28.22, .958, .040),
                         ("Mip-NeRF360", 12): (20.35, .618, .368), ("Mip-NeRF360", 24): (24.17, .762, .252)},
    "ICO-GS (CVPR'26)": {("LLFF", 3): (22.20, .778, .157), ("LLFF", 6): (25.37, .856, .109),
                         ("LLFF", 9): (26.45, .881, .096), ("DTU", 3): (21.77, .888, .092),
                         ("DTU", 6): (25.09, .928, .064), ("DTU", 9): (27.19, .953, .045),
                         ("Blender", 8): (25.56, .884, .100)},
}
# Reported numbers whose protocol differs from ours; shown, but excluded from "best reported".
NOT_COMPARABLE = {("TWINGS (CVPR'26)", "Mip-NeRF360"): "TWINGS evaluates Mip-NeRF360 at 1/4 resolution (ours 1/8)"}
LOCAL = {"StereoGS": "StereoGS + dropout (ECCV'26)", "TWINGS": "TWINGS (CVPR'26)"}   # local run -> paper row


def local_baselines(dataset):
    """{method: {views: mean}} over complete scene sets from benchmarks/{first_pass,views_69}."""
    out = {}
    for bench in ("first_pass", "views_69"):
        path = ROOT / "benchmarks" / bench / "per_scene.csv"
        if not path.exists():
            continue
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            if row["dataset"] == dataset and row["status"] == "completed" and row["method"] in LOCAL:
                out.setdefault(row["method"], {}).setdefault(int(row["views"]), {})[row["scene"]] = \
                    [float(row[k]) for k in KEYS]
    return {m: {n: np.mean([v[s] for s in SCENES[dataset]], axis=0) for n, v in by_views.items()
                if all(s in v for s in SCENES[dataset])} for m, by_views in out.items()}


def ours(dataset, views):
    folder = dataset if views == 3 else f"{dataset}_{views}v"   # scripts/sweep.py naming
    base = ROOT / "experiments/dev" / OURS[(dataset, views)] / folder
    values = []
    for scene in SCENES[dataset]:
        path = base / scene / "seed0/metrics.json"
        if not path.exists():
            return None
        m = json.loads(path.read_text())
        values.append([m[k] for k in KEYS])
    return np.mean(values, axis=0)


def best_reported(dataset, views):
    rows = [t[(dataset, views)] for name, t in REPORTED.items()
            if (dataset, views) in t and (name, dataset) not in NOT_COMPARABLE]
    return (max(r[0] for r in rows), max(r[1] for r in rows), min(r[2] for r in rows)) if rows else None


def fmt(v, missing="–"):
    return f"{v[0]:.2f} / {v[1]:.3f} / {v[2]:.3f}" if v is not None else missing


def delta(a, b):
    return f"{a[0] - b[0]:+.2f} / {a[1] - b[1]:+.3f} / {a[2] - b[2]:+.3f}" if a is not None and b is not None else "–"


def main():
    print("# Results (PSNR / SSIM / LPIPS)\n")
    print("Baselines: numbers reported by their papers. CertiSplat: single seed, StereoGS official evaluator.\n")
    for dataset, counts in VIEWS.items():
        print(f"## {dataset}\n")
        print("| Method | " + " | ".join(f"{n} views" for n in counts) + " |")
        print("|---" * (len(counts) + 1) + "|")
        notes = []
        for name, table in REPORTED.items():
            if any((dataset, n) in table for n in counts):
                mark = ""
                if (name, dataset) in NOT_COMPARABLE:
                    notes.append(NOT_COMPARABLE[(name, dataset)])
                    mark = "†" * len(notes)
                print(f"| {name}, reported{mark} | " + " | ".join(fmt(table.get((dataset, n))) for n in counts) + " |")
        mine = {n: ours(dataset, n) for n in counts}
        print("| **CertiSplat** | " + " | ".join(fmt(mine[n], "not run") for n in counts) + " |")
        print("| Δ vs best reported (per metric) | "
              + " | ".join(delta(mine[n], best_reported(dataset, n)) for n in counts) + " |\n")
        for k, note in enumerate(notes, 1):
            print(f"{'†' * k} {note}; excluded from the best-reported row.\n")

    print("## Reproduction check: local runs of the official code against the paper numbers\n")
    print("Same splits and bit-identical test GT as CertiSplat (docs/BASELINE_AUDIT.md); complete scene sets only.\n")
    print("| Method | Setting | Local run | Reported | Δ PSNR |")
    print("|---|---|---|---|---:|")
    for dataset, counts in VIEWS.items():
        for method, table in sorted(local_baselines(dataset).items()):
            for n in counts:
                if n in table:
                    paper = REPORTED[LOCAL[method]].get((dataset, n))
                    print(f"| {method} | {dataset} {n}v | {fmt(table[n])} | {fmt(paper)} | "
                          f"{table[n][0] - paper[0]:+.2f} |")


if __name__ == "__main__":
    main()
