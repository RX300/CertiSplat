"""Run the official TWINGS initializer on one selected scene."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["nerf_llff_data", "mipnerf360", "DTU"], required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--views", type=int, required=True)
parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("CERTISPLAT_DATA", str(ROOT / "datasets"))))
args = parser.parse_args()
init_dir = ROOT / "TWINGS/TWINGS-Init"
sys.path.insert(0, str(init_dir))
os.chdir(init_dir)
from main import per_scene_triangulation

base = args.data_root / args.dataset
if args.dataset == "DTU":
    base = base / "dtu_corgs"
per_scene_triangulation(
    [args.scene], args.dataset, str(base), args.views,
    reproj_error_minimize=True,
    n_nearest_views=2 if args.views == 3 else 4,
    MIN_VIEWS_FOR_TRIANGULATION=2,
    CBPS_ratio=8 if args.dataset == "nerf_llff_data" else 32,
    resize_target_long_side=512,
    selection_mode="nearest", random_seed=42, visualize_tracks=False,
)
