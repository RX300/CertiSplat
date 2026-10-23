"""Run the unmodified pipeline() of StableGS/tools/colmap_llff.py or colmap_360.py (shared FSGS/DNGaussian
preprocessing) for selected scenes and view counts; produces <scene>/<n>_views/dense/fused.ply.

The upstream file ends with a hard-coded loop over the authors' dataset path, so only the code
before that loop is executed here.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DATA = os.environ.get("CERTISPLAT_DATA", str(ROOT / "datasets"))
TOOLS = {"llff": ("StableGS/tools/colmap_llff.py", f"{_DATA}/nerf_llff_data"),
         "360": ("StableGS/tools/colmap_360.py", f"{_DATA}/mipnerf360")}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--views", type=int, nargs="+", required=True)
parser.add_argument("--tool", choices=sorted(TOOLS), default="llff")
parser.add_argument("--data-root", help="default: the dataset of --tool")
args = parser.parse_args()
SOURCE = ROOT / TOOLS[args.tool][0]
args.data_root = args.data_root or TOOLS[args.tool][1]

code = SOURCE.read_text()
code = code[:code.rindex("\nfor scene in")]
namespace = {"__name__": "colmap_llff"}
exec(compile(code, str(SOURCE), "exec"), namespace)
for n in args.views:
    if (Path(args.data_root) / args.scene / f"{n}_views/dense/fused.ply").is_file():
        print(f"{args.scene} {n} views: fused.ply exists, skipped")
        continue
    namespace["pipeline"](args.scene, base_path=args.data_root.rstrip("/") + "/", n_views=n)
    print(f"{args.scene} {n} views: done")
