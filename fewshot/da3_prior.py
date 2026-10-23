"""Complete (dense) depth prior: DA3-GIANT-1.1 on the training views with known cameras.

Runs in the separate foundation-model environment:
    CUDA_VISIBLE_DEVICES=0 $DA3_PYTHON \
        -m fewshot.da3_prior --dataset DTU
Writes experiments/priors/<dataset>/<n>v/<scene>/da3/<view>.npz (depth, valid, conf) at benchmark size.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
DA3_SOURCE = Path(os.environ.get("DA3_SOURCE", _ROOT / "third_party/Depth-Anything-3/src"))
DA3_WEIGHTS = Path(os.environ.get("DA3_WEIGHTS", _ROOT / "checkpoints/DA3-GIANT-1.1"))
sys.path.insert(0, str(DA3_SOURCE))

from .data import load_scene, prior_dir, SCENES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(str(DA3_WEIGHTS)).eval().cuda()
    for scene_name in args.scenes or SCENES[args.dataset]:
        scene = load_scene(args.dataset, scene_name, args.n_views)
        views = scene.train
        started = time.time()
        with torch.inference_mode():
            prediction = model.inference(
                image=[np.rint(v.load() * 255).astype(np.uint8) for v in views],
                extrinsics=np.stack([v.w2c for v in views]), intrinsics=np.stack([v.K for v in views]),
                align_to_input_ext_scale=True, process_res=args.process_res,
                process_res_method="upper_bound_resize", infer_gs=False, ref_view_strategy="first",
                export_dir=None)
        out = prior_dir(args.dataset, scene_name, args.n_views, "da3")
        out.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(views):
            depth = torch.tensor(prediction.depth[i], dtype=torch.float32)[None, None]
            depth = F.interpolate(depth, size=(view.height, view.width), mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
            conf = torch.tensor(prediction.conf[i], dtype=torch.float32)[None, None]
            conf = F.interpolate(conf, size=(view.height, view.width), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            np.savez_compressed(out / f"{view.name}.npz", depth=depth.astype(np.float32),
                                valid=depth > 0, conf=conf.astype(np.float32))
        (out / "meta.json").write_text(json.dumps({
            "model": "DA3-GIANT-1.1", "process_res": args.process_res, "views": [v.name for v in views],
            "seconds": time.time() - started, "resize": "bilinear to benchmark size"}, indent=2))
        print(f"{args.dataset}/{args.n_views}v/{scene_name}: da3 done", flush=True)


if __name__ == "__main__":
    main()
