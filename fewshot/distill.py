"""Posterior-mean pseudo-views: average the renders of several independently trained members.

Members trained from the same certified geometry but different seeds disagree mostly where the
three views do not determine the scene (single-view, occluded and out-of-frustum content). Their
average is a Monte-Carlo estimate of the posterior-mean image, the squared-error-optimal render.
This script renders that average at virtual cameras built from the training poses only
(pairwise interpolation / extrapolation) and stores them as pseudo-views for a student model:

    python -m fewshot.distill --dataset LLFF --scene fern --members run/seed0 run/seed1 run/seed2 \
        --output experiments/pseudo/LLFF/fern --ts=-0.25,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.25
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .camera import Camera
from .carve import virtual_cameras
from .data import load_scene
from .gaussians import Gaussians
from .render import render


def barycentric_cameras(cams, count, expand=0.15):
    """Cameras inside the (slightly expanded) simplex of the training cameras: centres are weighted
    combinations, rotations the chordal (SVD-projected) weighted mean. Uses the global torch RNG."""
    out = []
    n = len(cams)
    for k in range(count):
        w = torch.distributions.Dirichlet(torch.ones(n)).sample()
        w = (1 + expand) * w - expand / n                       # expand around the centroid
        w = w.to(cams[0].K.device)
        center = sum(wi * c.center for wi, c in zip(w, cams))
        M = sum(wi * c.c2w[:3, :3] for wi, c in zip(w, cams))
        U, _, Vt = torch.linalg.svd(M)
        R = U @ torch.diag(torch.tensor([1.0, 1.0, float(torch.det(U @ Vt))], device=M.device)) @ Vt
        w2c = torch.eye(4, device=M.device)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ center
        out.append(Camera(cams[0].K, w2c, cams[0].height, cams[0].width, None, f"bary_{k}", M.device))
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--members", nargs="+", required=True, help="training output dirs (point_cloud.ply)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ts", default="-0.25,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.25")
    parser.add_argument("--barycentric", type=int, default=0, help="extra cameras inside the camera simplex")
    args = parser.parse_args()
    scene = load_scene(args.dataset, args.scene)
    cams = [Camera.from_view(v) for v in scene.train]
    members = []
    for path in args.members:
        g = Gaussians(3)
        g.load_ply(Path(path) / "point_cloud.ply")
        members.append(g)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vcams = virtual_cameras(cams, tuple(float(t) for t in args.ts.split(",")))
    if args.barycentric:
        torch.manual_seed(0)
        vcams += barycentric_cameras(cams, args.barycentric)
    for i, vc in enumerate(vcams):
        renders = torch.stack([render(vc, g)["rgb"].clamp(0, 1) for g in members])
        np.savez_compressed(out / f"pseudo_{i:03d}.npz", K=vc.K.cpu().numpy(), w2c=vc.w2c.cpu().numpy(),
                            image=renders.mean(0).permute(1, 2, 0).cpu().numpy().astype(np.float32),
                            spread=renders.std(0).mean(0).cpu().numpy().astype(np.float32))
    print(f"{args.dataset}/{args.scene}: {len(vcams)} pseudo-views from {len(members)} members", flush=True)


if __name__ == "__main__":
    main()
