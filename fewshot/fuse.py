"""Fuse precise-but-incomplete depth (real-pair stereo, multi-view MVS) with a complete prior.

Per training view (see `precise_view` and `run`):
  precise = stereo / MVS depth whose maximum relative deviation from their mean is <= 1%
            (two-source range <= 2%; confidence 1, "certified"). Consistent stereo-pair
            estimates also certify depth upstream (pairwise range < 1%). A single
            uncertified estimate has confidence 0.7;
            conflicting estimates are rejected;
  filled  = the complete prior (Depth Anything 3) times a smooth correction field exp(r),
            r = log(precise / complete) interpolated by multi-scale normalised convolution
            (confidence 0.2).
Outputs experiments/priors/<dataset>/<n>v/<scene>/<name>/<view>.npz (depth, valid, conf, precise)
and init_points.npz (back-projected colour points of all training views).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera
from .data import load_scene, prior_dir, SCENES


def load_prior(directory, view):
    data = np.load(directory / f"{view}.npz")
    return data["depth"].astype(np.float32), data["valid"].astype(bool)


def load_full(directory, view):
    path = directory / f"{view}.npz"
    if not path.exists():      # e.g. MVS failed for this scene: treat as empty
        return None
    data = np.load(path)
    return {"depth": data["depth"].astype(np.float32), "valid": data["valid"].astype(bool),
            "conf": data["conf"].astype(np.float32)}


def smooth_fill(values, mask, sigmas=(4, 8, 16, 32, 64, 128)):
    """Normalised-convolution interpolation of `values` known on `mask`, finest scale first."""
    v = torch.as_tensor(values * mask, dtype=torch.float32, device="cuda")[None, None]
    m = torch.as_tensor(mask, dtype=torch.float32, device="cuda")[None, None]
    out = torch.full_like(v, float("nan"))
    for sigma in sigmas:
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, device="cuda", dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()

        def blur(t):
            t = F.conv2d(t, g.view(1, 1, 1, -1), padding=(0, radius))
            return F.conv2d(t, g.view(1, 1, -1, 1), padding=(radius, 0))

        num, den = blur(v), blur(m)
        ok = torch.isnan(out) & (den > 0.05)   # at least 5% (weighted) support at this scale
        out = torch.where(ok, num / den.clamp_min(1e-8), out)
    global_value = float(values[mask].mean()) if mask.any() else 0.0
    out = torch.nan_to_num(out, nan=global_value)
    return out[0, 0].cpu().numpy()


def correct(complete, depth, mask):
    """Complete prior times a smooth correction field fitted on `mask` pixels of `depth`."""
    ratio = np.log(np.where(mask, depth, 1) / np.maximum(complete, 1e-8))
    return complete * np.exp(smooth_fill(ratio, mask & (complete > 0)))


def precise_view(stereo, mvs, agree=0.01, complete_shape=None):
    """Mean depth when each valid source differs from the mean by at most `agree` (two-source
    range <= 2 * agree); a sole source is retained. Confidence 1 also preserves upstream
    stereo-pair agreement, which need not involve different algorithms; otherwise 0.7."""
    sources = [x for x in (stereo, mvs) if x is not None]
    if not sources:            # no precise source (ablation: foundation prior only)
        return np.zeros(complete_shape, np.float32), np.zeros(complete_shape, bool), np.zeros(complete_shape)
    depths = np.stack([x["depth"] for x in sources])
    valids = np.stack([x["valid"] for x in sources])
    count = valids.sum(0)
    mean = np.where(valids, depths, 0).sum(0) / np.maximum(count, 1)
    spread = np.where(valids, np.abs(depths - mean[None]), 0).max(0) / np.maximum(mean, 1e-8)
    ok = (count >= 1) & (spread <= agree)
    two = (count >= 2) | ((stereo["valid"] & (stereo["conf"] >= 0.99)) if stereo is not None else False)
    return np.where(ok, mean, 0).astype(np.float32), ok, np.where(ok, np.where(two, 1.0, 0.7), 0.0)


def run(dataset, scene, n_views, precise, complete, name, agree=0.01):
    s = load_scene(dataset, scene, n_views)
    out = prior_dir(dataset, scene, n_views, name)
    out.mkdir(parents=True, exist_ok=True)
    xyz, rgb, stats = [], [], []
    for view in s.train:
        comp, comp_ok = load_prior(prior_dir(dataset, scene, n_views, complete), view.name)
        stereo, mvs = ([load_full(prior_dir(dataset, scene, n_views, p), view.name) for p in precise] + [None, None])[:2]
        depth, ok, conf = precise_view(stereo, mvs, agree, comp.shape)
        comp = np.where(comp_ok, comp, 0)
        fused = np.where(ok, depth, correct(comp, depth, ok)).astype(np.float32)
        conf = np.where(ok, conf, 0.2).astype(np.float32)
        np.savez_compressed(out / f"{view.name}.npz", depth=fused, valid=fused > 0, conf=conf, precise=ok)
        cam = Camera.from_view(view)
        d = torch.tensor(fused, device="cuda")
        xyz.append(cam.backproject(d)[d > 0].cpu().numpy())
        rgb.append(cam.image.permute(1, 2, 0)[d > 0].cpu().numpy())
        stats.append((round(float((conf == 1).mean()), 3), round(float(ok.mean()), 3)))
    np.savez_compressed(out / "init_points.npz", xyz=np.concatenate(xyz), rgb=np.concatenate(rgb))
    print(f"{dataset}/{n_views}v/{scene} {name}: (certified, precise) fraction per view {stats}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--precise", nargs="*", default=["stereo", "mvs"], help="none: foundation prior only")
    parser.add_argument("--complete", default="da3")
    parser.add_argument("--name", default="fused_np")
    parser.add_argument("--agree", type=float, default=0.01)
    args = parser.parse_args()
    for scene in args.scenes or SCENES[args.dataset]:
        run(args.dataset, scene, args.n_views, args.precise, args.complete, args.name, agree=args.agree)


if __name__ == "__main__":
    main()
