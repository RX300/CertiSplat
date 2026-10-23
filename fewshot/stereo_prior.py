"""FoundationStereo priors from rectified pairs of real benchmark training images.

Run: CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh tools -m fewshot.stereo_prior --dataset LLFF --all
"""
import argparse
from collections import defaultdict
from itertools import combinations
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode

from .camera import neighbor_pairs
from .data import SCENES, load_scene, prior_dir
from .prior_io import save_prior, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    """Load release weights with scoped NumPy and configuration class allowances."""
    with torch.serialization.safe_globals([
        np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType,
        ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode,
        Any, dict, list, int, defaultdict,
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def bilinear(array, valid, u, v):
    """Sample pixel-index coordinates; require all four neighbours to be valid."""
    h, w = array.shape
    finite = np.isfinite(u) & np.isfinite(v)
    x = np.floor(np.where(finite, u, -1)).astype(int)
    y = np.floor(np.where(finite, v, -1)).astype(int)
    inside = finite & (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    xc, yc = x.clip(0, w - 2), y.clip(0, h - 2)
    ok = inside & valid[yc, xc] & valid[yc, xc + 1] & valid[yc + 1, xc] & valid[yc + 1, xc + 1]
    dx, dy = np.where(inside, u - x, 0), np.where(inside, v - y, 0)
    values = ((1 - dx) * (1 - dy) * array[yc, xc] + dx * (1 - dy) * array[yc, xc + 1]
              + (1 - dx) * dy * array[yc + 1, xc] + dx * dy * array[yc + 1, xc + 1])
    return np.where(ok, values, 0), ok


def working_image(view, long_side):
    with Image.open(view.path) as image:
        scale = long_side / max(image.size)
        size = tuple(round(n * scale) for n in image.size)
        rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BICUBIC))
    K = view.K.astype(np.float64).copy()
    K[0] *= size[0] / view.width
    K[1] *= size[1] / view.height
    K[:2, 2] -= 0.5  # COLMAP centres -> OpenCV integer centres, AFTER scaling.
    return rgb, K


class DegeneratePair(ValueError):
    """A pair cannot be rectified into useful overlapping images."""


def rectify(view1, view2, image1, image2, K1, K2):
    h, w = image1.shape[:2]
    assert image2.shape == image1.shape
    relative = view2.w2c.astype(np.float64) @ np.linalg.inv(view1.w2c.astype(np.float64))
    R, t = relative[:3, :3], relative[:3, 3]
    baseline = np.linalg.norm(t)
    if baseline < 1e-10:
        raise DegeneratePair("zero baseline")
    try:
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K1, np.zeros(5), K2, np.zeros(5), (w, h), R, t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    except cv2.error as error:
        raise DegeneratePair(str(error)) from error
    # Fit the union of both warped image footprints ourselves. OpenCV's alpha=1
    # scale can become negative for converging cameras with disjoint footprints
    # at infinity, even when their finite-depth scene overlap is useful.
    if (not np.isfinite(P1).all() or P1[0, 0] <= 0 or P1[1, 1] <= 0
            or min(R1[2, 2], R2[2, 2]) <= 0):
        raise DegeneratePair("nonpositive focal length or backwards rectified optical axis")
    corners = np.array([[0., 0., 1.], [w - 1., 0., 1.],
                        [0., h - 1., 1.], [w - 1., h - 1., 1.]])
    footprints = []
    for K, rotation, projection in [(K1, R1, P1), (K2, R2, P2)]:
        rays = corners @ np.linalg.inv(K).T @ rotation.T
        if np.any(rays[:, 2] <= 0):
            raise DegeneratePair("image field of view crosses the rectified camera plane")
        pixels = rays @ projection[:, :3].T
        footprints.append(pixels[:, :2] / pixels[:, 2:])
    points = np.concatenate(footprints)
    low, high = points.min(0), points.max(0)
    scale = min((w - 1) / (high[0] - low[0]), (h - 1) / (high[1] - low[1]))
    offset = (np.array([w - 1, h - 1]) - scale * (high + low)) * 0.5
    fit = np.array([[scale, 0., offset[0]], [0., scale, offset[1]], [0., 0., 1.]])
    P1, P2 = fit @ P1, fit @ P2
    rectified, support = [], []
    for image, K, rotation, projection in [(image1, K1, R1, P1), (image2, K2, R2, P2)]:
        mx, my = cv2.initUndistortRectifyMap(K, np.zeros(5), rotation, projection[:3, :3],
                                          (w, h), cv2.CV_32FC1)
        ok = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        rectified.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
        support.append(ok)
    # Rotate a vertical epipolar direction into a horizontal one. A final
    # horizontal reflection makes disparities positive without changing poses.
    A = np.eye(3)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        A = np.array([[0., 1., 0.], [-1., 0., w - 1.], [0., 0., 1.]])
        rectified = [np.rot90(a) for a in rectified]
        support = [np.rot90(a) for a in support]
    if (A @ P2)[0, 3] > 0:
        width = rectified[0].shape[1]
        A = np.array([[-1., 0., width - 1.], [0., 1., 0.], [0., 0., 1.]]) @ A
        rectified = [a[:, ::-1] for a in rectified]
        support = [a[:, ::-1] for a in support]
    if min(a.mean() for a in support) < 0.01:
        raise DegeneratePair("less than 1% valid rectified image support")
    return rectified, support, (R1, R2), (A @ P1, A @ P2), R2 @ t, A


class Stereo:
    def __init__(self, checkpoint_dir=ROOT / "checkpoints/FoundationStereo/vit_small"):
        sys.path.insert(0, str(ROOT / "StereoGS/submodules/FoundationStereo"))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from omegaconf import OmegaConf
        self.padder = InputPadder
        self.checkpoint = Path(checkpoint_dir) / "model_best_bp2.pth"
        checkpoint = self.checkpoint
        # The official large release omits vit_size; upstream run_demo uses vitl.
        self.config = OmegaConf.merge({"vit_size": "vitl"},
                                      OmegaConf.load(checkpoint.with_name("cfg.yaml")))
        self.model = FoundationStereo(self.config)
        self.model.load_state_dict(load_checkpoint(checkpoint)["model"])
        self.model.cuda().eval()

    @torch.inference_mode()
    def disparities(self, images, iters):
        left, right = [torch.from_numpy(np.array(a, dtype=np.float32)).permute(2, 0, 1)[None].cuda()
                       for a in images]
        padder = self.padder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        left_batch = torch.cat([left, torch.flip(right, [3])])
        right_batch = torch.cat([right, torch.flip(left, [3])])
        with torch.autocast(device_type="cuda"):
            disparity = self.model(left_batch, right_batch, iters=iters, test_mode=True)
        return [padder.unpad(disparity[:1].float())[0, 0].cpu().numpy(),
                padder.unpad(torch.flip(disparity[1:2], [3]).float())[0, 0].cpu().numpy()]


def consistent_disparities(disparities, support, threshold):
    valid = [s & np.isfinite(d) & (d > 0) for d, s in zip(disparities, support)]
    clean = [np.where(v, d, 0) for d, v in zip(disparities, valid)]
    y, x = np.indices(disparities[0].shape, dtype=np.float64)
    results = []
    for i, sign in [(0, -1), (1, 1)]:
        other, ok = bilinear(clean[1 - i], valid[1 - i], x + sign * clean[i], y)
        results.append(valid[i] & ok & (np.abs(clean[i] - other) <= threshold))
    return clean, results


def unrectify_depth(view, index, disparity, valid, rotations, projections, translation):
    """Inverse-map benchmark centres, then triangulate in the common rectified frame."""
    y, x = np.indices((view.height, view.width), dtype=np.float64)
    pixels = np.stack([x + 0.5, y + 0.5, np.ones_like(x)], -1)
    rays = pixels @ np.linalg.inv(view.K.astype(np.float64)).T
    rect_rays = rays @ rotations[index].T
    projected = rect_rays @ projections[index][:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = projected[..., 0] / projected[..., 2], projected[..., 1] / projected[..., 2]
    sampled, ok = bilinear(disparity, valid, u, v)
    ok &= rect_rays[..., 2] > 0
    locations = np.flatnonzero(ok)
    uv = np.stack([u.flat[locations], v.flat[locations]])
    other = uv.copy()
    other[0] += (-1 if index == 0 else 1) * sampled.flat[locations]
    left, right = (uv, other) if index == 0 else (other, uv)
    depth = np.zeros((view.height, view.width), dtype=np.float32)
    if locations.size:
        homogeneous = cv2.triangulatePoints(projections[0], projections[1], left, right)
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = (homogeneous[:3] / homogeneous[3]).T
        local = (xyz + (translation if index == 1 else 0)) @ rotations[index]
        z = local[:, 2]
        good = np.isfinite(z) & (z > 0)
        depth.flat[locations[good]] = z[good]
    return depth


def fuse(estimates, tau):
    """Largest agreeing subset; ties use the first subset in input order.

    Agreement is max(depth) - min(depth) < tau * mean(depth), preserving the
    original two-estimate threshold. Only a sole valid estimate gets conf 0.5.
    """
    depths = np.stack(estimates)
    valid = np.isfinite(depths) & (depths > 0)
    count = valid.sum(0)
    single = count == 1
    depth = np.where(single, np.where(valid, depths, 0).sum(0), 0).astype(np.float32)
    agree = np.zeros_like(single)
    for size in range(len(estimates), 1, -1):
        for indices in combinations(range(len(estimates)), size):
            subset = depths[list(indices)]
            mean = subset.mean(0)
            spread = subset.max(0) - subset.min(0)
            take = (~agree) & valid[list(indices)].all(0) & (spread < tau * mean)
            depth[take] = mean[take]
            agree |= take
    confidence = agree.astype(np.float32) + 0.5 * single
    stats = {"coverage": float((agree | single).mean()), "both_agree": float(agree.mean()),
             "single": float(single.mean()), "disagree": float(((count >= 2) & ~agree).mean()),
             "neither": float((count == 0).mean())}
    return depth, depth > 0, confidence, stats


def run_scene(dataset, name, args, stereo, model_seconds):
    started = time.perf_counter()
    scene = load_scene(dataset, name, args.n_views)
    work = [working_image(v, args.long_side) for v in scene.train]
    estimates = {v.name: [] for v in scene.train}
    meta = {"dataset": dataset, "scene": name, "n_views": args.n_views, "source": args.source_name,
            "train_names": [v.name for v in scene.train], "long_side": args.long_side,
            "iters": args.iters, "lr_threshold": args.lr_threshold, "fusion_tau": args.tau,
            "rectify_alpha": -1, "rectified_canvas": "shared similarity fitting the union of both image footprints",
            "minimum_rectified_support": 0.01,
            "checkpoint": os.path.relpath(stereo.checkpoint, ROOT),
            "checkpoint_max_disp": int(stereo.config.max_disp),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "image_resize": "PIL bicubic from full-resolution originals",
            "fusion_denominator": "arithmetic mean camera-z", "pairs": [], "views": {},
            "timings_seconds": {"model_load": model_seconds}}
    centers = np.stack([np.linalg.inv(v.w2c)[:3, 3] for v in scene.train])
    for i, j in neighbor_pairs(centers, k=2):
        tick = time.perf_counter()
        vi, vj = scene.train[i], scene.train[j]
        record = {"names": [vi.name, vj.name]}
        try:
            images, support, rotations, projections, translation, A = rectify(
                vi, vj, work[i][0], work[j][0], work[i][1], work[j][1])
        except DegeneratePair as error:
            record.update(status="skipped", reason=str(error))
            print(f"{dataset}/{name}: skip {vi.name}/{vj.name}: {error}", flush=True)
            for view in (vi, vj):
                estimates[view.name].append(np.zeros((view.height, view.width), np.float32))
        else:
            inference_start = time.perf_counter()
            disparities = stereo.disparities(images, args.iters)
            inference_seconds = time.perf_counter() - inference_start
            clean, valid = consistent_disparities(disparities, support, args.lr_threshold)
            for index, view in enumerate((vi, vj)):
                estimates[view.name].append(unrectify_depth(
                    view, index, clean[index], valid[index], rotations, projections, translation))
            record.update(status="ok", inference_seconds=inference_seconds,
                          rectified_shape=list(images[0].shape[:2]),
                          support=[float(s.mean()) for s in support],
                          consistent=[float(v.mean()) for v in valid],
                          benchmark_coverage=[float((estimates[v.name][-1] > 0).mean()) for v in (vi, vj)],
                          image_transform=A.tolist(), rotations=[r.tolist() for r in rotations],
                          projections=[p.tolist() for p in projections])
        record["seconds"] = time.perf_counter() - tick
        meta["pairs"].append(record)
        print(f"{dataset}/{name}: pair {i},{j} {record['status']} in {record['seconds']:.1f}s", flush=True)
    out = prior_dir(dataset, name, args.n_views, args.source_name)
    for view in scene.train:
        depth, valid, conf, stats = fuse(
            estimates[view.name] or [np.zeros((view.height, view.width), np.float32)], args.tau)
        save_prior(out, view.name, depth, valid, conf)
        meta["views"][view.name] = stats
    meta["timings_seconds"]["total"] = time.perf_counter() - started
    write_json(out / "meta.json", meta)
    print(f"{dataset}/{name}: complete in {meta['timings_seconds']['total']:.1f}s; "
          f"coverage {[round(s['coverage'], 3) for s in meta['views'].values()]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SCENES, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--long-side", type=int, default=1008)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--lr-threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/FoundationStereo/vit_small")
    parser.add_argument("--source-name", default="stereo")
    args = parser.parse_args()
    torch.manual_seed(0)
    tick = time.perf_counter()
    stereo = Stereo(args.checkpoint_dir)
    model_seconds = time.perf_counter() - tick
    for name in SCENES[args.dataset] if args.all else [args.scene]:
        run_scene(args.dataset, name, args, stereo, model_seconds)


if __name__ == "__main__":
    main()
