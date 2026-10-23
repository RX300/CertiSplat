"""Per-scene Gaussian splatting from sparse views (training loop and evaluation).

Defaults are the CertiSplat recipe: Gaussians initialised from fused certified depth
(fewshot.fuse), Gaussian dropout and adaptive opacity decay. When the training cameras span a wide
angle (object-centric capture, test views far from every training view), three far-view
constraints built on the certified geometry are switched on (far_view="auto"):
  * free-space carving   - Gaussians in space a training view saw as empty lose opacity;
  * virtual-view teacher - anchored surface points, splatted into virtual cameras, demand
                           coverage and depth there (no network at training time);
  * anisotropy cap       - splats cannot become needles that turn see-through off-axis.
"""
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torchvision

from .camera import Camera
from .carve import FreeSpace, virtual_teachers
from .data import load_scene, read_points3d, dtu_background_mask
from .gaussians import Gaussians
from .metrics import benchmark_metrics, photometric_loss
from .render import render


@dataclass
class Config:
    dataset: str = "LLFF"
    scene: str = "fern"
    output: str = ""
    n_views: int = 3
    dense: bool = False                 # train on all non-test views (upper-bound analysis)
    seed: int = 0
    iterations: int = 10000
    sh_degree: int = 3
    init: str = "colmap"                # colmap | npz:<xyz, rgb> | ply:<path> | stereogs (its cached MVS init)
    init_subsample: float = 0.5         # random fraction of initial points kept (StereoGS uses 0.1)
    init_opacity: float = 0.1
    # optimisation (3DGS defaults)
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 10000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lambda_dssim: float = 0.2
    # density control and regularisation
    densify_from: int = 1000
    densify_until: int = 10000
    densify_every: int = 100
    densify_grad: float = 0.0002
    min_opacity: float = 0.005
    opacity_reset_every: int = 0        # 0 disables
    opacity_decay: float = 0.99         # StereoGS adaptive decay min-rate (0 disables)
    decay_sensitivity: float = 0.5
    dropout: float = 0.3                # Gaussian dropout; test opacity is scaled by (1 - p)
    # far-view constraints (need the per-view fused depth: fused_dir, default = folder of an npz init)
    far_view: str = "auto"              # auto: on when training camera axes span > far_view_angle | on | off
    far_view_angle: float = 45.0        # degrees; forward-facing LLFF spans 3-24, object-centric DTU 72-93
    fused_dir: str = ""
    max_anisotropy: float = 10.0        # >0: smallest scale is kept >= largest / max_anisotropy
    carve_every: int = 10               # free-space carving period (0 disables)
    carve_factor: float = 0.5
    carve_from: int = 500
    carve_margin: float = 0.03
    vt_weight: float = 0.5              # virtual-view teacher weight (0 disables)
    vt_from: int = 1000
    vt_alpha: float = 1.0               # coverage term weight relative to the depth term
    vt_ts: str = "-3,-2,-1,-0.5,0.5,1.5,2,3,4"   # pairwise inter/extrapolation positions
    # posterior-mean distillation: pseudo-views from fewshot.distill (averaged member renders)
    pseudo_dir: str = ""
    pseudo_weight: float = 0.0
    eval_every: int = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


STEREOGS_INIT = {
    "LLFF": "MVSAnywhere_3_views_H_1440_W_1920_img_dist_thresh_3.0_depth_thresh_0.08_vthresh_2_init",
    "DTU": "MVSFormer_3_views_H_1152_W_1536_init",
}


def read_ply_points(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
    return xyz, rgb


def initial_points(cfg, scene):
    if cfg.init == "colmap":
        xyz, rgb = read_points3d(scene.source / "sparse/0")
    elif cfg.init.startswith("npz:"):
        data = np.load(cfg.init[4:])
        xyz, rgb = data["xyz"], data["rgb"]
    elif cfg.init.startswith("ply:"):
        xyz, rgb = read_ply_points(cfg.init[4:])
    elif cfg.init == "stereogs":
        xyz, rgb = read_ply_points(scene.source / STEREOGS_INIT[cfg.dataset] / "mvs_input.ply")
    else:
        raise ValueError(cfg.init)
    if cfg.init_subsample < 1.0:
        keep = np.random.choice(len(xyz), int(len(xyz) * cfg.init_subsample), replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


def capture_spread(cams):
    """Largest angle (degrees) between the optical axes of two training cameras."""
    axes = torch.stack([c.c2w[:3, 2] for c in cams])
    return float(torch.rad2deg(torch.arccos((axes @ axes.T).clamp(-1, 1))).max())


def load_fused(cfg, cams):
    if not cfg.fused_dir and not cfg.init.startswith("npz:"):
        raise ValueError("far-view constraints need --fused-dir (or an npz init inside a fused prior folder)")
    folder = Path(cfg.fused_dir) if cfg.fused_dir else Path(cfg.init[4:]).parent
    data = [np.load(folder / f"{c.name}.npz") for c in cams]
    tensor = lambda key: [torch.tensor(d[key], device="cuda") for d in data]
    return tensor("depth"), tensor("conf"), tensor("precise")


def teacher_loss(teachers, gaussians, keep, cfg):
    """Coverage and inverse-depth agreement at a random virtual camera."""
    vcam, target = random.choice(teachers)
    pkg = render(vcam, gaussians, mask=keep)
    alpha = pkg["alpha"][0]
    depth = pkg["depth"][0] / alpha.detach().clamp_min(1e-6)
    have = target > 0
    solid = have & (alpha.detach() > 0.5)
    depth_term = ((1 / depth[solid].clamp_min(1e-6) - 1 / target[solid]).abs().mean()
                  if solid.any() else depth.sum() * 0)
    cover_term = (1 - alpha[have]).mean() if have.any() else alpha.sum() * 0
    return depth_term + cfg.vt_alpha * cover_term, float(depth_term), float(cover_term)


def scene_background(scene):
    """White for datasets whose images are composited onto white (Blender), else the default black."""
    return torch.ones(3, device="cuda") if scene.views[0].white_background else None


def evaluate(gaussians, scene, out_dir, opacity_scale=1.0, save=True):
    out_dir = Path(out_dir)
    background = scene_background(scene)
    if save:
        (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    opacity = gaussians.opacity * opacity_scale
    per_view = []
    with torch.no_grad():
        for position, view in enumerate(scene.test):
            cam = Camera.from_view(view)
            rgb = render(cam, gaussians, background, opacity=opacity)["rgb"].clamp(0, 1)
            mask = scene.test_mask(position)
            mask = None if mask is None else torch.tensor(mask, device="cuda").permute(2, 0, 1)
            metrics = benchmark_metrics(rgb, cam.image, mask)
            per_view.append({"view": view.name, **metrics})
            if save:
                torchvision.utils.save_image(rgb, out_dir / "renders" / f"{view.name}.png")
    summary = {k: float(np.mean([v[k] for v in per_view])) for k in ("PSNR", "SSIM", "LPIPS")}
    return summary, per_view


def train(cfg: Config):
    set_seed(cfg.seed)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    scene = load_scene(cfg.dataset, cfg.scene, cfg.n_views, cfg.dense)
    cams = [Camera.from_view(v) for v in scene.train]
    background = scene_background(scene)
    bg_masks = None
    if cfg.dataset == "DTU":
        bg_masks = [torch.tensor(dtu_background_mask(c.image.permute(1, 2, 0).cpu().numpy(), cfg.scene),
                                 dtype=torch.float32, device="cuda") for c in cams]

    gaussians = Gaussians(cfg.sh_degree)
    xyz, rgb = initial_points(cfg, scene)
    gaussians.create(xyz, rgb, opacity=cfg.init_opacity)
    gaussians.setup_optimizer(cfg, scene.extent)

    spread = capture_spread(cams)
    far = cfg.far_view == "on" or (cfg.far_view == "auto" and spread > cfg.far_view_angle)
    carve_every, vt_weight, max_anisotropy = (cfg.carve_every, cfg.vt_weight, cfg.max_anisotropy) if far else (0, 0.0, 0.0)
    free_space, teachers = None, []
    if carve_every or vt_weight > 0:
        depths, confs, precise = load_fused(cfg, cams)
        if carve_every:
            free_space = FreeSpace(cams, depths, confs, cfg.carve_margin)
        if vt_weight > 0:
            teachers = virtual_teachers(cams, depths, precise, tuple(float(t) for t in cfg.vt_ts.split(",")))

    pseudo = []
    if cfg.pseudo_weight > 0:
        for path in sorted(Path(cfg.pseudo_dir).glob("pseudo_*.npz")):
            d = np.load(path)
            pseudo.append(Camera(d["K"], d["w2c"], d["image"].shape[0], d["image"].shape[1], d["image"], path.stem))

    log = open(out / "train_log.jsonl", "w")
    log.write(json.dumps({"capture_spread_degrees": spread, "far_view": far}) + "\n")
    started = time.time()
    order = []
    for step in range(1, cfg.iterations + 1):
        gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            gaussians.one_up_sh()
        if not order:
            order = list(range(len(cams)))
            random.shuffle(order)
        index = order.pop()
        cam = cams[index]

        keep = torch.rand(len(gaussians), device="cuda") >= cfg.dropout if cfg.dropout > 0 else None
        out_pkg = render(cam, gaussians, background, mask=keep)
        loss = photometric_loss(out_pkg["rgb"], cam.image, cfg.lambda_dssim)
        terms = {"photo": float(loss)}
        if bg_masks is not None:
            loss = loss + (out_pkg["alpha"][0] * bg_masks[index]).mean()
        if pseudo:
            pcam = random.choice(pseudo)
            ppkg = render(pcam, gaussians, background, mask=keep)
            pseudo_term = photometric_loss(ppkg["rgb"], pcam.image, cfg.lambda_dssim)
            loss = loss + cfg.pseudo_weight * pseudo_term
            terms["pseudo"] = float(pseudo_term)
        if teachers and step >= cfg.vt_from:
            vt, terms["vt_depth"], terms["vt_cover"] = teacher_loss(teachers, gaussians, keep, cfg)
            loss = loss + vt_weight * vt
        loss.backward()

        with torch.no_grad():
            if cfg.opacity_decay > 0:
                gaussians.decay_opacity(cfg.opacity_decay, cfg.decay_sensitivity)
            if free_space is not None and step >= cfg.carve_from and step % carve_every == 0:
                bad = free_space.violators(gaussians.xyz)
                opacity = gaussians.opacity
                opacity[bad] = (opacity[bad] * cfg.carve_factor).clamp_min(1e-6)
                gaussians._opacity.data = torch.log(opacity / (1 - opacity))
                terms["carved"] = int(bad.sum())
            if step < cfg.densify_until:
                gaussians.add_stats(out_pkg["viewspace"], out_pkg["visible"], out_pkg["radii"])
                if step > cfg.densify_from and step % cfg.densify_every == 0:
                    screen = 20 if (cfg.opacity_reset_every and step > cfg.opacity_reset_every) else None
                    gaussians.densify_and_prune(cfg.densify_grad, cfg.min_opacity, scene.extent, screen)
                if cfg.opacity_reset_every and step % cfg.opacity_reset_every == 0:
                    gaussians.reset_opacity()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if max_anisotropy > 0:
                floor = gaussians._scaling.max(dim=1, keepdim=True).values - float(np.log(max_anisotropy))
                gaussians._scaling.data = torch.maximum(gaussians._scaling.data, floor)

        if step % 500 == 0 or step == cfg.iterations:
            log.write(json.dumps({"step": step, "points": len(gaussians), **terms,
                                  "seconds": time.time() - started}) + "\n")
            log.flush()
        if cfg.eval_every and step % cfg.eval_every == 0 and step < cfg.iterations:
            summary, _ = evaluate(gaussians, scene, out / f"eval_{step}", 1 - cfg.dropout, save=False)
            log.write(json.dumps({"step": step, "eval": summary}) + "\n")
            log.flush()

    seconds = time.time() - started
    gaussians.save_ply(out / "point_cloud.ply", opacity_scale=1 - cfg.dropout)
    summary, per_view = evaluate(gaussians, scene, out / "test", 1 - cfg.dropout)
    result = {"dataset": cfg.dataset, "scene": cfg.scene, **summary, "points": len(gaussians),
              "train_seconds": seconds, "far_view": far}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "per_view.json").write_text(json.dumps(per_view, indent=2))
    print(json.dumps(result), flush=True)
    return result


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Config()).items():
        if isinstance(default, bool):
            parser.add_argument("--" + name.replace("_", "-"), action="store_true", default=default)
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_config())
