# Densification logic follows 3D Gaussian Splatting (Kerbl et al. 2023; Inria GRAPHDECO,
# non-commercial research licence, see ../StereoGS/LICENSE.md).
"""Gaussian parameters, optimiser, adaptive density control and PLY I/O."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def build_rotation(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                       -1).reshape(-1, 3, 3)


def expon_lr(step, lr_init, lr_final, max_steps, delay_steps=0, delay_mult=1.0):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    delay = 1.0
    if delay_steps > 0:
        delay = delay_mult + (1 - delay_mult) * math.sin(0.5 * math.pi * min(step / delay_steps, 1))
    t = min(max(step / max_steps, 0), 1)
    return delay * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


class Gaussians:
    PARAMS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

    def __init__(self, sh_degree):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.optimizer = None

    # ------------------------------------------------------------------ creation
    def create(self, xyz, rgb, opacity=0.1, scale_mult=1.0):
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device="cuda")
        rgb = torch.as_tensor(rgb, dtype=torch.float32, device="cuda")
        n = len(xyz)
        features = torch.zeros(n, 3, (self.max_sh_degree + 1) ** 2, device="cuda")
        features[:, :, 0] = rgb_to_sh(rgb)
        dist2 = distCUDA2(xyz).clamp_min(1e-7)
        scales = torch.log(torch.sqrt(dist2) * scale_mult)[:, None].repeat(1, 3)
        rotations = torch.zeros(n, 4, device="cuda")
        rotations[:, 0] = 1
        self._set({
            "xyz": xyz,
            "f_dc": features[:, :, :1].transpose(1, 2).contiguous(),
            "f_rest": features[:, :, 1:].transpose(1, 2).contiguous(),
            "opacity": inverse_sigmoid(torch.full((n, 1), opacity, device="cuda")),
            "scaling": scales,
            "rotation": rotations,
        })

    def _set(self, tensors):
        for key in self.PARAMS:
            setattr(self, "_" + key, torch.nn.Parameter(tensors[key].contiguous().requires_grad_(True)))
        self._reset_stats()

    def _reset_stats(self):
        n = len(self._xyz)
        self.grad_accum = torch.zeros(n, 1, device="cuda")
        self.grad_count = torch.zeros(n, 1, device="cuda")
        self.max_radii2D = torch.zeros(n, device="cuda")

    # --------------------------------------------------------------- activations
    @property
    def xyz(self):
        return self._xyz

    @property
    def scaling(self):
        return torch.exp(self._scaling)

    @property
    def rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def features(self):
        return torch.cat([self._f_dc, self._f_rest], dim=1)

    def __len__(self):
        return len(self._xyz)

    def covariance(self):
        M = build_rotation(self._rotation) @ torch.diag_embed(self.scaling)
        return M @ M.transpose(1, 2)

    def one_up_sh(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    # ----------------------------------------------------------------- optimiser
    def setup_optimizer(self, cfg, spatial_scale):
        self.spatial_scale = spatial_scale
        self.cfg = cfg
        groups = [
            {"params": [self._xyz], "lr": cfg.position_lr_init * spatial_scale, "name": "xyz"},
            {"params": [self._f_dc], "lr": cfg.feature_lr, "name": "f_dc"},
            {"params": [self._f_rest], "lr": cfg.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": cfg.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": cfg.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": cfg.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_learning_rate(self, step):
        cfg = self.cfg
        lr = expon_lr(step, cfg.position_lr_init * self.spatial_scale,
                      cfg.position_lr_final * self.spatial_scale, cfg.position_lr_max_steps,
                      delay_mult=cfg.position_lr_delay_mult)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr

    # ------------------------------------------------- optimiser-aware edits
    def _replace_tensor(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                state = self.optimizer.state.get(group["params"][0], None)
                if state is not None:
                    state["exp_avg"] = torch.zeros_like(tensor)
                    state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                if state is not None:
                    self.optimizer.state[group["params"][0]] = state
                setattr(self, "_" + name, group["params"][0])

    def _prune_optimizer(self, keep):
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                del self.optimizer.state[param]
            group["params"][0] = torch.nn.Parameter(param[keep].requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            setattr(self, "_" + group["name"], group["params"][0])

    def prune(self, remove):
        keep = ~remove
        self._prune_optimizer(keep)
        self.grad_accum = self.grad_accum[keep]
        self.grad_count = self.grad_count[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def append(self, tensors):
        for group in self.optimizer.param_groups:
            extra = tensors[group["name"]]
            param = group["params"][0]
            state = self.optimizer.state.get(param, None)
            if state is not None:
                state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros_like(extra)])
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(extra)])
                del self.optimizer.state[param]
            group["params"][0] = torch.nn.Parameter(torch.cat([param, extra]).requires_grad_(True))
            if state is not None:
                self.optimizer.state[group["params"][0]] = state
            setattr(self, "_" + group["name"], group["params"][0])
        n = len(self._xyz)
        self.grad_accum = torch.zeros(n, 1, device="cuda")
        self.grad_count = torch.zeros(n, 1, device="cuda")
        self.max_radii2D = torch.zeros(n, device="cuda")

    # ------------------------------------------------- adaptive density control
    def add_stats(self, viewspace_points, mask, radii):
        self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask].float())
        grad = viewspace_points.grad[mask, :2]
        self.grad_accum[mask] += torch.norm(grad, dim=-1, keepdim=True)
        self.grad_count[mask] += 1

    def densify_and_prune(self, grad_threshold, min_opacity, extent, max_screen_size=None,
                          percent_dense=0.01):
        grads = self.grad_accum / self.grad_count.clamp_min(1)
        grads[grads.isnan()] = 0.0
        selected = grads[:, 0] >= grad_threshold
        large = self.scaling.max(dim=1).values > percent_dense * extent

        clone = selected & ~large
        if clone.any():
            self.append({k: getattr(self, "_" + k)[clone].detach() for k in self.PARAMS})

        # The clone call above appended rows; pad the masks accordingly.
        n_new = len(self) - len(selected)
        selected = torch.cat([selected, torch.zeros(n_new, dtype=torch.bool, device="cuda")])
        large = torch.cat([large, torch.zeros(n_new, dtype=torch.bool, device="cuda")])
        split = selected & large
        if split.any():
            stds = self.scaling[split].repeat(2, 1)
            samples = torch.normal(torch.zeros_like(stds), stds)
            rots = build_rotation(self._rotation[split]).repeat(2, 1, 1)
            new = {
                "xyz": (rots @ samples[..., None])[..., 0] + self._xyz[split].repeat(2, 1),
                "scaling": torch.log(self.scaling[split].repeat(2, 1) / 1.6),
                "rotation": self._rotation[split].repeat(2, 1),
                "f_dc": self._f_dc[split].repeat(2, 1, 1),
                "f_rest": self._f_rest[split].repeat(2, 1, 1),
                "opacity": self._opacity[split].repeat(2, 1),
            }
            self.append({k: v.detach() for k, v in new.items()})
            remove = torch.cat([split, torch.zeros(2 * int(split.sum()), dtype=torch.bool, device="cuda")])
            self.prune(remove)

        remove = self.opacity[:, 0] < min_opacity
        if max_screen_size:
            remove |= self.max_radii2D > max_screen_size
            remove |= self.scaling.max(dim=1).values > 0.1 * extent
        self.prune(remove)
        torch.cuda.empty_cache()

    def reset_opacity(self, value=0.01):
        new = inverse_sigmoid(torch.minimum(self.opacity, torch.full_like(self.opacity, value)))
        self._replace_tensor(new, "opacity")

    def decay_opacity(self, min_rate=0.99, sensitivity=0.5):
        """StereoGS/Binocular3DGS gradient-adaptive opacity decay (call after backward)."""
        if self._opacity.grad is None:
            return
        g = self._opacity.grad.abs()
        g = g / (g.mean() + 1e-8)
        factor = 1.0 - (1.0 - min_rate) * torch.exp(-sensitivity * g)
        new = (self.opacity * factor).clamp(1e-6, 1 - 1e-6)
        self._opacity.data = inverse_sigmoid(new)

    # ------------------------------------------------------------------- I/O
    def save_ply(self, path, opacity_scale=1.0):
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._f_dc.detach().transpose(1, 2).flatten(1).cpu().numpy()
        f_rest = self._f_rest.detach().transpose(1, 2).flatten(1).cpu().numpy()
        opacity = inverse_sigmoid((self.opacity.detach() * opacity_scale).clamp(1e-6, 1 - 1e-6)).cpu().numpy()
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        attributes += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
        attributes += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        attributes += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        data = np.concatenate([xyz, np.zeros_like(xyz), f_dc, f_rest, opacity,
                               self._scaling.detach().cpu().numpy(),
                               self._rotation.detach().cpu().numpy()], axis=1)
        elements = np.empty(len(xyz), dtype=[(name, "f4") for name in attributes])
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))

    def load_ply(self, path):
        vertex = PlyData.read(str(path))["vertex"]
        names = [p.name for p in vertex.properties]
        get = lambda keys: np.stack([np.asarray(vertex[k], np.float32) for k in keys], axis=1)
        rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda n: int(n.split("_")[-1]))
        n = len(vertex["x"])
        tensors = {
            "xyz": get(["x", "y", "z"]),
            "f_dc": get([f"f_dc_{i}" for i in range(3)]).reshape(n, 3, 1).transpose(0, 2, 1),
            "f_rest": (get(rest).reshape(n, 3, -1).transpose(0, 2, 1) if rest
                       else np.zeros((n, 0, 3), np.float32)),
            "opacity": get(["opacity"]),
            "scaling": get([f"scale_{i}" for i in range(3)]),
            "rotation": get([f"rot_{i}" for i in range(4)]),
        }
        self.max_sh_degree = self.active_sh_degree = int(round(math.sqrt(tensors["f_rest"].shape[1] + 1))) - 1
        self._set({k: torch.tensor(np.ascontiguousarray(v), device="cuda") for k, v in tensors.items()})
