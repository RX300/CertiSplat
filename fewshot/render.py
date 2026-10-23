"""Thin wrapper over the CUDA Gaussian rasterizer (colour, depth and alpha with gradients)."""
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(cam, gaussians, background=None, colors=None, opacity=None, mask=None, sh_degree=None):
    """Render `gaussians` from `cam`.

    colors:  optional (N, 3) colours replacing SH appearance.
    opacity: optional (N, 1) opacities replacing the model's.
    mask:    optional (N,) bool, Gaussians to rasterize (others are skipped entirely).
    Returns rgb (3, H, W), depth (1, H, W) = sum_i w_i z_i, alpha (1, H, W), and
    densification bookkeeping (viewspace points, visibility, radii) in full-N indexing.
    """
    means = gaussians.xyz
    viewspace = torch.zeros_like(means, requires_grad=True)
    if torch.is_grad_enabled():
        viewspace.retain_grad()
    if background is None:
        background = torch.zeros(3, device=means.device)
    settings = GaussianRasterizationSettings(
        image_height=cam.height, image_width=cam.width, tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=background, scale_modifier=1.0, viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree if sh_degree is None else sh_degree,
        campos=cam.center, prefiltered=False, debug=False)
    opacity = gaussians.opacity if opacity is None else opacity
    shs = gaussians.features if colors is None else None
    scales, rotations = gaussians.scaling, gaussians.rotation
    if mask is not None:
        index = mask.nonzero(as_tuple=True)[0]
        means_, view_ = means[index], viewspace[index]
        rgb, radii_sub, depth, alpha = GaussianRasterizer(settings)(
            means3D=means_, means2D=view_, shs=None if shs is None else shs[index],
            colors_precomp=None if colors is None else colors[index], opacities=opacity[index],
            scales=scales[index], rotations=rotations[index], cov3D_precomp=None)
        radii = torch.zeros(len(means), dtype=radii_sub.dtype, device=means.device)
        radii[index] = radii_sub
    else:
        rgb, radii, depth, alpha = GaussianRasterizer(settings)(
            means3D=means, means2D=viewspace, shs=shs, colors_precomp=colors, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=None)
    return {"rgb": rgb, "depth": depth, "alpha": alpha, "viewspace": viewspace,
            "visible": radii > 0, "radii": radii}


class Frozen:
    """Detached view of a Gaussian model, for diagnostic renders that must not touch its gradients."""

    def __init__(self, gaussians):
        self.xyz = gaussians.xyz.detach()
        self.scaling = gaussians.scaling.detach()
        self.rotation = gaussians.rotation.detach()
        self.opacity = gaussians.opacity.detach()
        self.features = gaussians.features.detach()
        self.active_sh_degree = gaussians.active_sh_degree

    def __len__(self):
        return len(self.xyz)


def blend_weights(cam, gaussians, mask=None, opacity=None):
    """Per-Gaussian total compositing weight sum_p T_p * alpha_p in this view, (N,)."""
    frozen = Frozen(gaussians)
    colors = torch.ones(len(frozen), 3, device="cuda", requires_grad=True)
    with torch.enable_grad():
        out = render(cam, frozen, colors=colors, mask=mask, opacity=opacity)
        out["rgb"][0].sum().backward()
    return colors.grad[:, 0].detach()
