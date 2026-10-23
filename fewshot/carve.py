"""Free-space carving with per-view fused depth.

A training view that sees a surface at depth D along a ray has observed the space in
front of it to be empty. Gaussians lying clearly in front of D (by a margin that grows
with the depth's uncertainty) contradict that observation; their opacity is attenuated.
This removes floaters that are invisible from the training views (edge-on, thin or
low-opacity there) but appear in novel views.
"""
import torch

from .camera import sample_image


class FreeSpace:
    def __init__(self, cams, depths, confs, margin_precise=0.03, margin_filled=0.15):
        self.cams = list(cams)
        self.depths = list(depths)
        self.margins = [torch.where(c >= 0.99, margin_precise,
                                    torch.where(c >= 0.5, 2 * margin_precise, margin_filled)) for c in confs]

    @torch.no_grad()
    def violators(self, xyz):
        """(N,) bool: Gaussians in observed free space of at least one training view."""
        bad = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        for cam, depth, margin in zip(self.cams, self.depths, self.margins):
            u, v, z = cam.project(xyz)
            inside = (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1) & (z > 0)
            seen = sample_image(depth[None], u.clamp(0, cam.width - 1), v.clamp(0, cam.height - 1), "nearest")[0]
            m = sample_image(margin[None], u.clamp(0, cam.width - 1), v.clamp(0, cam.height - 1), "nearest")[0]
            bad |= inside & (seen > 0) & (z < seen * (1 - m))
        return bad


def _slerp(R0, R1, t):
    """Rotation interpolation/extrapolation via the axis-angle of R0^T R1."""
    rel = R0.T @ R1
    angle = torch.arccos(((torch.trace(rel) - 1) / 2).clamp(-1, 1))
    if angle < 1e-6:
        return R0.clone()
    axis = torch.stack([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * torch.sin(angle))
    a = angle * t
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=R0.device)
    return R0 @ (torch.eye(3, device=R0.device) + torch.sin(a) * K + (1 - torch.cos(a)) * K @ K)


def virtual_cameras(cams, ts=(-0.5, -0.25, 0.25, 0.5, 0.75, 1.25, 1.5), with_pairs=False):
    """Cameras interpolated / extrapolated along neighbouring training-camera pairs (optionally
    returned with the index pair they come from)."""
    from .camera import Camera, neighbor_pairs
    out = []
    for i, j in neighbor_pairs([c.center.cpu().numpy() for c in cams]):
        a, b = cams[i], cams[j]
        for t in ts:
            R = _slerp(a.c2w[:3, :3], b.c2w[:3, :3], t)
            center = (1 - t) * a.center + t * b.center
            w2c = torch.eye(4, device=R.device)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ center
            vc = Camera(a.K, w2c, a.height, a.width, None, f"virtual_{i}{j}_{t}", R.device)
            out.append((vc, (i, j)) if with_pairs else vc)
    return out


def virtual_teachers(cams, depths, anchors, ts=(-1.0, -0.5, -0.25, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0),
                     window=3):
    """(camera, depth) pairs: for every virtual camera, the anchored points of the two training
    views it interpolates, splatted into it and min-filtered over `window` to suppress see-through
    between sparse foreground samples. Using only the camera pair's own evidence keeps the target
    clean: the union over all views piles slightly misaligned points in front of surfaces, which the
    coverage term turns into opaque sheets once 6-9 views are used."""
    import torch.nn.functional as F
    from .geometry import splat_depth
    per_view = [c.backproject(d)[a] for c, d, a in zip(cams, depths, anchors)]
    out = []
    for vc, (i, j) in virtual_cameras(cams, ts, with_pairs=True):
        d = splat_depth(vc, torch.cat([per_view[i], per_view[j]]))
        big = torch.where(d > 0, d, torch.full_like(d, float("inf")))
        d_min = -F.max_pool2d(-big[None, None], window, stride=1, padding=window // 2)[0, 0]
        out.append((vc, torch.where(torch.isfinite(d_min), d_min, torch.zeros_like(d_min))))
    return out
