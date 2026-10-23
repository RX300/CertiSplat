"""Point splatting into depth maps."""
import torch


def splat_depth(cam, points):
    """Z-buffered forward splat of world points into `cam` (2x2 bilinear footprint, nearest depth).
    Returns (H, W) camera-z depth, 0 where nothing lands."""
    u, v, z = cam.project(points)
    h, w = cam.height, cam.width
    ok = (z > 0) & (u > -1) & (v > -1) & (u < w) & (v < h)
    u, v, z = u[ok], v[ok], z[ok]
    x0, y0 = torch.floor(u).long(), torch.floor(v).long()
    zbuf = torch.full((h * w,), float("inf"), device=points.device)
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            zbuf.scatter_reduce_(0, yi[m] * w + xi[m], z[m], reduce="amin")
    zbuf[torch.isinf(zbuf)] = 0
    return zbuf.view(h, w)
