"""Pinhole cameras for the CUDA rasterizer, with exact intrinsics (COLMAP pixel convention)."""
import numpy as np
import torch
import torch.nn.functional as F

NEAR, FAR = 0.01, 1000.0


def _tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class Camera:
    def __init__(self, K, w2c, height, width, image=None, name="", device="cuda"):
        self.name = name
        self.height, self.width = int(height), int(width)
        self.K = _tensor(K, device)
        self.w2c = _tensor(w2c, device)
        self.image = None if image is None else torch.as_tensor(
            np.ascontiguousarray(image), dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        # The rasterizer maps NDC to pixel index p = ((v + 1) * S - 1) / 2. With
        # P[0, 2] = 2 cx / W - 1 this gives p = fx x / z + cx - 0.5, i.e. COLMAP's centre-at-0.5.
        P = torch.zeros(4, 4, device=device)
        P[0, 0] = 2 * fx / self.width
        P[1, 1] = 2 * fy / self.height
        P[0, 2] = 2 * cx / self.width - 1
        P[1, 2] = 2 * cy / self.height - 1
        P[2, 2] = FAR / (FAR - NEAR)
        P[2, 3] = -FAR * NEAR / (FAR - NEAR)
        P[3, 2] = 1
        self.world_view_transform = self.w2c.T.contiguous()
        self.full_proj_transform = (P @ self.w2c).T.contiguous()
        self.c2w = torch.linalg.inv(self.w2c)
        self.center = self.c2w[:3, 3].contiguous()
        self.tanfovx = float(self.width / (2 * fx))
        self.tanfovy = float(self.height / (2 * fy))

    @classmethod
    def from_view(cls, view, with_image=True, device="cuda"):
        return cls(view.K, view.w2c, view.height, view.width,
                   view.load() if with_image else None, view.name, device)

    def pixel_rays(self):
        """World-space ray directions (H, W, 3) through pixel centres, with unit camera-z."""
        ys, xs = torch.meshgrid(torch.arange(self.height, device=self.K.device, dtype=torch.float32),
                                torch.arange(self.width, device=self.K.device, dtype=torch.float32),
                                indexing="ij")
        pixels = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], -1)
        return (pixels @ torch.linalg.inv(self.K).T) @ self.c2w[:3, :3].T

    def backproject(self, depth):
        """Camera-z depth (H, W) -> world points (H, W, 3)."""
        return self.center + self.pixel_rays() * depth[..., None]

    def project(self, points):
        """World points (..., 3) -> pixel-index coordinates u, v and camera-z depth."""
        local = points @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        z = local[..., 2]
        uvw = local @ self.K.T
        safe = uvw[..., 2].clamp_min(1e-8)
        return uvw[..., 0] / safe - 0.5, uvw[..., 1] / safe - 0.5, z

    def shifted(self, offset):
        """Camera translated by `offset` (3,) expressed in this camera's axes."""
        w2c = self.w2c.clone()
        w2c[:3, 3] -= torch.as_tensor(offset, dtype=torch.float32, device=w2c.device)
        return Camera(self.K, w2c, self.height, self.width, None, self.name + "_shift", self.K.device)


def sample_image(image, u, v, mode="bilinear", padding="border"):
    """Sample (C, H, W) at pixel-index coordinates u, v (same shape) -> (C, *u.shape)."""
    h, w = image.shape[-2:]
    grid = torch.stack([(2 * u + 1) / w - 1, (2 * v + 1) / h - 1], -1).reshape(1, -1, 1, 2)
    out = F.grid_sample(image[None], grid, mode=mode, padding_mode=padding, align_corners=False)
    return out[0, :, :, 0].reshape(image.shape[0], *u.shape)


def neighbor_pairs(centers, k=2):
    """Unordered pairs (i < j) linking every camera to its k nearest cameras by centre distance.
    For three cameras with k = 2 this is every pair; for more views it stays O(k n)."""
    centers = np.asarray(centers, dtype=np.float64)
    pairs = set()
    for i in range(len(centers)):
        distances = np.linalg.norm(centers - centers[i], axis=1)
        for j in np.argsort(distances)[1:k + 1]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)
