"""Image losses and the benchmark metrics.

Metrics reuse the unmodified StereoGS implementations (PSNR, 11x11 SSIM, LPIPS-VGG v0.1),
applied to 8-bit quantised renders exactly as its render.py + metrics.py pipeline does.
DTU: render and GT are composited onto white with the object mask; PSNR counts mask == 1
pixels, SSIM/LPIPS use the composited images.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_STEREOGS = Path(__file__).resolve().parents[1] / "StereoGS"
_LPIPS = None


def _gaussian_window(channels, device, size=11, sigma=1.5):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim_map(a, b):
    """SSIM map of (C, H, W) images with the standard 11x11, sigma 1.5 window."""
    c = a.shape[0]
    w = _gaussian_window(c, a.device)
    a, b = a[None], b[None]
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=c), F.conv2d(b, w, padding=5, groups=c)
    var_a = F.conv2d(a * a, w, padding=5, groups=c) - mu_a ** 2
    var_b = F.conv2d(b * b, w, padding=5, groups=c) - mu_b ** 2
    cov = F.conv2d(a * b, w, padding=5, groups=c) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))[0]


def photometric_loss(pred, gt, lambda_dssim=0.2, weight=None):
    """(1 - l) L1 + l (1 - SSIM); `weight` (H, W) optionally masks pixels."""
    l1 = (pred - gt).abs().mean(0)
    dssim = 1 - ssim_map(pred, gt).mean(0)
    loss = (1 - lambda_dssim) * l1 + lambda_dssim * dssim
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def quantize(image):
    """torchvision.save_image rounding, as used by the baselines' render scripts."""
    return image.clamp(0, 1).mul(255).add(0.5).floor().clamp(0, 255) / 255


@torch.no_grad()
def benchmark_metrics(render, gt, mask=None):
    """render, gt: (3, H, W) in [0, 1]; mask: optional (3, H, W) DTU mask. Returns dict."""
    global _LPIPS
    if str(_STEREOGS) not in sys.path:
        sys.path.insert(0, str(_STEREOGS))
    from lpipsPyTorch import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    if _LPIPS is None:
        _LPIPS = LPIPS(net_type="vgg", version="0.1").eval().cuda()
    render, gt = quantize(render)[None], quantize(gt)[None]
    if mask is None:
        mask = torch.ones_like(render[0])
    render = render * mask + (1 - mask)
    gt = gt * mask + (1 - mask)
    return {"PSNR": float(psnr(render, gt, mask=mask[None])),
            "SSIM": float(ssim(render, gt)),
            "LPIPS": float(_LPIPS(render, gt).mean())}
