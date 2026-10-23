# Results (PSNR / SSIM / LPIPS)

Baselines: numbers reported by their papers. CertiSplat: single seed, StereoGS official evaluator.

## LLFF

| Method | 3 views | 6 views | 9 views |
|---|---|---|---|
| StereoGS + dropout (ECCV'26), reported | 22.05 / 0.783 / 0.147 | 25.40 / 0.856 / 0.102 | 26.44 / 0.883 / 0.082 |
| TWINGS (CVPR'26), reported | 21.49 / 0.754 / 0.167 | 24.97 / 0.844 / 0.106 | 26.23 / 0.876 / 0.085 |
| ICO-GS (CVPR'26), reported | 22.20 / 0.778 / 0.157 | 25.37 / 0.856 / 0.109 | 26.45 / 0.881 / 0.096 |
| **CertiSplat** | 22.54 / 0.792 / 0.145 | 25.86 / 0.864 / 0.100 | 26.76 / 0.885 / 0.087 |
| Δ vs best reported (per metric) | +0.34 / +0.009 / -0.002 | +0.46 / +0.008 / -0.002 | +0.31 / +0.002 / +0.005 |

## DTU

| Method | 3 views | 6 views | 9 views |
|---|---|---|---|
| StereoGS + dropout (ECCV'26), reported | 22.00 / 0.890 / 0.095 | 25.41 / 0.938 / 0.063 | 27.39 / 0.955 / 0.047 |
| TWINGS (CVPR'26), reported | 21.52 / 0.880 / 0.107 | 25.81 / 0.930 / 0.063 | 28.22 / 0.958 / 0.040 |
| ICO-GS (CVPR'26), reported | 21.77 / 0.888 / 0.092 | 25.09 / 0.928 / 0.064 | 27.19 / 0.953 / 0.045 |
| **CertiSplat** | 23.29 / 0.904 / 0.096 | 26.20 / 0.935 / 0.073 | 28.25 / 0.956 / 0.056 |
| Δ vs best reported (per metric) | +1.29 / +0.014 / +0.004 | +0.39 / -0.003 / +0.010 | +0.03 / -0.002 / +0.016 |

## Mip-NeRF360

| Method | 12 views | 24 views |
|---|---|---|
| StereoGS + dropout (ECCV'26), reported | 20.51 / 0.625 / 0.309 | 24.25 / 0.783 / 0.206 |
| TWINGS (CVPR'26), reported† | 20.35 / 0.618 / 0.368 | 24.17 / 0.762 / 0.252 |
| **CertiSplat** | not run | not run |
| Δ vs best reported (per metric) | – | – |

† TWINGS evaluates Mip-NeRF360 at 1/4 resolution (ours 1/8); excluded from the best-reported row.

## Blender

| Method | 8 views |
|---|---|
| StereoGS + dropout (ECCV'26), reported | 25.04 / 0.899 / 0.078 |
| ICO-GS (CVPR'26), reported | 25.56 / 0.884 / 0.100 |
| **CertiSplat** | not run |
| Δ vs best reported (per metric) | – |

## Reproduction check: local runs of the official code against the paper numbers

Same splits and bit-identical test GT as CertiSplat (docs/BASELINE_AUDIT.md); complete scene sets only.

| Method | Setting | Local run | Reported | Δ PSNR |
|---|---|---|---|---:|
| StereoGS | LLFF 3v | 21.98 / 0.774 / 0.156 | 22.05 / 0.783 / 0.147 | -0.07 |
| StereoGS | LLFF 6v | 25.28 / 0.854 / 0.109 | 25.40 / 0.856 / 0.102 | -0.12 |
| TWINGS | LLFF 3v | 21.37 / 0.752 / 0.167 | 21.49 / 0.754 / 0.167 | -0.12 |
| TWINGS | LLFF 6v | 25.00 / 0.844 / 0.104 | 24.97 / 0.844 / 0.106 | +0.03 |
| StereoGS | DTU 3v | 22.45 / 0.891 / 0.102 | 22.00 / 0.890 / 0.095 | +0.45 |
| TWINGS | DTU 3v | 21.64 / 0.878 / 0.109 | 21.52 / 0.880 / 0.107 | +0.12 |
