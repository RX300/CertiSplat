# Depth-prior check

All 3/6/9-view settings use fused_np without cross-view propagation; raw stereo, MVS and DA3 are also checked.

| Dataset | views | scenes | files checked | certified pixels | weighted scene-median rel. diff | < 1% |
|---|---:|---:|---:|---:|---:|---:|
| LLFF | 3 | 8 | 96 | 27.8% | 0.14% | 98.4% |
| LLFF | 6 | 8 | 192 | 37.2% | 0.21% | 92.4% |
| LLFF | 9 | 8 | 288 | 35.2% | 0.27% | 88.2% |
| DTU | 3 | 15 | 180 | 21.8% | 0.05% | 99.9% |
| DTU | 6 | 15 | 360 | 36.0% | 0.06% | 94.9% |
| DTU | 9 | 15 | 540 | 44.4% | 0.06% | 93.6% |

Certified fractions average equally over scenes and training views. Cross-view statistics compare projected certified source depths with certified depths at the nearest target pixel; the reported difference is a count-weighted mean of per-scene medians, and the < 1% rate uses the same projected pixel-pair counts. These are internal consistency measurements, not errors against reference geometry.
