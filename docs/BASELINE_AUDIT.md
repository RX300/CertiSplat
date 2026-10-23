# Baseline audit (2026-10-20)

Question: do the local baseline runs in `benchmarks/first_pass/` use the official code and settings,
and why do they differ from the numbers the papers report?

**Verdict.** No defect was found in the pipeline. TWINGS and StereoGS reproduce their papers within
single-seed noise. StableGS is the one baseline whose paper numbers its released code does not
reproduce; this is not specific to our setup (see [StableGS](#stablegs)).

## Update 2026-10-21: 6/9-view runs; baselines now compared by their paper numbers

After the reproduction below, we decided to compare against the numbers the papers report and to stop
local baseline runs (`benchmarks/views_69` stopped at 01:52; `benchmarks/other_datasets` cancelled). The local
runs remain the evidence that those paper numbers are reproducible under our protocol.

* **Split and GT check, 6/9 views** (`scripts/verify_baselines.py --bench views_69`, CPU only,
  `benchmarks/views_69/split_gt_check.json`): all 21 completed TWINGS/StereoGS runs pass. `cameras.json`
  lists the test cameras, then the training cameras, in exactly the order of `fewshot.data.load_scene(...)`
  `.test` / `.train`; camera centres and rotations agree with CertiSplat's poses to ≤ 4e-7; every scored test GT
  PNG equals CertiSplat's GT (`View.load()`, 8-bit) with maximum pixel difference 0. The checker flags a wrong
  view count or scene (tested).
* **LLFF 6 views** (8 scenes, PSNR / SSIM / LPIPS):

  | Method | Local run | Paper | Δ PSNR |
  |---|---|---|---:|
  | StereoGS, dropout 0.3 | 25.28 / 0.854 / 0.109 | 25.40 / 0.856 / 0.102 | −0.12 |
  | TWINGS | 25.00 / 0.844 / 0.104 | 24.97 / 0.844 / 0.106 | +0.03 |

* **LLFF 9 views**: incomplete (TWINGS fern/flower/fortress/horns, StereoGS fern), so no mean is reported. Their
  partial means (27.61 / 27.84) exceed the papers by ~1.4 dB only because these are easy scenes: the same subsets
  score +1.69 (TWINGS) and +1.32 dB (StereoGS) above the 8-scene mean at 6 views. DTU 6/9 views was not run.
* StereoGS's local LPIPS is consistently ~0.007 above (worse than) its paper (LLFF 3/6v, DTU 3v); using the paper
  numbers is therefore the stricter comparison for CertiSplat.
* **Mip-NeRF360 and Blender paper numbers** (re-read from the arXiv HTML; in `scripts/report.py`): StereoGS v2
  Table 3 "Ours\*" 20.51 / 0.625 / 0.309 (12 views) and 24.25 / 0.783 / 0.206 (24 views), Table 4 Blender
  25.04 / 0.899 / 0.078, downsampling 8 (Mip-NeRF360) and 2 (Blender) as ours. TWINGS Table 2: 20.35 / 0.618 /
  0.368 and 24.17 / 0.762 / 0.252, but at ×4 downsampling ("×8, ×4, and ×4" for LLFF, DTU, Mip-NeRF360), so not
  comparable with our 1/8 and excluded from the best-reported row. ICO-GS Table 3 Blender 25.56 / 0.884 / 0.100
  (mean of three seeds); ICO-GS has no Mip-NeRF360 results and TWINGS none on Blender.

## Local runs against the papers (3 views, PSNR / SSIM / LPIPS)

| Method | Dataset | Local run | Paper | Δ PSNR |
|---|---|---|---|---:|
| StereoGS, dropout 0.3 (paper "Ours\*") | LLFF | 21.98 / 0.774 / 0.156 | 22.05 / 0.783 / 0.147 | −0.07 |
| | DTU | 22.45 / 0.891 / 0.102 | 22.00 / 0.890 / 0.095 | +0.45 |
| TWINGS | LLFF | 21.37 / 0.752 / 0.167 | 21.49 / 0.754 / 0.167 | −0.12 |
| | DTU | 21.64 / 0.878 / 0.109 | 21.52 / 0.880 / 0.107 | +0.12 |
| StableGS | LLFF | 20.86 / 0.718 / 0.206 | 21.19 / 0.726 / 0.208 | −0.33 |
| | DTU | 18.46 / 0.849 / 0.137 | 20.86 / 0.848 / 0.130 | −2.40 |

Paper values were re-read from the sources: [StereoGS arXiv v2](https://arxiv.org/html/2606.30545v2)
Tables 1–2 ("\*" = DropGaussian-style dropout; App. D: "a fixed dropout rate of 0.3"; the base
variant is 21.91 / 0.773 / 0.157 and 21.46 / 0.879 / 0.099), [TWINGS](https://arxiv.org/html/2605.22069)
Tables 3 and 1, [ICO-GS](https://arxiv.org/html/2603.02893) Tables 1–2 (a mean over three seeds), and
the [StableGS README](https://github.com/zyl123456aB/StableGS). The values in `scripts/report.py` are
transcribed correctly.

## Code: official repositories, algorithm code unchanged

| Method | Repository, commit | Local edits (`git diff`) |
|---|---|---|
| StableGS | `zyl123456aB/StableGS` `d18a8e2` | none; its own rasterizer is built into `runtimes/stablegs/` |
| StereoGS | `StringerYwh00/StereoGS` `2238f55` | 8 files: the authors' absolute checkpoint/repo paths → local paths; DINOv2 loaded from the bundled copy |
| TWINGS | `sandokim/TWINGS-official` `1585e89` | dataset root and model ids → local copies (`TWINGS-Init`); DepthPro `ToTensor()` removed as the README instructs |

StereoGS and TWINGS ship byte-identical rasterizer sources; the installed extension matches them.

## Settings: same flags as the official scripts

The official batch scripts hard-code the authors' paths and GPUs, so `scripts/benchmark.py` issues
the same Python commands itself (`manifest.json` lists every command).

* **StereoGS**: flag-for-flag the settings of `run_scripts/LLFF/run_llff.sh` and `run_scripts/DTU/run_dtu.sh`
  (LLFF: MVSAnywhere at 1440×1920, shift 4; DTU: MVSFormer, `img_dist_thresh` 50, shift 0.5; both:
  dropout 0.3, opacity decay 0.99 / 0.5, FoundationStereo from 20k of 30k iterations). The paper text names
  MVSAnywhere for initialisation, while the official DTU script and the README configuration table use MVSFormer;
  we follow the code. Dropout is active in training and test opacity is scaled by 0.7 at render, as upstream.
* **TWINGS**: `scripts/llff.sh` / `scripts/dtu.sh` (10k iterations, depth weight 0.03 / 0.01, pseudo
  weight 0.1); TWINGS-Init called with exactly the parameters of its `main()` for 3 views.
* **StableGS**: LLFF `train3.py`, rendering stage-two `iteration_5000`, as the author prescribes in
  [issue #1](https://github.com/zyl123456aB/StableGS/issues/1); DTU `train5.py` (the README entry point; the
  repository has no DTU script and the issue does not discuss DTU). scan110 uses the upstream `--rand_pcd`
  because the public 3-view COLMAP fusion is empty.

## Evaluation: identical for every method, including CertiSplat

* All baselines are scored by the unmodified StereoGS `metrics.py` on saved PNGs (LPIPS-VGG; DTU: white
  composite with the RegNeRF/IDR masks, PSNR over mask = 1 pixels). TWINGS' own `metrics_dtu.py`
  implements the same protocol. On LLFF, StableGS' and TWINGS' native float metrics agree with the
  shared evaluator to ≤ 0.001 dB.
* Training views are identical for all three baselines and CertiSplat on all 23 scenes, including the
  `linspace` ties (horns, fortress, trex, leaves); test GT images are bit-identical (max pixel difference 0).
* An independent re-implementation of the DTU evaluator reproduces every baseline mean to 3 decimals.
* The official evaluator applied to CertiSplat's saved renders reproduces its `metrics.json` to < 1e-5
  in all six settings (LLFF and DTU, 3/6/9 views).
* DTU masks (1600×1200) align with the 1588×1191 `dtu_corgs` images to within ~0.5–1.5 px at the
  evaluation resolution (measured by edge alignment); this is shared by all methods and by TWINGS'
  own paper setup, which uses the same data and masks.
* Logs: no errors or NaNs; StereoGS starts from its dense MVS clouds (0.3–0.66 M points LLFF, 38–220 k DTU)
  and loads FoundationStereo; the one failed attempt (TWINGS scan8) was a GUI-port clash, re-run with `--port 0`.

## StableGS

* LLFF (−0.33 dB) is in line with an independent attempt in issue #1: train5.py gave 19.02 vs 19.98
  on Mip-NeRF360, train3.py at iteration 10000 gave ~20.17 on LLFF; the author's answer was train3.py at
  iteration 5000, which is what we ran.
* DTU (−2.40 dB): our 18.46 / 0.849 / 0.137 matches TWINGS' own measurement of DropGaussian, the code
  StableGS is built on (18.41 / 0.849 / 0.136, TWINGS Table 1). SSIM equals the paper's 0.848 while PSNR
  is 2.4 dB lower. Counting the background in PSNR adds +3.02 to +3.07 dB to every method (StableGS 21.53,
  TWINGS 24.66, StereoGS 25.50), so neither protocol yields 20.86. Identical stage-one code differs by
  0.5–1 dB per scene between two runs (unmasked; scan8 14.03 vs 15.01), so single-scene comparisons are noisy.
* Diagnostic ([`benchmarks/diagnostics/stablegs_dtu_train3/`](../benchmarks/diagnostics/stablegs_dtu_train3/),
  same data, initialisation and scoring; one run each):

  | StableGS, DTU 3 views | PSNR / SSIM / LPIPS |
  |---|---|
  | `train5.py`, stage-two iteration 5000 (benchmark row) | 18.46 / 0.849 / 0.137 |
  | `train3.py`, stage-two iteration 5000 | 18.82 / 0.796 / 0.218 |
  | stage one only (DropGaussian-like), iteration 10000 | 18.42 / 0.844 / 0.136 |
  | paper | 20.86 / 0.848 / 0.130 |

  No released configuration reproduces the paper's DTU score, and the probing stage adds
  ~0 dB over stage one on DTU. The benchmark keeps `train5.py` (best SSIM/LPIPS, documented entry point).
* Pitfall: `train5.py` also writes test renders during training while `Scene(shuffle=True)` has shuffled
  the test cameras, so `test/ours_*` files written by training are in shuffled order. Every benchmark score
  comes from `render.py` (unshuffled), which overwrites all 25 `ours_5000` files; their GT is bit-identical
  to the canonical order. Never score training-time renders directly.
