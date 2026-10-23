# Why sparse-view 3DGS fails: diagnostics behind the method

**Historical diagnostic record, not the current benchmark.** The following probes predate the
2026-10-21 unification of all view counts to `fused_np`. Old three-view full-model scores, oracle
gaps, error budgets and ensemble deltas below must not be compared with the new main results.
They remain only to explain research decisions; current results and matched six-view ablations
come from the canonical saved runs described in `experiments/dev/README.md`. None of the all-view
reference geometry or test-derived oracle predictions is an input to the current training pipeline.

All numbers: 3 training views, LLFF at 1/8 resolution (8 scenes), DTU at 1/4 with object masks
(15 scenes unless stated), metrics computed with the unmodified StereoGS evaluator. Test images
are used only for these diagnostics, never by a method. Reference geometry for analysis comes
from COLMAP PatchMatch MVS over *all* images of a scene (an analysis oracle). Scripts that
produced these numbers live in the session scratchpad; the conclusions and the settings needed
to reproduce them are recorded here.

## 1. Where the error is (StereoGS, the strongest local baseline)

Test pixels classified by how many training views see the surface they show (DA3 all-view depth,
3% occlusion tolerance):

| LLFF | area | share of squared error | PSNR inside class |
|---|---:|---:|---:|
| seen by >= 2 training views | 70.6% | 33.5% | 23.8 |
| seen by 1 training view | 18.3% | 23.7% | 19.4 |
| occluded in all training views | 7.2% | 17.1% | 16.8 |
| outside all training frusta | 3.9% | 25.8% | 12.3 |

On DTU (inside object masks) the multi-view class dominates: 84-87% of the area and ~55-59% of
the error.

Oracle probes on StereoGS renders (LLFF mean 21.98 dB):

* per-view colour correction fitted on GT: +0.23 dB -> exposure is not the problem;
* optical-flow warp of the render onto GT: +1.89 dB (median shift 0.7 px, p90 6.4 px). **This
  oracle overstates geometric error** (see section 5): models trained on all non-test views,
  whose geometry is essentially correct, gain +1.0 to +3.3 dB from the same warp. Blurring GT by
  sigma = 1 px costs ~5 dB, so these scenes are full of high-frequency texture;
* averaging the StereoGS and TWINGS renders: +0.40 dB (LLFF), +0.37 dB (DTU) -> part of the error
  is method-specific variance;
* filling never-observed pixels with blurred GT: +0.46 dB (outside-frustum only); naive 2-D
  filling hurts.

## 2. Geometric precision, not prior strength, is the bottleneck

Same training recipe (dropout 0.3, adaptive opacity decay, 10k iterations), only the depth used to
initialise the Gaussians changes (fern):

| initial depth | PSNR |
|---|---:|
| StereoGS's MVSAnywhere depth (as StereoGS) | 23.73 |
| DA3 (multi-view, known poses) | 22.48 |
| DA3, oracle depth substituted in co-visible pixels only | **24.82** |
| oracle depth everywhere | 25.00 |

Precise depth in pixels that two training views share is worth +1.1 to +2.3 dB, and those pixels
are, in principle, triangulable from the three views. Photometric 3DGS optimisation does not
recover this precision by itself: the final model keeps roughly the precision of its
initialisation (loss terms that try to triangulate during training, such as multi-view NCC or
cross-view re-coloured rendering, did not close the gap).

Measured precision of depth sources in co-visible pixels (pooled LLFF, vs the all-view reference):

| source (3 training views only) | coverage | median rel. error | < 1% |
|---|---:|---:|---:|
| FoundationStereo on rectified **real** pairs | 57% | 0.45-0.49% | 67-70% |
| COLMAP PatchMatch (3 views, 2016 px) | 46% | 0.44-0.48% | 70% |
| both stereo pairs agree | - | 0.33% | 81% |
| stereo and 3-view MVS agree | - | 0.28% | 87% |
| MVSAnywhere (StereoGS's initialisation) | 62% | 1.24-1.33% | 41-43% |
| DA3-GIANT (504 px) | 100% | 3.5-4.0% | 16-18% |

In single-view regions every learned prior (DA3, DA3 at 1008 px, MapAnything, VGGT-Omega and their
medians, raw or corrected) has ~10-14% median error: an information limit, not a model-choice
problem. Coverage of precise depth matters more than rejecting its rare outliers: a stricter
fusion that discarded single-pair stereo lost 0.6-1.4 dB per scene.

## 3. DTU-specific failure modes

* Priors are already precise on DTU (scan82, co-visible foreground: stereo 0.15%, DA3 0.39%), yet a
  plain recipe trails StereoGS by ~2 dB. The loss concentrates in a few far, low test views
  (e.g. scan82 views 42-45: -11 to -15 dB), while near views are often better than StereoGS.
* In those views the object turns semi-transparent and the white table shows through. Only 0.6%
  of the compositing weight there comes from Gaussians in front of the true surface; the
  contributing splats are extremely flat (median anisotropy ~93:1): disks facing the training
  cameras that become edge-on from oblique views.
* StereoGS depends on its virtual-view stereo teacher for exactly this: without it, StereoGS scores
  18.85 dB on scan82 (23.21 with it).

## 4. Negative results (kept for the record)

| idea | outcome |
|---|---|
| posterior-mean "hedging" by inflating splats along their depth-uncertainty direction | 0 to -0.6 dB |
| Monte-Carlo dropout averaging instead of opacity scaling | +0.04 dB |
| image-based rendering (warping training photos) with predicted or verified depth | -2 to -6 dB vs the 3DGS render |
| heuristic completion of unobserved space (border extrapolation, background continuation), with or without space carving | -0.5 to -2.9 dB |
| depth supervision with the fused depth | LLFF +0.07, DTU -2.6 dB |
| longer training (20k/30k) | LLFF 0, DTU unstable |
| global anisotropy cap on LLFF | hurts thin structures (trex -1.4 dB) |

The old project's method was additionally trained with a half-pixel principal-point offset
(OpenCV centre convention applied to COLMAP intrinsics), which shifts every render by 0.5 px.

## 5. After certified geometry: where the remaining error is (October 2026 re-analysis)

Model: the 3-view LLFF configuration with certified fused depth (22.53 dB, seed 0). Classes as in
section 1; "dense" = the same recipe trained on all non-test images (an information upper bound,
under-trained on leaves/trex where it scores below the 3-view model).

| LLFF class | share of our squared error | our PSNR in class | dense model | pooled gain if matched |
|---|---:|---:|---:|---:|
| seen by >= 2 training views | 34.8% | 24.30 | 25.25 | +0.31 dB |
| seen by 1 training view | 22.7% | 20.27 | 21.97 | +0.33 dB |
| occluded in all training views | 20.0% | 16.79 | 20.13 | +0.49 dB |
| outside all training frusta | 22.5% | 13.62 | 18.61 | +0.73 dB |

* **Co-visible geometry is near its 3-view limit.** Initialising from all-view COLMAP depth
  (oracle geometry, same recipe) gives fern 25.00 / flower 22.14 dB against 24.48 / 21.96 for
  certified depth; in co-visible pixels oracle geometry still trails the dense model by 1.5 dB
  (fern) and 5.9 dB (flower: wind-moved foliage), i.e. the rest is appearance and motion
  information that three photographs do not contain.
* **The flow-warp oracle is not a geometry measure.** Dense models gain +1.05 (fern, horns),
  +2.02 (flower) and +3.28 dB (room) from it; our fern model gains +0.80 dB, the oracle-geometry
  model +0.78 dB.
* **Lens distortion is not a factor.** LLFF cameras are SIMPLE_RADIAL (k = 0.007-0.037), which
  every sparse-view method, including ours and the baselines, ignores (pinhole reprojection error of
  COLMAP's own keypoints: 0.5-0.8 px at benchmark size vs 0.15-0.2 px with k). Rendering through the
  distortion (exact per-camera warping of splat centres) changes a dense fern model by -0.01 dB and
  3-view leaves/orchids by -0.30/-0.01 dB: forward-facing captures absorb a common radial
  distortion into slightly warped geometry.
* **Seed variance is broad.** Averaging the renders of three seeds gives +0.36 dB (22.53 -> 22.89);
  the squared error drops by 6.6% (co-visible), 8.1% (single-view), 9.4% (occluded) and 2.1%
  (outside frusta) within each class. The gain is the same with view-independent colour (SH 0:
  +0.29 dB) and is not a low-pass effect (best blur of one render: +0.05 dB).
* **Bounds for filling never-observed content.** Replacing never-observed pixels of our renders
  by blurred GT gives +1.25 / +0.74 / +0.18 dB at sigma = 2 / 4 / 8 px (outside-frustum pixels
  only: +0.60 / +0.50 / +0.37 dB). On DTU (masked) the same bound is +0.53 / +0.42 / +0.26 dB:
  never-observed object pixels are only ~3-5% of the object even in far views.

DTU far views (test ids 1, 2, 9-12, 41-47; 17-22 dB against 24-28 dB near the training cameras)
fail mostly on co-visible pixels: a per-view affine colour fit on GT adds +0.9 dB on average and
+3.0 dB on view 2 (the GT is 6-12% darker than the render in views 2 and 45-47), and the median
misregistration there is 1.3-1.8 px against ~0.2 px near the training cameras.

### Negative probes (test-time, on the trained 3-view LLFF models)

| probe | effect |
|---|---:|
| Monte-Carlo depth jitter of single-view splats along their source ray (sigma 2 / 5 / 10%), 8 renders averaged | +0.01 / +0.00 / -0.03 dB |
| GP interpolation of colour over the training-view directions instead of SH | -0.28 to -0.41 dB |
| image-based rendering through the model's depth (pixels seen by >= 2 views) | -3.2 dB on covered pixels |
| depth-normal consistency (splats in the surface tangent plane), training | DTU +0.79 dB vs carving only, -1.27 vs teacher + cap; LLFF -0.44 dB |

### Error budget versus number of views (LLFF, our models)

Classes as above, but visibility tested against all-view COLMAP depth maps of the N training views
(holes treated as visible), which moves some pixels from "seen" to "occluded" relative to the
DA3-based 3-view table; shares of squared error:

| views | seen by >= 2 | seen by 1 | occluded in all | outside all frusta |
|---:|---:|---:|---:|---:|
| 3 | 27.9% | 21.9% | 27.6% | 22.5% |
| 6 | 48.4% | 18.4% | 22.5% | 10.8% |
| 9 | 54.4% | 17.8% | 19.5% | 8.4% |

With more views the error moves into observed content, where precision of geometry and appearance
decides; with three views half of it lies in content no or one training view observes.

### Instability of the virtual-view teacher with more views (DTU)

With the teacher's targets built from the certified points of *all* training views, DTU at 6 views
showed random collapses of individual test views (down to 7-9 dB; 7 of 15 scans had a test view
below 15 dB, against none at 3 views). Seed spread on scan30: 23.5 / 26.6 / 29.3 dB. The cause is a
white semi-transparent sheet that forms during training (it is absent from the initial points):
it projects onto the white table in every training view, so the photometric loss accepts it, and
covers the object from the neighbouring low test views. Diagnosis on scan30 (6 views, seeds 0/1):
no teacher 29.5 / 29.5; no carving 27.8 / 29.4; teacher cameras restricted to within 45 degrees of a
training camera 25.1 / 23.6 (no fix); teacher targets from the two views each virtual camera
interpolates 29.9 / 29.8. Unioned targets accumulate slightly misaligned points in front of
surfaces, which the coverage term then makes opaque; per-pair targets avoid it.
