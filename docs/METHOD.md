# CertiSplat: sparse-view Gaussian splatting from what the photographs certify

**Setting.** Per-scene 3D Gaussian Splatting from N posed RGB images (N = 3, 6, 9) on the LLFF and
DTU protocols of the local baselines. Frozen pretrained networks are used only to compute depth;
no network predicts Gaussians.

## Thesis

*Sparse-view Gaussian splatting is limited by how precisely geometry is pinned where the views
overlap. This precision cannot come from geometric priors, nor from photometric optimisation; only
the photographs themselves can certify it. The reconstruction should therefore be built on, and
constrained by, certified evidence — and by nothing else.*

The thesis was motivated by exploratory measurements recorded in [ANALYSIS.md](ANALYSIS.md).
Those studies predate the unified 3-view rerun: oracle comparisons and old 3-view error budgets
are diagnostics, not measurements of the current model. Their main observations were:

1. Precise depth in pixels that two training views share is worth +1 to +2 dB over a
   foundation-model initialisation (oracle substitution, same training).
2. Foundation depth models are 4-13x too imprecise there (median relative error 1.2-4% against
   0.3% for the certificates below), and in single-view regions every model errs by ~10%.
3. Photometric 3DGS optimisation keeps the precision of its initialisation: it does not
   triangulate, and losses that try to make it do so did not close the gap.
4. Unobserved content and seed variance remain after improving the initialization; the current
   unified model has not been rerun with oracle geometry or multiple seeds.

## 1. Certificates (`stereo_prior.py`, `mvs_prior.py`, `fuse.py`)

A pixel's depth is *certified* by agreement among estimates computed from the training photographs.
The two stages use different, fixed tests:

* learned stereo (FoundationStereo) on rectified pairs of real training images, each view paired
  with its two nearest training cameras; stereo estimates certify when their depth range is
  less than 1% of their mean (these estimates can come from the same learned algorithm);
* classical PatchMatch MVS on the N training images with fixed poses; stereo and MVS agreeing
  certify when each source differs from their mean by at most 1%, equivalent to a two-source
  range of at most 2%. This threshold is unchanged from the existing 6/9-view implementation.

Current certified fractions and cross-view consistency are regenerated for all six settings in
[PRIORS_CHECK.md](PRIORS_CHECK.md), with exact values in `priors_check.json`. A single estimate is kept as a
lower tier; conflicting estimates are discarded.

## 2. What a certificate implies, and the constraint each implication becomes

A certified depth d at pixel p of training view i states three facts.

All 3/6/9-view settings use the same `fused_np` depth, without cross-view propagation.

**A surface is at d.** The Gaussians are initialised from the per-view depth: certified (and
single-estimate) pixels at their measured depth; every other pixel takes Depth Anything 3 rescaled
by a smooth correction field fitted on the certified pixels — a placeholder that the certificates
align but never enforce.

**The ray from view i to the surface is empty.** Free-space carving: every 10 iterations,
Gaussians lying in front of a training view's depth by more than a tier-dependent margin (3%
certified, 6% single estimate, 15% placeholder) lose half their opacity.

**The surface stays visible from viewpoints between the views that certified it.** For cameras
inter/extrapolated along each neighbouring pair of training cameras (t = -3 ... 4), the certified
points of *that pair's two views* are splatted into the virtual camera, where the model must be
opaque and at the certified depth (coverage + inverse-depth loss, weight 0.5). A certificate holds
near the views that produced it: pooling all views' certificates into every virtual camera stacks
slightly misaligned points in front of surfaces, and with 6-9 views the coverage term turned them
into opaque sheets (DTU 6 views: 25.44 dB pooled, 26.20 dB pair-local; collapsed test views down to
7 dB disappear). An anisotropy cap of 10:1 keeps certified surfaces from turning into needle-like
splats that vanish when seen edge-on from such viewpoints.

## 3. When certificates are enforced

Inside the angular cover of the training photographs the photometric loss already uses the same
evidence; enforcing the free-space and visibility facts there adds nothing but the rare errors of
certificates (a reflection certified at its mirrored depth, thin structures); historical development
probes found a cost of 0.2-1.4 dB on LLFF. Beyond the cover — object-centric captures whose test views
lie far from every training view — nothing else constrains those viewpoints. The matched six-view
constraint ablations are listed in the README (DTU, 6 views). The method decides from the cameras: the constraints are on when the training
cameras' optical axes span more than 45 degrees (DTU 72-93 degrees, LLFF 3-24 degrees, at every
view count). The certified initialisation is always used.

## 4. Training (`train.py`; the defaults are the method)

Standard regularised 3DGS for sparse views, not claimed as a contribution: 10k iterations,
spherical harmonics of degree 3, Gaussian dropout 0.3 (test opacity scaled by 0.7), StereoGS's
adaptive opacity decay (0.99), half of the per-view depth pixels as initial points. On DTU the
black-background loss of DNGaussian and StereoGS is kept as in those baselines (a pixel that is dark
together with the 49 pixels above it must render transparent; threshold 30/255, 15/255 on scan110).
One command for every dataset and view count (see the README).

## 5. Results (single seed; PSNR / SSIM / LPIPS)

The current six-setting table is generated in [RESULTS.md](RESULTS.md) from the canonical runs,
with reported baseline numbers and a separately identified local reproduction check.
All 23 three-view runs are regenerated with `fused_np`; the 46 six/nine-view runs are retained.
Regenerate with `bash scripts/run.sh tools scripts/report.py`. Historical cost per scene at 3 views was about
4-5 minutes (stereo 12 s, MVS 2.3 min, DA3 a few seconds, training 1.7 min) against 55 min for
StereoGS and 14 min for TWINGS.

## 6. Ablations along the thesis

The matched current ablations are DTU six-view runs without carving, without visibility
supervision, and without the anisotropy cap (15 scenes each, the same `fused_np` prior).
The previous three-view ablations used another prior, so their records and figure are retired;
they are not compared against the new three-view full model. A matched three-view initialization
or constraint ablation remains unmeasured and is outside this rerun.

## 7. Evaluated and not adopted

Historical development diagnostics, not rerun against the unified three-view setting (single seed;
LLFF over 8 scenes unless noted): prompted
depth completion for the placeholder (LDCM, half the error of the fill: -0.3 dB), cross-view
propagation of certificates (LLFF -0.08), depth-normal consistency (LLFF -0.44), angular-novelty
or coverage-only virtual supervision on LLFF (-0.2 to -0.9), a ray-extent shape regulariser
(+0.04), a relative Adam epsilon (-0.03), anisotropy cap 30 on LLFF (-0.07), modelling LLFF's radial
lens distortion (0.0), stereo-only certificates at 6 views (-0.07), virtual cameras restricted to
45 degrees from the training views (no fix for the pooled-target instability). Test-time probes
(posterior-mean depth jitter, colour interpolation over view directions, image-based rendering)
are in ANALYSIS.md section 5. Ensemble averaging of three seeds adds +0.36 dB (LLFF) and +0.33 dB
(DTU); ensemble distillation (`distill.py`) is on hold and outside the thesis.
