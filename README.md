# CertiSplat

**Sparse-view 3D Gaussian Splatting from what the photographs certify.**

[English](#english) · [中文](#中文) · [日本語](#日本語)

> This project was produced with **Claude Opus 5.5 + Astra** (AI research/coding agents).
> It will not be submitted to a venue. Anyone is welcome to use, modify and improve it —
> no need to contact the authors.
>
> 本项目由 **Claude Opus 5.5 + Astra** 完成，不会投稿。欢迎任何人自由使用和改进，无需联系作者。
>
> 本プロジェクトは **Claude Opus 5.5 + Astra** によって作成されました。投稿予定はありません。
> どなたでも自由に利用・改良していただけます。作者への連絡は不要です。

---

## English

### What it does

Per-scene 3D Gaussian Splatting from **3, 6 or 9 posed photographs** (LLFF at 1/8 resolution,
DTU at 1/4 resolution with object-masked evaluation). No network predicts Gaussians; frozen
pretrained networks are only used to estimate depth.

### Method (overview)

Thesis: sparse-view 3DGS is limited by how precisely geometry is pinned where the views overlap,
and only the photographs themselves can certify that precision.

1. **Depth certificates.** A pixel's depth is *certified* when independent estimates agree:
   two learned-stereo pairs (FoundationStereo on rectified pairs of real training images) agree
   within 1%, or stereo and classical PatchMatch MVS (COLMAP, fixed poses) agree within 1% of their mean.
   Remaining pixels use Depth Anything 3 rescaled to the certified pixels (a placeholder, never enforced).
2. **Each certificate implies three constraints.** A surface is there (initialisation); the ray in
   front of it is empty (free-space carving); the surface stays visible from virtual viewpoints between
   the two views that certified it (pair-local virtual-view supervision) plus a 10:1 anisotropy cap.
3. **When to enforce them** is decided from the cameras: the far-view constraints switch on only when the
   training cameras' optical axes span more than 45° (object-centric DTU), not for forward-facing LLFF.
4. Training is otherwise standard regularised sparse-view 3DGS (10k iterations, Gaussian dropout,
   adaptive opacity decay). One recipe for every dataset and view count — no per-dataset tuning.

Details: [docs/METHOD.md](docs/METHOD.md), measurements behind the thesis: [docs/ANALYSIS.md](docs/ANALYSIS.md).

### Results (single seed, PSNR↑ / SSIM↑ / LPIPS↓)

Baselines are the numbers reported in their papers; CertiSplat uses the StereoGS official evaluator
with the same splits. Per-scene numbers: [docs/per_scene_results.csv](docs/per_scene_results.csv).

**LLFF (8 scenes)**

| Method | 3 views | 6 views | 9 views |
|---|---|---|---|
| StereoGS + dropout (ECCV'26) | 22.05 / 0.783 / 0.147 | 25.40 / 0.856 / 0.102 | 26.44 / 0.883 / 0.082 |
| TWINGS (CVPR'26) | 21.49 / 0.754 / 0.167 | 24.97 / 0.844 / 0.106 | 26.23 / 0.876 / 0.085 |
| ICO-GS (CVPR'26) | 22.20 / 0.778 / 0.157 | 25.37 / 0.856 / 0.109 | 26.45 / 0.881 / 0.096 |
| **CertiSplat** | **22.54 / 0.792 / 0.145** | **25.86 / 0.864 / 0.100** | **26.76 / 0.885** / 0.087 |

**DTU (15 scenes)**

| Method | 3 views | 6 views | 9 views |
|---|---|---|---|
| StereoGS + dropout (ECCV'26) | 22.00 / 0.890 / 0.095 | 25.41 / **0.938** / **0.063** | 27.39 / 0.955 / 0.047 |
| TWINGS (CVPR'26) | 21.52 / 0.880 / 0.107 | 25.81 / 0.930 / **0.063** | 28.22 / **0.958** / **0.040** |
| ICO-GS (CVPR'26) | 21.77 / 0.888 / **0.092** | 25.09 / 0.928 / 0.064 | 27.19 / 0.953 / 0.045 |
| **CertiSplat** | **23.29 / 0.904** / 0.096 | **26.20** / 0.935 / 0.073 | **28.25** / 0.956 / 0.056 |

**Ablation (DTU, 6 views, PSNR):** full 26.20 · w/o anisotropy cap 26.05 · w/o free-space carving 25.95 ·
w/o virtual-view visibility 25.75.

PSNR is best in all six settings; LPIPS on DTU is weaker than the best baselines. Mip-NeRF360 and
Blender were not run. Per-scene cost at 3 views is roughly 4–5 minutes on one RTX 6000 Ada
(stereo 12 s, MVS 2.3 min, DA3 a few seconds, training ~1.7 min).
Local reproduction of the baselines against their papers: [docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md).

### Environment setup

Tested on Linux, Python 3.10, PyTorch 2.5.1 + CUDA 12.1, NVIDIA RTX 6000 Ada.
A full package snapshot is in [environment.resolved.yml](environment.resolved.yml) /
[requirements.lock.txt](requirements.lock.txt).

```bash
git clone https://github.com/RX300/CertiSplat.git && cd CertiSplat

# 1. Main environment
conda create -n certisplat python=3.10 -y && conda activate certisplat
conda install -c conda-forge "colmap=3.8=gpu*" -y          # CUDA COLMAP for PatchMatch MVS
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 scipy==1.15.3 pillow plyfile==1.1.3 opencv-python omegaconf==2.4.0 \
            timm==0.9.16 einops loguru xformers==0.0.29

# 2. StereoGS (provides FoundationStereo code, the CUDA rasterizer and the official metrics)
git clone https://github.com/StringerYwh00/StereoGS.git
git -C StereoGS checkout 2238f559ce6da2c8f9b0e98110eedc7e53ea952b
git -C StereoGS apply ../patches/stereogs_local_paths.patch   # replaces the upstream authors' absolute paths
export CUDA_HOME=/usr/local/cuda-12.1                          # any CUDA 12.1 toolkit
pip install --no-build-isolation StereoGS/submodules/diff-gaussian-rasterization StereoGS/submodules/simple-knn

# 3. Weights
#   checkpoints/FoundationStereo/vit_small/{cfg.yaml,model_best_bp2.pth}  (FoundationStereo ViT-S, NVlabs release
#       https://github.com/NVlabs/FoundationStereo; mirror: huggingface.co/shriarul5273/FoundationStereo_models, folder 11-33-40)
#   checkpoints/edgenext_small/pytorch_model.bin                           (huggingface.co/timm/edgenext_small.usi_in1k)
```

**Depth Anything 3** runs in its own environment (Python 3.11, PyTorch 2.6 in our setup):
clone <https://github.com/ByteDance-Seed/Depth-Anything-3> (we used commit `3d835ec`) to
`third_party/Depth-Anything-3`, install it following its README, and put the
`DA3-GIANT-1.1` weights (`config.json`, `model.safetensors`) in `checkpoints/DA3-GIANT-1.1`.
Paths can be overridden with `DA3_SOURCE`, `DA3_WEIGHTS`, and `DA3_PYTHON` (the DA3 environment's Python).

**Data** go to `datasets/` (or set `CERTISPLAT_DATA`):

| Path | Source |
|---|---|
| `datasets/nerf_llff_data/<scene>` | LLFF from the official NeRF data (`nerf_llff_data.zip`) |
| `datasets/DTU/dtu_corgs/<scan>` | DTU few-shot package of [CoR-GS](https://github.com/jiaw-z/CoR-GS) (15 scans) |
| `datasets/DTU/submission_data/idrmasks` | RegNeRF evaluation masks (backup linked by DNGaussian) |

`scripts/run.sh` activates the conda env (`CERTISPLAT_ENV`, default `certisplat`) and sets `CUDA_HOME`,
`TORCH_HOME=checkpoints/torch`, `HF_HOME=checkpoints/huggingface`.

### Running

```bash
# Everything for a setting (priors -> fusion -> training -> evaluation):
CUDA_VISIBLE_DEVICES=0 DA3_PYTHON=/path/to/da3-env/bin/python \
  bash scripts/run.sh tools scripts/run_certisplat.py LLFF:3 DTU:3

# Or step by step
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh tools -m fewshot.stereo_prior --dataset LLFF --all --n-views 3
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh tools -m fewshot.mvs_prior    --dataset LLFF --all --n-views 3
CUDA_VISIBLE_DEVICES=0 $DA3_PYTHON -m fewshot.da3_prior --dataset LLFF --n-views 3
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh tools -m fewshot.fuse --dataset LLFF --n-views 3 --name fused_np
bash scripts/run.sh tools scripts/sweep.py run --name certisplat --dataset LLFF --gpu 0 --n-views 3 -- \
    --init npz:experiments/priors/LLFF/{views}v/{scene}/fused_np/init_points.npz
bash scripts/run.sh tools scripts/report.py     # results table
```

Outputs: priors in `experiments/priors/<dataset>/<n>v/<scene>/`, runs in
`experiments/dev/certisplat/<dataset>[_<n>v]/<scene>/seed0/` (`metrics.json`, renders, point cloud).

### Code layout

| Path | Content |
|---|---|
| `fewshot/` | data/splits, cameras, Gaussians, renderer, metrics, training (`train.py`), depth priors (`stereo_prior.py`, `mvs_prior.py`, `da3_prior.py`), certificate fusion (`fuse.py`), carving + virtual-view teacher (`carve.py`) |
| `scripts/` | end-to-end runner, sweeps, report, verification tools, baseline runner (`benchmark.py`, needs the baseline repos) |
| `docs/` | method, analysis, baseline audit, prior checks, results |
| `patches/` | path fixes needed to run the upstream StereoGS / FoundationStereo code |

### License

Original CertiSplat code is released under the [MIT License](LICENSE). Note that
`fewshot/gaussians.py` follows the densification logic of 3D Gaussian Splatting (Inria, non-commercial
research license), and the required third-party code and weights (StereoGS / diff-gaussian-rasterization,
FoundationStereo, Depth Anything 3 Giant) carry their own, mostly **non-commercial**, licenses.

---

## 中文

### 简介

从 **3 / 6 / 9 张已知位姿照片** 做逐场景 3D Gaussian Splatting（LLFF 1/8 分辨率；DTU 1/4 分辨率，物体 mask 评测）。
不使用网络直接预测高斯，冻结的预训练网络只用于估计深度。

### 方法概要

核心观点：稀疏视角 3DGS 的瓶颈在于视角重叠区域几何被"钉住"的精度，而这种精度只能由照片本身来"认证"。

1. **深度认证（certificate）**：当独立估计一致时认为该像素深度被认证——两组学习型双目（FoundationStereo，基于训练图像的校正图像对）
   相差 < 1%，或双目与经典 PatchMatch MVS（COLMAP，固定位姿）相对均值偏差 ≤ 1%。其余像素用按认证像素校正尺度的
   Depth Anything 3 填充（仅作占位，不施加约束）。
2. **每个认证蕴含三个约束**：该处有表面（初始化）；表面前方光线为空（free-space carving）；在认证它的两视角之间的虚拟视角中表面保持可见
   （成对局部虚拟视角监督），并限制高斯各向异性 ≤ 10:1。
3. **何时启用**由相机决定：仅当训练相机光轴夹角跨度 > 45°（如物体中心的 DTU）时启用远视角约束，前向的 LLFF 不启用。
4. 其余为标准的稀疏视角正则化 3DGS（10k 迭代、Gaussian dropout、自适应不透明度衰减）。所有数据集与视角数使用同一套配置，无逐数据集调参。

详见 [docs/METHOD.md](docs/METHOD.md)、[docs/ANALYSIS.md](docs/ANALYSIS.md)。

### 性能（单种子，PSNR↑ / SSIM↑ / LPIPS↓）

基线数字取自各自论文；CertiSplat 使用 StereoGS 官方评测器与相同划分。表格见上方 [English → Results](#english)。

- LLFF：3/6/9 视角 PSNR 分别为 **22.54 / 25.86 / 26.76**，比论文报告的最佳基线高 +0.34 / +0.46 / +0.31 dB。
- DTU：3/6/9 视角 PSNR 分别为 **23.29 / 26.20 / 28.25**，比最佳基线高 +1.29 / +0.39 / +0.03 dB；但 DTU 上 LPIPS 弱于最佳基线。
- 消融（DTU 6 视角 PSNR）：完整 26.20 · 去掉各向异性限制 26.05 · 去掉 carving 25.95 · 去掉虚拟视角可见性 25.75。
- 3 视角下每场景约 4–5 分钟（RTX 6000 Ada）。Mip-NeRF360 与 Blender 未运行。

### 环境配置

见上方 [English → Environment setup](#environment-setup) 的命令（Linux、Python 3.10、PyTorch 2.5.1 + CUDA 12.1）。要点：

1. 创建 conda 环境 `certisplat`，安装 CUDA 版 COLMAP 3.8、PyTorch 2.5.1 (cu121) 及依赖；
2. 克隆 StereoGS 到项目根目录并切到指定 commit，应用 `patches/stereogs_local_paths.patch`，编译其中的 `diff-gaussian-rasterization` 与 `simple-knn`；
3. 下载 FoundationStereo ViT-S 与 edgenext_small 权重到 `checkpoints/`；
4. Depth Anything 3 在单独环境中运行（通过 `DA3_PYTHON`、`DA3_SOURCE`、`DA3_WEIGHTS` 指定）；
5. 数据放在 `datasets/`（或设置 `CERTISPLAT_DATA`）。

### 许可

CertiSplat 原创代码采用 [MIT License](LICENSE)。`fewshot/gaussians.py` 的致密化逻辑沿用 3DGS（Inria，非商业研究许可），
所依赖的 StereoGS、FoundationStereo、Depth Anything 3 Giant 等第三方代码/权重各有其许可（多为**非商业**）。

---

## 日本語

### 概要

**3 / 6 / 9 枚の姿勢既知の写真** からシーンごとに 3D Gaussian Splatting を最適化します（LLFF は 1/8 解像度、DTU は 1/4 解像度・物体マスク評価）。
ガウシアンをネットワークで直接予測することはなく、凍結した事前学習モデルは深度推定にのみ使います。

### 手法の概要

主張：疎視点 3DGS の限界は、視点が重なる領域で幾何がどれだけ正確に固定されるかにあり、その精度は写真そのものだけが「証明（certify）」できる。

1. **深度の証明**：独立な推定が一致した画素の深度を「証明済み」とする。学習ベースのステレオ（FoundationStereo、実訓練画像の平行化ペア）2 組が 1% 以内で一致、
   またはステレオと古典的 PatchMatch MVS（COLMAP、姿勢固定）が平均から 1% 以内で一致する場合。残りの画素は証明済み画素でスケール補正した
   Depth Anything 3 で埋める（プレースホルダーで、制約としては使わない）。
2. **証明から 3 つの制約を導く**：そこに表面がある（初期化）／表面の手前の光線は空（free-space carving）／証明した 2 視点の間の仮想視点からも表面が見える
   （ペア局所の仮想視点教師）。加えて異方性を 10:1 に制限。
3. **適用の判断**はカメラ配置から自動で行う：訓練カメラの光軸の広がりが 45° を超える場合（物体中心の DTU）のみ遠方視点の制約を有効化し、前向きの LLFF では無効。
4. それ以外は標準的な疎視点向け正則化 3DGS（10k 反復、Gaussian dropout、適応的不透明度減衰）。全データセット・全視点数で同一設定（データセットごとの調整なし）。

詳細は [docs/METHOD.md](docs/METHOD.md)、[docs/ANALYSIS.md](docs/ANALYSIS.md) を参照してください。

### 性能（単一シード、PSNR↑ / SSIM↑ / LPIPS↓）

ベースラインは各論文の報告値、CertiSplat は StereoGS 公式評価器・同一分割で評価。表は [English → Results](#english) を参照。

- LLFF：3/6/9 視点の PSNR は **22.54 / 25.86 / 26.76**（論文報告の最良ベースライン比 +0.34 / +0.46 / +0.31 dB）。
- DTU：3/6/9 視点の PSNR は **23.29 / 26.20 / 28.25**（同 +1.29 / +0.39 / +0.03 dB）。ただし DTU の LPIPS は最良ベースラインに劣る。
- アブレーション（DTU 6 視点 PSNR）：完全版 26.20・異方性制限なし 26.05・carving なし 25.95・仮想視点可視性なし 25.75。
- 3 視点で 1 シーンあたり約 4〜5 分（RTX 6000 Ada）。Mip-NeRF360 と Blender は未実行。

### 環境構築

[English → Environment setup](#environment-setup) のコマンドを参照してください（Linux、Python 3.10、PyTorch 2.5.1 + CUDA 12.1）。要点：

1. conda 環境 `certisplat` を作成し、CUDA 版 COLMAP 3.8、PyTorch 2.5.1 (cu121) と依存パッケージをインストール；
2. StereoGS をリポジトリ直下に clone して指定コミットに切り替え、`patches/stereogs_local_paths.patch` を適用し、`diff-gaussian-rasterization` と `simple-knn` をビルド；
3. FoundationStereo ViT-S と edgenext_small の重みを `checkpoints/` に配置；
4. Depth Anything 3 は別環境で実行（`DA3_PYTHON`、`DA3_SOURCE`、`DA3_WEIGHTS` で指定）；
5. データは `datasets/` に配置（または `CERTISPLAT_DATA` を設定）。

### ライセンス

CertiSplat のオリジナルコードは [MIT License](LICENSE) です。`fewshot/gaussians.py` の densification は 3DGS（Inria、非商用研究ライセンス）に従い、
必要なサードパーティのコード・重み（StereoGS、FoundationStereo、Depth Anything 3 Giant など）はそれぞれのライセンス（多くは**非商用**）に従います。
