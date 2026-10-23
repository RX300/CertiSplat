# 非前馈 3DGS Few Shot 文献与 Baseline 调研

调研日期：2026-10-14。目标是选择能支持后续论文实验的强 baseline，优先 2026 年顶会的逐场景优化方法。

**建议先复现 TWINGS、StableGS、StereoGS；若重点研究不依赖外部学习先验的优化算法，则以 SE-GS 和 StableGS 为主。** DropAnSH-GS、FlatMinGS 值得重点阅读，但当前官方仓库尚未发布训练代码。下面的“SOTA 候选”指各论文在其评测协议下报告领先或有竞争力的结果，不表示存在一个跨数据集、初始化和先验设置的统一冠军。

## 范围与时间口径

- 主任务：静态场景的 sparse-view / few-shot novel view synthesis，保留逐场景优化 Gaussian 参数的工作。
- 主时间窗口：2025-10-14 至 2026-10-14，按正式会议发表计。预印本首次公开时间单独考虑：例如 D²GS 首次提交于 2025-10-09，SE-GS 于 2024-10-31，均早于严格的首次公开时间窗口，但分别在 ICLR 2026、ICCV 2025 正式发表。
- A 类：训练改进主要来自 Gaussian 优化、正则和多视图约束，不额外引入深度或生成大模型。
- B 类：仍逐场景优化，但使用预训练深度、匹配或立体模型等先验。
- C 类：利用 diffusion 等模型生成或修复监督，再逐场景优化；单独比较。
- 前馈预测 Gaussian、以及“前馈 Gaussian 初始化后短暂微调”的混合路线，不作为本项目主 baseline。表面重建、逆渲染、动态人体也与标准 NVS 分开。

## 2026 年顶会优先清单

“有训练代码”仅表示公开仓库中已核实训练入口及相关目录；后续部署和实际运行状态见 DEPLOYMENT.md。“未找到”表示本次检索未定位到可用官方实现，并非断言其不存在。

| 方法 | 会议与依据 | 核心研究点 | 先验与任务 | 代码状态及优先级 |
|---|---|---|---|---|
| **TWINGS** | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_TWINGS_Thin_Plate_Splines_Warp-aligned_Initialization_for_Sparse-View_Gaussian_Splatting_CVPR_2026_paper.pdf) | 用 TPS 对齐深度反投影点与三角化控制点，改善初始化 | B；深度及匹配先验；LLFF、DTU、Mip-NeRF360 | [官方代码](https://github.com/sandokim/TWINGS-official)，有初始化、训练、渲染、评测；**第一批** |
| **D²GS** | [ICLR 2026 正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ca2963d1cfb25e93362e86fb427a9524-Abstract-Conference.html) | 深度与密度引导 dropout，增强远处欠拟合区域监督 | B；单目深度用于远场掩码；LLFF、Mip-NeRF360 | [官方 DDGS 仓库](https://github.com/Willacold/DDGS)，有训练和评测，但缺少官方 DAFE mask 生成流程；**已按用户要求退出部署** |
| **StereoGS** | [ECCV 2026，arXiv 录用说明](https://arxiv.org/abs/2606.30545) | 虚拟双目深度约束、梯度感知 opacity 衰减、稠密初始化 | B；FoundationStereo 与 MVS 先验；LLFF、DTU、Mip-NeRF360、Blender | [官方代码](https://github.com/StringerYwh00/StereoGS)，有训练及数据集脚本；**第一批强先验对照** |
| **StableGS** | [ECCV 2026 Springer 正式论文](https://link.springer.com/chapter/10.1007/978-3-032-36839-3_5) | 属性空间与视点空间扰动，缓解优化曲面的方向性不稳定 | A；不额外使用深度或语义监督；LLFF、DTU、Mip-NeRF360 | [官方代码](https://github.com/zyl123456aB/StableGS)，有官方训练、预处理、渲染和评测源码；已按作者说明验证 `train3.py` 与第二阶段 `iteration_5000`；**第一批优化对照** |
| **DropAnSH-GS** | [CVPR 2026 正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_Dropping_Anchor_and_Spherical_Harmonics_for_Sparse-view_Gaussian_Splatting_CVPR_2026_paper.html) | 成组丢弃 anchor 及邻居，并对高阶 SH 做 dropout | A；结构与外观正则；LLFF、Mip-NeRF360、Blender | [官方仓库](https://github.com/Fangkang515/DropAnSH-GS)只有 README，发布训练代码的任务未完成；**必读，等待代码** |
| **FlatMinGS** | [ECCV 2026 Springer 正式论文](https://link.springer.com/chapter/10.1007/978-3-032-37038-9_2) | 尺度自适应位置扰动与周期参数重初始化 | A；不引入额外网络；主实验 LLFF、Mip-NeRF360，附录另含 DTU、Blender、Tanks & Temples、NeO360 | [官方仓库](https://github.com/kangrnin/FlatMinGS)只有 README，安装和训练均 Coming soon；**必读，等待代码** |
| **ICO-GS** | [CVPR 2026 正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiong_Intrinsic_Geometry-Appearance_Consistency_Optimization_for_Sparse-View_Gaussian_Splatting_CVPR_2026_paper.pdf) | 特征域多视图几何约束，循环一致性过滤深度，再监督外观 | 逐场景几何与外观联合优化；LLFF、DTU、Blender；具体特征依赖需随实现核对 | [论文项目地址](https://KaiqiangXiong.github.io/ICO-GS/)访问为 404，未找到训练仓库；**阅读与追踪** |
| **CAGS** | [CVPR 2026 正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Confidence-Guided_Multi-Scale_Aggregation_for_Sparse-View_High-Resolution_3D_Gaussian_Splatting_CVPR_2026_paper.pdf) | 多尺度 Gaussian 场聚合、置信度 opacity 加权、伪视角一致性 | 逐场景优化；突出高分辨率 few-shot NVS；完整依赖待实现核对 | 未定位到官方训练仓库；**高分辨率方向重点阅读** |
| **HeroGS** | [CVPR 2026 正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_HeroGS_Hierarchical_Guidance_for_Robust_3D_Gaussian_Splatting_under_Sparse_CVPR_2026_paper.pdf) | 图像伪监督、特征自适应密度控制、辅助 Gaussian 场协同剪枝 | B；插帧、深度、边缘先验；LLFF、Tanks & Temples 等 | 未定位到官方训练仓库；**适合 2–3 视角及伪视图方向** |

### 如何选择第一篇实现基础

**以工程落地和对比覆盖为先：TWINGS。** 初始化模块可独立提供点云，官方仓库给出了接入其他 GS 方法的方式，便于研究“同一优化算法、不同初始化”的效果。其完整方法还包含深度损失和 DropGaussian 风格正则，实验时应区分 TWINGS-Init 与完整 TWINGS。依赖包含 DepthPro，不能归为无外部先验方法。[官方实现](https://github.com/sandokim/TWINGS-official)

**以优化机制和可解释性为先：StableGS。** D²GS 将近处过拟合、远处欠拟合分开处理，已有训练入口；StableGS 直接研究 Gaussian 参数与视点扰动，更贴近无外部学习先验的优化研究。StableGS README 的标题、匿名引用和 under-review 字样尚未统一更新，引用时使用 Springer 的正式题目与作者；作者已在 issue #1 澄清 LLFF 应使用 train3.py 与第二阶段 iteration_5000；当前部署遵循这一说明。D²GS 因 mask 预处理缺失已退出部署。[D²GS 原文](https://arxiv.org/html/2510.08566)、[StableGS 正式记录](https://link.springer.com/chapter/10.1007/978-3-032-36839-3_5)

**以较强几何先验的结果为对照：StereoGS。** 它在优化期间调用立体模型，最终依然得到逐场景优化的 Gaussian 场。训练成本要包括 MVS 初始化及在线立体估计。官方 README 的不同数据集配置使用不同 MVS 模型，不能将所有结果都视为完全相同的初始化设置。[官方实现与配置](https://github.com/StringerYwh00/StereoGS)

## 论文结果提供的证据

以下数字来自作者报告，不是本地复现结果。每一行的比较限定在对应论文表格内，不应将各行直接排序。PSNR 单位为 dB。

| 方法与原文位置 | 设置 | 作者报告 | 同文中的参照及含义 |
|---|---|---|---|
| [D²GS，表 1](https://arxiv.org/html/2510.08566) | LLFF，3 views，1/8 分辨率 | 21.35 / 0.746 / 0.179，依次为 PSNR / SSIM / LPIPS | DropGaussian 为 20.76 / 0.713 / 0.200，PSNR 提高 0.59 |
| [TWINGS，附录 G 汇总表及主表](https://arxiv.org/abs/2605.22069) | DTU，3 views | PSNR 21.52，SSIM 0.880 | 同一汇总表中 Binocular3DGS 为 20.71 / 0.862；对初始化研究很有参考价值 |
| [StereoGS，项目结果表](https://stringerywh00.github.io/StereoGS_project_page/) | LLFF，3 views，1/8 分辨率 | 基础版 21.91；固定 dropout 0.3 的 Ours* 为 22.05 | 同表 D²GS 为 21.35；两种 StereoGS 变体应分别命名 |
| [StableGS，仓库结果表](https://github.com/zyl123456aB/StableGS) | DTU，3 views，object masks | PSNR 20.86，SSIM 0.848，LPIPS 0.130 | 同表 DropoutGS 为 20.22 / 0.830 / 0.150；正式摘要也报告 DTU +0.64 dB |
| [DropAnSH-GS，LLFF 主表](https://arxiv.org/html/2602.20933) | LLFF，3 views | 20.68 / 0.724 / 0.194 | 本文的 DropGaussian 为 20.33 / 0.709 / 0.201，和 D²GS 表中的 20.76 不同，说明不能跨表直接排名 |
| [FlatMinGS，表 1–2](https://arxiv.org/html/2607.00885v1) | LLFF 3 views；Mip-NeRF360 12 views | LLFF PSNR 20.88；Mip-NeRF360 PSNR 19.50、LPIPS 0.348 | 后者 DropGaussian 为 PSNR 19.74、LPIPS 0.364；感知质量改善不等于所有指标领先 |
| [CAGS，表 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Confidence-Guided_Multi-Scale_Aggregation_for_Sparse-View_High-Resolution_3D_Gaussian_Splatting_CVPR_2026_paper.pdf) | LLFF，3 views，原始分辨率 | PSNR 18.85 | 同表 NexusGS 为 16.12；不能拿它与 1/8 分辨率的 21–22 dB 直接比较 |
| [HeroGS，表 1](https://arxiv.org/html/2603.01099) | LLFF，3 views | 21.30 / 0.739 / 0.189 | 同表 DropGaussian 为 20.55 / 0.710 / 0.200；论文 2-view 结果另有失败场景排除，必须保留该条件 |

## 近一年内的 2025 顶会补充

| 工作 | 会议与来源 | 纳入理由 | 实现与定位 |
|---|---|---|---|
| **SE-GS**，Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis | [ICCV 2025 原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Self-Ensembling_Gaussian_Splatting_for_Few-Shot_Novel_View_Synthesis_ICCV_2025_paper.pdf)；作者标注 Oral | 不确定性感知扰动、自集成、一致性约束；非常适合优化路线 | [官方代码](https://github.com/sailor-z/SE-GS)，LLFF、DTU、Mip-NeRF360 训练与评测可见。首次预印本在 2024 年，按会议口径纳入 |
| **Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting** | [NeurIPS 2025 正式论文集](https://papers.neurips.cc/paper_files/paper/2025/hash/a831bd2b1bf8935641a5c890a158c519-Abstract-Conference.html) | 用 co-adaptation 指标分析高斯间过度依赖，并以 dropout / opacity 噪声缓解 | 方法分析与消融依据；本次未定位到官方训练仓库。预印本首次公开于 2025-08 |
| **RI3D** | [ICCV 2025 正式论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Paliwal_RI3D_Few-Shot_Gaussian_Splatting_With_Repair_and_Inpainting_Diffusion_Priors_ICCV_2025_paper.html) | repair 和 inpainting diffusion 分别处理已观测区域与未观测区域，随后优化 3DGS | [官方代码](https://github.com/avinashpaliwal/RI3D)；C 类，仅在允许生成先验时列入主比较 |

历史锚点仍建议保留：原始 3DGS、FSGS、DNGaussian、CoR-GS、DropGaussian、DropoutGS。它们主要发表于 2023–2025 年上半年，**不计入近一年新作**。其中 DropGaussian 与 DropoutGS 是不同论文，实验表和实现目录应保持区分。[DropGaussian 官方仓库](https://github.com/DCVL-3D/DropGaussian_release)、[DropoutGS 作者项目页](https://xuyx55.github.io/DropoutGS/)

## 相邻任务与其他近期工作

- **DP-GS，ECCV 2026**：重点是稀疏视角表面重建，使用法线引导深度传播；有训练与 mesh 提取代码。若研究目标包含几何精度和 Chamfer Distance，应另开 surface reconstruction 比较表。[论文项目页](https://hanl2010.github.io/DP-GS/)、[代码](https://github.com/hanl2010/DP-GS)
- **MS-GS，NeurIPS 2025**：研究稀疏视角与多外观条件共同存在时的重建。可作为 in-the-wild 扩展，不能直接替代固定外观的标准 NVS 对照。[正式原文](https://papers.nips.cc/paper_files/paper/2025/file/58d451ce195db5729b0386a35c74405a-Paper-Conference.pdf)
- **FSA-GS，CVIU 2026 年 9 月**：以细结构重采样、分裂及 opacity / density dropout 改善稀疏 NVS；属于近期期刊补充，优先级低于上面的顶会复现集合。[出版社页面](https://www.sciencedirect.com/science/article/pii/S1077314226002596)
- **ReGAGS，Computers & Graphics，2026-09-14 在线**：针对几何与外观过拟合，采用贡献度 dropout 和几何感知颜色一致性；出版社摘要说明不引入额外外部先验。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0097849326002335)

本轮没有核实到应加入主清单的 ICML 2026 或 NeurIPS 2026 同类正式论文；不把投稿中的 OpenReview 页面当作录用证明。

## 排除与边界

- **SparseSplat、EcoSplat、FreeSplatter、SPFSplat、GenWildSplat** 等以跨场景网络直接预测 Gaussian 为主，排除出非前馈主表。例如 [SparseSplat 的 CVPR 2026 正式摘要](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_SparseSplat_Towards_Applicable_Feed-Forward_3D_Gaussian_Splatting_with_Pixel-Unaligned_Prediction_CVPR_2026_paper.html)和 [FreeSplatter 的 ICCV 2025 正式摘要](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_FreeSplatter_Pose-free_Gaussian_Splatting_for_Sparse-view_3D_Reconstruction_ICCV_2025_paper.html)明确采用 feed-forward 框架。
- **FSFSplatter** 虽然包含逐场景优化，但核心有 Transformer 与 Gaussian head 直接预测场景初始化，按本项目当前范围不放入主 baseline。[作者原文](https://arxiv.org/abs/2510.02691)
- **GAINS、SGS-Intrinsic** 属于 sparse-view inverse rendering；**SV-GS** 属于动态 4D 重建。任务和指标不同，留作扩展方向，不与静态 NVS 混排。[GAINS 项目页](https://patrickbail.github.io/gains/)、[SGS-Intrinsic 官方仓库](https://github.com/GrumpySloths/SGS_Intrinsic.github.io)、[SV-GS 正式原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Chao_SV-GS_Sparse_View_4D_Reconstruction_with_Skeleton-Driven_Gaussian_Splatting_CVPR_2026_paper.pdf)

## 建议的第一轮实验

### Baseline 组合

1. **基础与历史对照**：3DGS、CoR-GS、DropGaussian。
2. **无额外学习先验的主要比较**：SE-GS、StableGS。DropAnSH-GS 和 FlatMinGS 等官方代码发布后再加入；若自行实现，应明确标注 reimplementation。
3. **允许几何先验的主要比较**：TWINGS、StereoGS。D²GS 仅保留文献记录，因官方 mask 预处理缺失退出部署。
4. **可选生成先验组**：RI3D。若研究不做 diffusion，可先不运行。

若只能先跑三个新 baseline，选 **TWINGS + StableGS + StereoGS**，覆盖初始化、优化正则、立体几何先验；若要求整个流程也不使用外部学习模型，则选 **SE-GS + StableGS + DropGaussian**，其中 DropGaussian 为历史锚点。

### 各论文的数据集和输入视角

下表的数字是每个场景用于优化的输入图像数，不是测试图像数。只列已核实的主要设置；附录扩展单独标注。均在留出的测试视角渲染后计算新视角图像指标。

| 方法 | 数据集和输入视角数 | 主要图像指标与补充 |
|---|---|---|
| [TWINGS](https://arxiv.org/abs/2605.22069) | LLFF 3/6/9；DTU 3/6/9；Mip-NeRF360 12/24 | PSNR、SSIM、LPIPS，另有 AVGE |
| [D²GS](https://arxiv.org/html/2510.08566) | 主质量比较：LLFF 3、Mip-NeRF360 12；LLFF 6 另用于稳定性等实验 | PSNR、SSIM、LPIPS、AVGE；额外报告 IMR 稳定性指标 |
| [StereoGS](https://github.com/StringerYwh00/StereoGS) | LLFF 3/6/9；DTU 3/6/9；Mip-NeRF360 12/24；Blender 8 | PSNR、SSIM、LPIPS |
| [StableGS](https://github.com/zyl123456aB/StableGS) | 仓库主结果：LLFF 3/6/9；DTU 3；Mip-NeRF360 12/24 | PSNR、SSIM、LPIPS；另分析优化曲面各向异性 |
| [DropAnSH-GS](https://arxiv.org/html/2602.20933) | LLFF 3/6/9；Mip-NeRF360 12；Blender 8 | PSNR、SSIM、LPIPS，另分析模型大小和效率 |
| [FlatMinGS](https://arxiv.org/html/2607.00885v1) | 主实验 LLFF 3/6/9、Mip-NeRF360 12/24；附录含 DTU 3、Blender 8、Tanks & Temples 12、NeO360 12 | PSNR、SSIM、LPIPS；附录另有较多输入视角的实验 |
| [ICO-GS](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiong_Intrinsic_Geometry-Appearance_Consistency_Optimization_for_Sparse-View_Gaussian_Splatting_CVPR_2026_paper.pdf) | LLFF 3/6/9；DTU 3/6/9；Blender 8 | PSNR、SSIM、LPIPS |
| [CAGS](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Confidence-Guided_Multi-Scale_Aggregation_for_Sparse-View_High-Resolution_3D_Gaussian_Splatting_CVPR_2026_paper.pdf) | LLFF 3；DTU 3；Mip-NeRF360 12/24；Blender 8 | PSNR、SSIM、LPIPS；高分辨率为主要评测重点 |
| [HeroGS](https://arxiv.org/html/2603.01099) | 主实验 LLFF 2/3/6、Tanks & Temples 3/6；补充实验 Mip-NeRF360 12/24 | PSNR、SSIM、LPIPS；LLFF 2-view 仅评测 SfM 成功的 5 个场景 |
| [SE-GS](https://github.com/sailor-z/SE-GS) | LLFF、DTU、Mip-NeRF360、MVImgNet | 图像质量评测；各数据集视角索引以作者配置为准 |

常规主实验组合建议为 LLFF 3/6/9、DTU 3/6/9、Mip-NeRF360 12/24，Blender 8 作为合成场景补充。指标以 PSNR↑、SSIM↑、LPIPS↓ 为主。DTU 上的 NVS 图像指标与表面重建的 Chamfer Distance 是不同任务，不能混用。

分辨率需单独记录：例如 TWINGS 原文对 Mip-NeRF360 写 1/4 分辨率，而 StereoGS 与 FlatMinGS 使用 1/8。除分辨率外，还需核对测试视角索引、场景子集、DTU 前景 mask、LPIPS 网络和初始化来源，才能形成可直接比较的实验表。

### 数据集与统一记录

第一轮可采用 LLFF 3 / 6 / 9 views、DTU 3 / 6 / 9 views、Mip-NeRF360 12 / 24 views；算力有限时先做 LLFF 3 views 的开发实验，再运行完整场景评测。下面是为本项目提出的比较协议，不能自动视作每篇论文的原始设置。

| 项目 | 需要固定或单独报告的内容 |
|---|---|
| 视角选择 | 明确训练、验证、测试图像索引；相同 views 数不保证图像相同 |
| 位姿 | 已知位姿与 pose-free 方法分组；记录位姿来源以及是否优化位姿 |
| 初始化 | 区分稀疏 SfM、COLMAP dense stereo、学习式稠密点云；只使用允许的训练图像构建点云，避免测试 RGB 泄漏 |
| 图像与掩码 | 固定最终渲染及评测分辨率；DTU 明确 object-mask 指标与背景处理；高分辨率实验单列 |
| 外部先验 | 记录深度、MVS、匹配、立体、插帧或 diffusion 模型及权重；同时给出无先验消融 |
| 失败场景 | 记录 SfM 失败、空点云等情况及处理，不静默排除难例；SE-GS 已说明部分 DTU 场景使用随机初始化 |
| 计算成本 | 总时间包含预处理、点云初始化、先验推理和 3DGS 优化；另报渲染 FPS、峰值显存和 Gaussian 数量 |
| 随机性 | 固定并记录 seed；正式主结果建议至少 3 次运行，报告均值与标准差 |
| 实现版本 | 复现时记录 Git commit、配置、CUDA / PyTorch 和 rasterizer 版本；本轮没有生成实验结果 |

建议保留两种实验：先按各作者原配置验证可复现性，再用统一输入、初始化和计算预算衡量算法本身。修改初始化或先验后的版本应另命名，避免误用原论文方法名与数值。
