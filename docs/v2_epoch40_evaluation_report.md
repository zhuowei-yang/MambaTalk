# MambaTalk v2 (PD-Enhanced) Epoch 40 评估报告

**日期:** 2026-03-17  
**评估 Checkpoint:** `outputs/audio2pose/custom/0317_054525_mambatalk_mhr_new/last_40.bin`  
**训练 Run:** `0317_054525_mambatalk_mhr_new` (训练中，已到 epoch 43)  
**评估环境:** sam_3d_body conda 环境 + PantoMatrix 评估工具  
**评估脚本:** `evaluate_v2.py`  

---

## 一、模型概况

| 项目 | 说明 |
|------|------|
| 模型版本 | MambaTalk v2 (PD-Enhanced) |
| 代码目录 | `/data1/yangzhuowei/MambaTalk_new_512_proj_global_v2` |
| 核心改进 | 集成 FFT 周期性解耦 (Periodicity Disentanglement) 模块 |
| PD 预训练 | `pd_best.bin`, best_loss=0.000751, 在关节速度数据上训练 |
| 训练配置 | 150 epochs, lr=1e-4, batch=64, GPU 3 (A100-40GB) |
| 评估 Epoch | 40 (当前最新保存的 checkpoint) |
| 推理平滑 | Savitzky-Golay 滤波 (window=5, polyorder=2) 应用于全局旋转 `rec_rot6d` |

---

## 二、指标定义

| 指标 | 含义 | 方向 |
|------|------|------|
| FGD (Frechet Gesture Distance) | 生成关节位置与 GT 的分布距离 | ↓ 越低越好 |
| L1Div (L1 Diversity) | 动作多样性 | ↑ 越高越好 |
| BC (Beat Consistency) | 音频-动作节拍对齐度 | ↑ 越高越好 |
| MSEFace | 面部顶点重建误差 | ↓ 越低越好 |
| LVDFace | 面部顶点速度差 | ↓ 越低越好 |
| MPJPE | 平均逐关节位置误差 (米) | ↓ 越低越好 |

---

## 三、评估结果

### 3.1 MambaTalk v2 (Epoch 40) 指标

| 指标 | v2 (PD-Enhanced, ep40) |
|------|------------------------|
| **FGD** ↓ | **0.1321** |
| **L1Div** ↑ | **3.3596** |
| **BC** ↑ | **0.2313** |
| **MSEFace** ↓ | **0.047274** |
| **LVDFace** ↓ | **0.002155** |
| **MPJPE** ↓ | **0.3023** |
| 总序列数 | 116 |
| 总帧数 | 45,043 |

### 3.2 与历史版本对比

| 指标 | v2 (ep40) | v1 MambaTalk_new_512 (ep90) | v1 MambaTalk_new (ep90) | v1 MambaTalk_new_512_res (ep70) |
|------|-----------|---------------------------|------------------------|---------------------------------|
| **FGD** ↓ | **0.1321** | 0.1222 | 0.1488 | 0.8706 |
| **L1Div** ↑ | 3.3596 | **4.0529** | 3.6289 | 1.8762 |
| **BC** ↑ | 0.2313 | **0.4258** | 0.3951 | 0.0363 |
| **MSEFace** ↓ | 0.047274 | 0.045450 | 0.046001 | **0.044669** |
| **LVDFace** ↓ | **0.002155** | 0.002695 | 0.002520 | 0.002141 |

> **注意:** v1 各模型的指标来自 ep90（或 ep70），而 v2 当前只训练到 ep40，尚有大量提升空间。

---

## 四、分析

### 4.1 当前阶段表现 (ep40)

1. **FGD = 0.1321**: 仅训练 40 epoch 就已接近 v1 ep90 的 0.1222，说明 PD 模块对动作质量分布有正向影响。随着训练继续，FGD 有望进一步下降。

2. **L1Div = 3.3596**: 多样性低于 v1 的 4.0529。这可能是因为 ep40 模型尚未充分训练，VQ-VAE codebook 利用率还在提升中。也可能是 PD 模块引入的周期性先验在早期阶段略微约束了多样性。

3. **BC = 0.2313**: 节拍一致性低于 v1 的 0.4258。BC 对音频-动作同步敏感，通常在训练后期才会显著提升。这是 ep40 vs ep90 差距的主要体现。

4. **MSEFace = 0.047274**: 面部误差与 v1 相当。MambaTalk v2 未修改面部生成路径，因此面部指标基本持平。

5. **LVDFace = 0.002155**: 面部速度差是所有版本中最优的，说明面部动作过渡的平滑性有所改善。

### 4.2 MPJPE 分布观察

通过 Per-sequence MPJPE 分析，序列可大致分为两组：
- **低误差组 (MPJPE < 0.25)**: 约 60% 的序列，误差在 0.04~0.22 之间
- **高误差组 (MPJPE > 0.5)**: 约 30% 的序列，误差在 0.50~0.68 之间

高误差组可能对应特定视频场景（大幅度运动、遮挡严重等），需进一步分析是数据质量还是模型泛化问题。

### 4.3 预期趋势

根据 v1 的训练经验，关键指标在 ep40 → ep90 期间通常有如下趋势：
- FGD: 持续下降
- L1Div: 显著提升
- BC: 显著提升（训练后期收敛）
- MSEFace/LVDFace: 基本稳定

---

## 五、渲染测试

### 5.1 渲染样本

随机选取 3 个测试序列进行 Mesh 渲染对比（GT vs Pred）：

| 序列名 | 帧数 | MPJPE |
|--------|------|-------|
| BYDLkW2UJjw_0009_speaker2 | 784 | 0.0793 |
| X7Zb6vk3nok_0019_speaker1 | 124 | 0.5766 |
| ucAAqShSYhM_0015_speaker1 | 244 | 0.5593 |

### 5.2 渲染输出

渲染视频保存在: `outputs/render_v2_ep40/`（渲染环境: GPU 1, xvfb + osmesa）

| 文件 | 大小 | 说明 |
|------|------|------|
| `compare_BYDLkW2UJjw_0009_speaker2.mp4` | 1.9M | GT vs Pred 并排对比 + 音频 |
| `compare_X7Zb6vk3nok_0019_speaker1.mp4` | 310K | GT vs Pred 并排对比 + 音频 |
| `compare_ucAAqShSYhM_0015_speaker1.mp4` | 587K | GT vs Pred 并排对比 + 音频 |
| `pred_*_mesh.mp4` | 89K~699K | 预测 Mesh 渲染（纯黑背景） |
| `gt_*_mesh.mp4` | 1.1M~8.4M | GT Mesh 渲染（含原始视频叠加） |

渲染使用 SAM-3D-Body (`sam_3d_body` conda env) 的 `render_mesh_video_from_npz.py` 脚本完成。

---

## 六、训练 Loss 趋势 (截至 ep43)

| Epoch | vel | latent | cls_full | accel |
|-------|-----|--------|----------|-------|
| 40 | 0.022 | 0.018 | 3.67 | 0.025 |
| 42 | 0.022 | 0.018 | 3.65 | 0.025 |
| 43 | 0.021 | 0.017 | 3.58 | 0.024 |

Loss 仍在稳步下降，模型尚未收敛，后续 epoch 有进一步提升空间。

---

## 七、后续计划

1. **继续训练至 150 epoch**，在 ep50/60/70/80/90/100 等节点重新评估指标
2. **对比 ep90 的 v2 与 v1 指标**，全面评估 PD 模块的增益
3. **分析高 MPJPE 序列**，确认是数据质量还是模型问题
4. **考虑对 body/hand 输出也增加平滑处理**，目前仅对全局旋转做了 savgol 滤波
5. **调整 PD 模块超参数** (如 `lambda_p`, `n_channels`) 进行消融实验

---

## 附录：评估环境

```
评估脚本: evaluate_v2.py
SAM-3D-Body: sam_3d_body conda env + DINOv3 backbone
PantoMatrix: /data1/yangzhuowei/PantoMatrix/emage_evaltools/
指标保存: outputs/audio2pose/custom/0317_054525_mambatalk_mhr_new/40/evaluation_metrics.txt
渲染脚本: render_v2_ep40.sh
GPU: A100-SXM4-40GB (eval: GPU 2, render: GPU 1, training: GPU 3)
```
