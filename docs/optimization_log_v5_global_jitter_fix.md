# MambaTalk 优化记录 V5 — Global 分支抖动修复

> 本文档接续 `optimization_log_v4.md`，记录 V5 版本针对 global 分支 clip 边界跳变问题的完整排查过程、多次尝试及最终解决方案。

---

## 一、问题描述

V4 版本解决了 body 的连续小抖动后，仍然存在**每 2 秒一次的全局旋转跳变**，表现为人物朝向突然偏移后瞬间回正。跳变频率与 clip 边界完全吻合（`pose_length=64`, `pre_frames=4`, `round_l=60` 帧 = 2 秒）。

另外还有**视野忽大忽小**的问题，渲染时相机远近不断变化。

## 二、根因分析

### 2.1 与 MambaTalk_paper 的逐步对比

MambaTalk_new_512 和 MambaTalk_paper 的推理流程（`_g_test`）具有相同的 clip 拼接结构：

1. 逐 clip 循环，模型预测各分支输出
2. clip 间通过 `pre_frames` overlap 做自回归反馈
3. 循环结束后拼接所有 clip 的 index/prediction
4. 对完整序列一次性调用 VQ-VAE decode

**关键差异在于 global 分支的处理方式：**

| 阶段 | MambaTalk_paper (trans+contact) | MambaTalk_new_512 (global_rot+contact) |
|------|-------------------------------|---------------------------------------|
| VQ 量化约束 | 在 lower 61d 里一起做 VQ（256 codebook entries） | 无（nn.Linear raw 输出） |
| 整段 Conv1d decode | vq_model_lower.decode(cat(all_indices)) | 无（直接拼接 raw prediction） |
| 二次 refine | global_motion VAEConvZero 整段处理 | 无 |
| 速度积分 | velocity2position | 无（绝对值回归） |

**body/hands/face** 在两个版本中都经过 VQ index → 整段 Conv1d decode，因此不抖动。

**global 是唯一没有任何平滑机制的分支，是唯一的抖动来源。**

### 2.2 Paper 的 4 层平滑链详解

Paper 的 trans(3)+contact(4) 被打包在 lower VQ-VAE 的 61d 中（`leg_6d(54) + trans(3) + contact(4)`），经过：

1. **VQ 量化**：将连续值离散化到 256 个 codebook entry，限制了 clip 边界处不连续的幅度
2. **Conv1d decoder 整段 decode**：`vq_model_lower.decode(cat(all_clip_indices))`，kernel=3 的 Conv1d 感受野跨越 clip 边界
3. **global_motion VAEConvZero 二次 refine**：decode 后的 rec_lower 再经过一个 Encoder+Decoder Conv1d 网络
4. **velocity2position 积分**：refined 速度积分为位置，天然连续

### 2.3 视野忽大忽小的原因

推理保存的 `pred_cam_t` 使用了 GT 的逐帧 cam_t 值（不是均值），Z 轴范围 [2.82, 3.60]，导致渲染时相机远近变化。

## 三、尝试过的方案及结果

### 3.1 方案一：给 global 加 VAEConvZero decoder（无 VQ）

**做法**：训练一个 Global AE（VAEConvZero，无量化器），7d → 240d latent → 7d。主模型预测 240d latent，推理时统一 decode。

**结果**：抖动更剧烈。

**原因**：240d 连续 latent 在 clip 边界不连续幅度大（无 VQ 约束），Conv1d decoder 的 kernel=3 把相邻的不一致 latent 混合，反而放大了失真。与 body/hands/face 的 VQ 机制本质不同——VQ 的 codebook 约束了不连续的幅度，而纯 AE 的连续 latent 没有这种约束。

### 3.2 方案二：增大 pre_frames 4→16

**做法**：增加 overlap 上下文帧数，让模型在 clip 边界看到更多前文。

**结果**：有一定改善但不充分，且需要重训。后续被其他方案覆盖。

### 3.3 方案三：速度积分（global_rot + cam_t 转速度预测）

**做法**：将 cam_t(3d) 纳入 global 分支（global 从 7d 变为 10d），global_rot 和 cam_t 都改为帧间速度预测，推理时 velocity2position 积分还原绝对值。

**结果**：效果更差。

**原因**：nn.Linear 逐帧预测速度在 clip 边界仍不连续，且改变了训练分布。速度积分保证位置连续但速度本身不连续，加速度 jitter 更明显。

### 3.4 方案四：将 global 打包进 body VQ-VAE（body 130d → 137d）

**做法**：仿照 Paper 将 trans+contact 打包进 lower VQ-VAE 的思路，将 global_rot(3)+contact(4) 打包进 body VQ-VAE（130d → 137d），让 VQ 量化 + Conv1d decode 覆盖 global。

**结果**：clip 边界跳变仍然严重。

**原因**：VQ 的 codebook 对 clip 边界两侧独立预测的 index 没有一致性约束。即使整段 decode 时 Conv1d 有感受野跨越边界，如果两侧预测的 index 本身差异大，Conv1d 的 3 帧窗口不足以弥合 index 级别的跳变。

### 3.5 方案五：savgol_filter 后处理（最终采用）

**做法**：在 `_g_test` 最终拼接 `rec_latent_global` 后，对 global 7d raw prediction 做 Savitzky-Golay 滤波（window_length=13, polyorder=2）。

**结果**：效果很好。clip 边界跳变降低 82%。

**关键位置**：`mambatalk_trainer.py` 第 406 行 `rec_global = rec_latent_global` 之后：

```python
from scipy.signal import savgol_filter
rec_global_np = rec_global.detach().cpu().numpy()
for b in range(rec_global_np.shape[0]):
    rec_global_np[b] = savgol_filter(rec_global_np[b], window_length=13, polyorder=2, axis=0)
rec_global = torch.from_numpy(rec_global_np).to(rec_latent_global.device)
```

**不同 window_length 的效果对比**：

| window_length | 边界帧间差均值 | 非边界帧间差 | 比率 | 边界最大差 |
|--------------|-------------|-----------|------|----------|
| 原始 | 0.2985 | 0.0058 | 51.7x | 0.8709 |
| 5 | 0.1505 | 0.0084 | 17.9x | 0.4178 |
| 9 | 0.0800 | 0.0090 | 8.9x | 0.2093 |
| **13** | **0.0532** | **0.0084** | **6.3x** | **0.1422** |
| 17 | 0.0388 | 0.0079 | 4.9x | 0.1068 |
| 21 | 0.0304 | 0.0074 | 4.1x | 0.0853 |

选择 wl=13 作为平衡点（边界跳变降低 82%，不会过度平滑）。

## 四、camfix 修复

### 问题

推理保存的 `pred_cam_t` 使用了 GT 逐帧 cam_t（每帧不同），Z 轴变化范围大，导致渲染视频中人物忽大忽小。

### 修复

将 `pred_cam_t` 替换为所有帧的均值（camfix），使相机位置在整个序列中保持恒定：

```python
cam_t_mean = cam_t_seq.mean(axis=0, keepdims=True).repeat(N, axis=0)
```

这在 `test()` 方法的保存逻辑中已有实现（第 545 行），但需确保所有推理路径都使用此逻辑。

## 五、最终方案总结

| 问题 | 原因 | 修复方案 | 效果 |
|------|------|---------|------|
| global_rot clip 边界跳变 | global 分支无 VQ/Conv1d 平滑 | savgol_filter(wl=13, poly=2) 后处理 | 边界跳变 -82% |
| 视野忽大忽小 | pred_cam_t 逐帧变化 | camfix（使用 cam_t 均值） | 视野稳定 |

### 代码修改清单

1. **`mambatalk_trainer.py` - `_g_test` 方法**：在 `rec_global = rec_latent_global` 后加 savgol_filter
2. **`mambatalk_trainer.py` - `__init__` 方法**：加 `self.eval_copy = None` fix（防止 AttributeError）
3. **`mambatalk_trainer.py` - `test()` 和 `test_demo()` 方法**：确保 `pred_cam_t` 使用均值

### 渲染输出

- savgol + camfix 对比视频：`outputs/render_savgol_camfix/compare_orig_vs_savgol_camfix.mp4`
- savgol + camfix 单独预测：`outputs/render_savgol_camfix/pred_hvNdco0wcPE_0017_speaker1_mesh.mp4`

## 六、经验总结

1. **VQ 量化是 clip 边界平滑的核心**：Paper 的 body/upper/lower/hands/face 都经过 VQ 量化 + Conv1d decode，VQ 的 codebook 约束了不连续幅度。单独加 Conv1d decoder（无 VQ）会放大不连续。

2. **不是所有 Paper 的机制都能移植**：Paper 的 4 层平滑链（VQ → Conv1d → global_motion → velocity2position）是紧密耦合的，单独移植其中一层（如速度积分、VAEConvZero）效果反而更差。

3. **低维数据不适合单独 VQ**：global 只有 7d，单独做 VQ 量化效果不好。Paper 的 lower 是 61d（leg 54d + trans 3d + contact 4d），高维打包才能有效 VQ。即使打包进 body（137d），VQ index 在 clip 边界的一致性仍不够。

4. **后处理平滑是实用的解决方案**：savgol_filter 直接在最终的 7d raw prediction 上做，不涉及高维 latent 空间，不会放大不连续性，且不需要重训。

5. **camfix 是必要的**：渲染视频时 `pred_cam_t` 必须使用均值，否则相机位置逐帧变化导致视野不稳定。
