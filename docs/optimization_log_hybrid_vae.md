# MambaTalk_new_512 开发文档：混合 Codebook 策略实施记录

**日期**: 2026-03-10  
**项目**: MambaTalk_new_512  
**状态**: 实施与验证阶段

---

## 1. 项目背景与目标

本项目 `MambaTalk_new_512` 是基于 `MambaTalk_new` 的优化分支，旨在解决原始配置中存在的"动作幅度与细节精度"权衡问题。

### 1.1 核心问题
在对比实验中发现：
- **Codebook=256 (全配置)**: 手部细节还原度极高，但身体大幅度动作（如高抬臂）丢失，呈现"削波"现象。
- **Codebook=512 (全配置)**: 身体大幅度动作还原较好，但手部动作模糊，缺乏精细度（因 Codebook Collapse 导致）。

### 1.2 优化目标
实施 **混合 Codebook (Hybrid Codebook)** 策略，结合两者的优势：
- **Body VAE**: 采用 `Codebook=512`，以覆盖高维、大幅度的身体动作空间。
- **Hand VAE**: 保持 `Codebook=256`，以维持低维 PCA 空间中的手部精细度。

---

## 2. 架构与配置变更

### 2.1 VAE 模型配置
| 模块 | 原始配置 (New) | 优化配置 (New_512) | 配置文件路径 |
| :--- | :--- | :--- | :--- |
| **Body VAE** | `codebook_size: 256` | **`codebook_size: 512`** | `configs/mhr_cont_body.yaml` |
| **Hand VAE** | `codebook_size: 256` | `codebook_size: 256` | `configs/mhr_cont_hand.yaml` |

### 2.2 主模型架构调整
为了支持不同分支输出不同维度的 Latent Code，对主模型进行了以下改造：

**文件**: `models/mambatalk_mhr.py`
- **新增参数**: 引入 `body_codebook_size` (512) 和 `hand_codebook_size` (256)。
- **Decoder 层**: 
  - `self.motion_down_body`: 输出维度调整为 512。
  - `self.motion_down_hands`: 输出维度保持 256。
- **Classifier 层**: 
  - `self.body_classifier`: 映射维度调整为 512。
  - `self.hands_classifier`: 映射维度保持 256。

**文件**: `mambatalk_mhr_new_trainer.py`
- **常量拆分**: 废弃全局 `VAE_CODEBOOK`，拆分为 `BODY_CODEBOOK = 512` 和 `HAND_CODEBOOK = 256`。
- **损失计算**: 在计算 Classification Loss 和 Reconstruction Loss 时，分别使用对应的 Codebook 大小进行 Reshape 和 Projection。

---

## 3. 实施步骤记录

### 3.1 环境搭建
- 从 `MambaTalk_new` 克隆代码库，排除数据和输出目录。
- 建立软链接复用原始数据 (`data/`, `data_new/`) 和缓存 (`datasets/beat_cache/`)。
  - **注意**: 主模型缓存存储的是原始 381d 连续数据，不受 Codebook 大小影响，因此可直接复用。

### 3.2 权重迁移与训练
1.  **Hand VAE**: 直接复制 `MambaTalk_new` 中训练好的最佳权重 (`rec.bin`, epoch 1000, rec loss≈0.001) 到 `pretrained/pretrained_vq/mhr_cont_hand.bin`。
2.  **Body VAE**: 在 `tmux 87` (GPU 1) 上重新训练，配置为 `codebook=512`。
    - 状态: 已训练至 1000 epochs。
    - 最佳权重: `outputs/audio2pose/custom/0306_155503_mhr_cont_body/rec.bin`。

### 3.3 验证与渲染
使用 `render_vae_compare.py` 脚本进行了一系列对比验证：
- **脚本更新**: 支持通过参数 `--body_codebook_size 512` 动态加载不同配置的 VAE。
- **渲染结果**: 
  - 生成了 13 个样本的对比视频 (3 固定 + 10 随机)。
  - **观察结论**: 混合策略成功修复了抬臂动作（如样本 `32dkF...`），同时保留了手指的细腻动作，验证了方案的有效性。

---

## 4. 后续工作建议

1.  **主模型训练**: 
    - 使用新的混合 VAE 权重 (`pretrained_vq/` 下的 body 512 + hand 256) 启动主模型 `MambaTalk` 的训练。
    - 监控 Loss 曲线，确保模型能适应不同维度的 Latent 空间。
2.  **长序列评估**: 
    - 在更长的生成序列中评估动作的连贯性，观察是否存在因 Codebook 维度不一致导致的协调性问题（虽然理论上通过 Cross-Attention 机制已解决）。
3.  **文档维护**: 
    - 持续更新本开发文档，记录主模型训练的参数和最终评估结果。
