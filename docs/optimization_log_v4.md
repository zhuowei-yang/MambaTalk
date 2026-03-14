# MambaTalk 优化记录 V4 — 身体抖动修复 + DualMambaTalk 双人对话

> 本文档接续 `optimization_log.md`（V1-V3），记录 V4 版本的新增优化和 DualMambaTalk 双人对话功能开发。

---

## 一、V4 身体抖动修复（单人 MambaTalk）

### 1.1 问题描述

V3 版本训练的模型，手部动作已有改善，但渲染出的 res 视频仍存在两类身体抖动：

| 症状 | 频率 | 严重程度 |
|------|------|---------|
| 连续小抖动 | 每帧 | 中等，身体微颤 |
| 偶发大跳动 | 约 5-6 秒一次 | 严重，整体位移 |

GT 视频（原始数据渲染）完全无此问题，确认是模型预测端的问题。

### 1.2 根因分析

**连续小抖动 — VQ-VAE 量化跳变**：
- Body 分支推理路径：模型输出 256d logits → argmax 取离散 codebook index → VQ-VAE decoder → 130d body pose
- 训练时的 velocity loss (`vel_body_weight=0.5`) 作用在 latent 空间，但 argmax 是离散操作
- 相邻帧的 logits 可能很接近但 argmax 选中不同 codebook 条目，导致解码后的 body pose 跳变
- `vel_body_weight=0.5` 是三个分支中最低的（global=2.0, hands=1.0）

**偶发大跳动 — 自回归片段边界**：
- `_g_test()` 使用 `pose_length=64` 帧/片段，`pre_frames=4` 帧重叠
- 每片段推进 60 帧（2 秒 @30fps），每 2 秒有一个片段边界
- 训练-推理分布偏移：训练时 `in_motion` 始终是 GT，推理时初始帧来自有噪声的 VQ 解码输出
- 误差逐片段积累，在第 3 个边界（~6s）可能产生明显跳动

### 1.3 实施的优化方案

#### 方案 A：加速度损失（二阶平滑约束）

在 `_g_training()` 中添加对 body/global/hands 的二阶差分 L1 loss，直接惩罚速度的突变：

```python
# 离散二阶导：acc = x[t+1] - 2*x[t] + x[t-1]
rec_acc_body = rec_body_lat[:, 2:] - 2 * rec_body_lat[:, 1:-1] + rec_body_lat[:, :-2]
tar_acc_body = tar_body_lat[:, 2:] - 2 * tar_body_lat[:, 1:-1] + tar_body_lat[:, :-2]
loss_acc_body = self.vel_loss(rec_acc_body, tar_acc_body)
```

新增配置参数：
- `acc_body_weight: 1.0`
- `acc_global_weight: 1.0`
- `acc_hands_weight: 0.5`

#### 方案 B：Body NLLLoss 标签平滑

将 body 分支的 `NLLLoss` 替换为 `CrossEntropyLoss(label_smoothing=0.1)`，减少相邻帧在两个接近 codebook 条目间的振荡：

```python
self.cls_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
# 应用到所有三个训练分支（full-mask、self-mask、word-mask）的 body cls loss
loss_cls_body = self.cls_loss_smooth(
    net_out["cls_body"].reshape(-1, vae_codebook_size), tar_index_body)
```

#### 方案 C：姿态空间 body 损失（尝试后移除）

尝试通过可微软解码路径（`softmax(logits) @ codebook → VQ decoder`）在实际 130D body pose 空间计算 loss。

**问题**：通过冻结 VQ-VAE decoder 的 conv 层回传梯度，在 epoch 32 触发梯度爆炸（loss 突变为 NaN）。

**第二次尝试**：改为 detached decoder + soft-latent 空间 loss。在 epoch 55 再次出现 NaN。

**最终决定**：移除此方案。加速度损失 + 标签平滑已足够解决抖动问题。

### 1.4 NaN 问题排查

| 尝试 | NaN 出现时间 | 原因 |
|------|-------------|------|
| 方案 C v1（full decoder gradient） | epoch 32, batch 300 | 梯度通过冻结 conv 层放大 → 爆炸 |
| 方案 C v2（detached decoder + soft-latent） | epoch 55, batch 590 | soft-latent path 与 latent loss 冲突 |
| 方案 C 移除 + lr 降至 2e-4 | 无 NaN | 稳定训练 |

### 1.5 最终配置（V4 单人）

```yaml
lr_base: 2e-4           # 从 5e-4 降低，提升稳定性
batch_size: 64           # 从 8 提升，GPU 6 可用 40 GiB
epochs: 150              # 从 200 减少
test_start_epoch: 40     # 从 70 提前

# 新增 loss 权重
acc_body_weight: 1.0     # 身体加速度约束
acc_global_weight: 1.0   # 全局旋转加速度约束
acc_hands_weight: 0.5    # 手部加速度约束

# body 分类改用 CrossEntropyLoss(label_smoothing=0.1)
```

### 1.6 训练效果

训练到 epoch 65 时的 loss 趋势（无 NaN，稳定收敛）：

| 指标 | Epoch 0 | Epoch 30 | Epoch 65 | 说明 |
|------|---------|---------|---------|------|
| vel | 0.833 | 0.084 | 0.061 | 一阶速度损失 |
| accel | 1.006 | 0.087 | 0.064 | 二阶加速度损失（新增） |
| latent | 7.882 | 0.317 | 0.184 | latent 重建 |
| cls_full | 16.736 | 6.425 | 4.043 | 分类损失（含标签平滑） |

渲染验证：res 视频身体抖动明显减少，手部动作流畅。

### 1.7 修改文件清单

| 文件 | 改动内容 |
|------|---------|
| `mambatalk_trainer.py` | 添加 accel loss（193-205行）；body NLLLoss → CrossEntropyLoss(label_smoothing=0.1)（73行初始化，214-216/238-240/259-261行应用） |
| `configs/mambatalk.yaml` | 新增 `acc_*_weight` 参数；lr 降至 2e-4；batch_size=64；epochs=150；test_start_epoch=40 |
| `utils/config.py` | 注册 `acc_body_weight`, `acc_global_weight`, `acc_hands_weight`, `pose_body_weight`, `pose_vel_body_weight` 参数 |

---

## 二、DualMambaTalk 双人对话功能

### 2.1 功能定义

用 MambaTalk 的架构（WavEncoder + GlobalScan + LocalScan + VQ-VAE）实现双人对话动作生成：

- **输入**: speaker1.npz (动作) + speaker1.wav (音频) + speaker2.wav (音频)
- **输出**: speaker2.npz (动作)
- **核心思想**: 给定一方的动作和双方音频，生成另一方的响应动作

### 2.2 架构设计

纯用 MambaTalk 组件，不引入任何新架构（无 CVAE、无 DualTalk 的组件）：

**与单人 MambaTalk 的对比**：
- 单人：`in_motion` = 自身 GT 动作，`in_audio` = 自己的音频
- 双人：`in_motion` = speaker2 的 GT 动作（生成目标），`in_audio` = speaker2 的音频（主信号），新增 speaker1 的音频 + 动作作为跨说话人条件

**新增组件（全部复用 MambaTalk 已有模块）**：
- `cond_audio_encoder`: `WavEncoder(audio_f, audio_in=2)` — 编码 speaker1 音频
- `cond_motion_encoder`: `VQEncoderV6` — 编码 speaker1 动作序列
- `cond_proj`: `nn.Linear(motion_f + audio_f, motion_f)` — 融合条件特征

**条件融合方式**：将 speaker1 的条件特征加到 `global_features` 上（GlobalScan 和 LocalScan 之间）：
```python
global_motions, global_features = self.global_scan(in_motion, mask, ...)
cond_feat = self.cond_proj(cat([cond_motion_feat, cond_audio_feat], dim=-1))
global_features = global_features + cond_feat  # 条件融合
output = self.local_scan(global_motions, global_features, ...)
```

### 2.3 数据加载器设计

新建 `dataloaders/beat_dual.py`，从 `beat_sep_lower.py` 派生：

**配对逻辑**：
1. 从 `train_test_split.csv` 筛选 `speaker1` 行（避免重复配对）
2. 对每个 `xxx_speaker1`，找到对应 `xxx_speaker2` 的 npz/wav/textgrid
3. 两方同时加载 `load_mhr_native()`，对齐帧数（取 min）
4. 同步切段：两方使用相同的 `start_idx:fin_idx` 窗口

**LMDB 缓存扩展为 13 字段**：
```python
v = [pose1, audio1, word1, vid1, trans1, shape1,
     pose2, audio2, word2, vid2, trans2, shape2, facial2]
```

**`__getitem__` 返回**：
```python
{"pose1", "audio1", "word1", "id1", "trans1", "beta1",
 "pose2", "audio2", "word2", "id2", "trans2", "beta2"}
```

### 2.4 训练器设计

新建 `dualtalk_trainer.py`，从 `mambatalk_trainer.py` 派生：

**`_load_data` 改动**：
- speaker1 → 条件输入（`cond_pose`, `cond_audio`, `cond_word`），不需要 VQ 编码
- speaker2 → 生成目标（`tar_pose`, `in_audio`, `in_word`），需要 VQ 编码计算 loss

**`_g_training` 改动**：
- 所有模型调用增加 `cond_audio=..., cond_motion=...` 参数
- Loss 体系与 MambaTalk 完全一致（latent MSE + velocity + acceleration + NLLLoss）

**`_g_test` 改动**：
- 自回归推理时，`cond_audio` 和 `cond_motion` 按 chunk 同步切片

**`test` / `test_demo` 改动**：
- 输出文件名使用 speaker2 的 ID
- 渲染常量从 speaker2 的原始 npz 读取

### 2.5 新建文件清单

| 文件 | 说明 |
|------|------|
| `dataloaders/beat_dual.py` | 双人配对数据加载器 |
| `models/dualmambatalk.py` | DualMambaTalk 模型 |
| `dualtalk_trainer.py` | 双人训练器 |
| `configs/dualmambatalk.yaml` | 配置文件 |

**不需要修改的文件**：
- `train.py` / `test.py` — 已有动态加载机制 `__import__(f"{args.trainer}_trainer")`
- VQ-VAE 权重 — 复用 `pretrained/pretrained_vq/mhr_body|hand|face.bin`
- `render.py` — 输出 npz 格式兼容

### 2.6 配置参数

```yaml
# configs/dualmambatalk.yaml
dataset: beat_dual                          # 双人数据加载器
model: dualmambatalk                        # 双人模型
g_name: DualMambaTalk                       # 模型类名
trainer: dualtalk                           # 双人训练器
cache_path: datasets/beat_cache/mhr_dualmambatalk/
new_cache: True
batch_size: 16
lr_base: 2e-4
epochs: 150
# loss 参数与单人 MambaTalk V4 一致
```

### 2.7 训练命令

```bash
# 首次训练（需要构建双人缓存，约 30 分钟）
export https_proxy=http://127.0.0.1:7893
CUDA_VISIBLE_DEVICES=6 python train.py --config configs/dualmambatalk.yaml

# 推理
python test.py --config configs/dualmambatalk.yaml

# 渲染
conda activate sam_3d_body
export PYOPENGL_PLATFORM=osmesa
xvfb-run -a python render.py --npy_path <result.npz> --wav_path <audio.wav> --save_dir outputs/render
```

---

## 三、版本演进总览

| 版本 | 核心改动 | 状态 |
|------|---------|------|
| V1 | MHR 基础适配（323d 布局） | 完成 |
| V2 | Bug 修复 + velocity loss | 完成 |
| V3 | cam_t 移除（320d）+ 手部改进 | 完成 |
| V4 | 加速度损失 + 标签平滑 + lr 调优 | 完成，epoch 65 checkpoint 可用 |
| DualMambaTalk | 双人对话功能 | 缓存构建中 |

---

## 四、完整文件修改清单

| 文件 | V4 改动 | DualMambaTalk |
|------|---------|--------------|
| `mambatalk_trainer.py` | accel loss + label smoothing | 不变 |
| `configs/mambatalk.yaml` | acc 权重 + lr + batch_size + epochs | 不变 |
| `utils/config.py` | 注册 acc/pose 参数 | 不变 |
| `dataloaders/beat_dual.py` | — | 新建 |
| `models/dualmambatalk.py` | — | 新建 |
| `dualtalk_trainer.py` | — | 新建 |
| `configs/dualmambatalk.yaml` | — | 新建 |
