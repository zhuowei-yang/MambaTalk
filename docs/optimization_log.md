# MambaTalk MHR 适配 — 问题排查与优化记录

> 本文档记录了 MambaTalk 从 SMPL-X 迁移到 MHR 原生参数后，在训练、推理、渲染全链路中发现的问题及对应的优化方案。按时间线排列。

---

## 一、训练阶段问题

### 1.1 CUDA device-side assert triggered

**现象**: 训练启动后在第 8-9 个 batch 崩溃，报 `RuntimeError: CUDA error: device-side assert triggered`。

**排查过程**:
- 用 `CUDA_LAUNCH_BLOCKING=1` 定位到 `nn.Embedding(25, 768)` 的 speaker embedding
- 用 `compute-sanitizer` 获取底层错误：`Indexing.cu:1292: indexSelectLargeIndex: Assertion srcIndex < srcSelectDimSize failed`
- 编写诊断脚本逐批扫描数据集，发现添加 NLLLoss 计算后特定批次崩溃
- 最终扫描全部 1639 个缓存样本的 `vid` 字段

**根因**: 数据集中部分样本的 speaker ID (`vid`) 值为 `-1`（非 BEAT 格式数据的默认填充值）。`float(-1.0)` 转 `.long()` 得到 `-1`，传入 `nn.Embedding(25, 768)` 时索引越界。

**修复**:
```python
# mambatalk_trainer.py _load_data()
tar_id = dict_data["id"].to(self.rank).long().clamp(0, 24)
```

**教训**: 数据预处理中的哨兵值（-1）必须在模型输入前过滤，尤其是用作 Embedding 索引的字段。

---

### 1.2 VQ-VAE Quantizer NaN 导致索引越界

**现象**: 部分数据在 VQ-VAE 编码时，encoder 输出包含 NaN，导致 `argmin` 在距离矩阵上返回未定义值，后续 `get_codebook_entry` 的 Embedding 查找崩溃。

**修复**: 在 `models/quantizer.py` 的 `forward()`, `map2index()`, `get_codebook_entry()` 三个方法中添加防护：
```python
# map2index() 中
if torch.isnan(z_flattened).any() or torch.isinf(z_flattened).any():
    z_flattened = torch.nan_to_num(z_flattened, nan=0.0, posinf=1e6, neginf=-1e6)
min_encoding_indices = min_encoding_indices.clamp(0, self.n_e - 1)

# get_codebook_entry() 中
index_flattened = indices.view(-1).clamp(0, self.n_e - 1)
```

---

### 1.3 eval_model (VAESKConv) 维度不匹配

**现象**: 原始 `VAESKConv` 评估模型权重基于 SMPL-X (330维)，与 MHR (323/320维) 不兼容，`load_state_dict` 报 key mismatch。

**修复**: train.py 和 test.py 中添加 try-except 容错，FID 评估降级跳过：
```python
try:
    self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
    other_tools.load_checkpoints(...)
except Exception as e:
    logger.warning(f"eval_model load failed: {e}, FID evaluation will be skipped")
    self.eval_model = None
    self.eval_copy = None
```

---

### 1.4 训练结束 wandb 日志报错

**现象**: 200 epoch 训练完成后报 `TypeError: 'EpochTracker' object is not subscriptable`。

**原因**: `train.py` 末尾 `wandb.log({"fid_test": trainer.tracker["fid"]["test"]["best"]})` 使用了 `[]` 下标访问 `EpochTracker`，但该类不支持。

**影响**: 不影响训练和模型保存，仅影响最终的 wandb 指标记录。

---

## 二、推理阶段问题

### 2.1 _g_test 的 in-place 修改 bug（GT 数据被污染）

**现象**: 渲染 GT 视频时人物每 60 帧突然跳动一次，但用参考渲染脚本直接渲染原始 npz 无此问题。

**排查**:
```python
# GT frame 60 vs 原始数据 frame 60
GT[60] vs orig[60]: diff=7.8009  # 完全不同！
GT[64] vs orig[64]: diff=0.0000  # 又恢复正常
```
跳变位置严格对应 clip 边界（每 60 帧 = pose_length 64 - pre_frames 4）。

**根因**: `_g_test()` 中的自回归反馈修改了 GT 数据：
```python
latent_all = tar_pose          # 引用，不是副本！
latent_all_tmp = latent_all[:, slice, :]  # 视图
latent_all_tmp[:, :4, :] = latent_last[...]  # 直接覆写了 tar_pose！
```
`latent_all` 是 `tar_pose` 的引用，slice 返回视图，赋值操作直接覆盖了 GT 的 frame 60-63。

**修复**: 添加 `.clone()` 创建副本：
```python
latent_all_tmp = latent_all[:, slice, :].clone()
```

**教训**: PyTorch tensor slice 是视图（view），in-place 赋值会修改原始数据。涉及自回归生成时必须 clone。

---

### 2.2 推理输出缺少渲染必需的常量参数

**现象**: 渲染时缺少 `focal_length`、`scale_params`、`width`、`height`，被迫从原始 SAM-3D-Body npz 补全。

**分析**: 这些参数是每视频/每人物的常量，不随帧变化，不应由模型逐帧预测：
| 参数 | 维度 | 特性 | 来源 |
|------|------|------|------|
| focal_length | 1 | per-video 常量 | SAM-3D-Body 估计 |
| scale_params | 28 | per-person 常量 | SAM-3D-Body 估计 |
| shape_params | 45 | per-person 常量 | SAM-3D-Body 估计 |
| width, height | 各1 | per-video 常量 | 原始视频元数据 |

**修复**: 在 `test()` 和 `test_demo()` 中从原始数据读取并保存到输出 npz：
```python
scale_params = gt_npz['scale_params'][0]
focal_length = float(gt_npz['focal_length'][0])
# ... 写入 render_consts dict
np.savez(..., **render_consts)
```

---

## 三、渲染阶段问题

### 3.1 视野忽大忽小

**现象**: res 视频中人物时大时小，视角不稳定。GT 在修复 clone bug 后正常。

**数据对比**:
| 指标 | 原始数据 | 模型预测 |
|------|---------|---------|
| cam_t z-axis std | 0.000000 | 0.295 |
| cam_t z-axis 均值 | 3.93 | 2.84 |
| clip 边界跳变(>0.1) | 0 次 | 5 次 |

**根因**: `cam_t`（相机位置）在原始数据中几乎恒定，但模型的 global 分支（10d）把它当成逐帧变化参数学习，预测值波动大且均值偏移。

**优化方案（V3）**: 将 cam_t 从 global 分支彻底移除：
```
旧 global (10d): global_rot(3) + cam_t(3) + contact(4)
新 global (7d):  global_rot(3) + contact(4)
cam_t: 序列级常量，从数据的 trans 字段取均值
```

---

### 3.2 身体抖动

**现象**: res 视频中人物身体帧间抖动明显，GT 无此问题。

**数据对比**:
| 指标 | GT | 模型预测 |
|------|------|---------|
| body 帧间速度 | 0.0007 | 0.0017 (2.36x) |
| global_rot std | 0.23 | 0.73 (3.2x) |

**根因**: 训练 loss 中无时序平滑约束，模型预测的帧间跳变比 GT 大 2-3 倍。

**修复**: 在 `_g_training` 中添加 velocity loss：
```python
loss_vel_global = vel_loss(rec[:, 1:] - rec[:, :-1], tar[:, 1:] - tar[:, :-1])
loss_vel_body = vel_loss(...)
loss_vel = vel_global_weight * loss_vel_global + vel_body_weight * loss_vel_body
```
训练结果：vel loss 从 0.555 降至 0.033（降幅 94%）。

---

### 3.3 手部动作僵硬

**现象**: 渲染出的手部动作缺乏表现力，看起来比较僵硬。

**数据对比**:
| 指标 | GT | 模型预测 | 比值 |
|------|------|---------|------|
| 手部 MAE | — | 0.0462 | — |
| 手部帧间速度 | 0.0027 | 0.0013 | 0.47x (太僵) |

手部预测速度只有 GT 的 47%，模型输出过于平滑，丢失了细粒度的手指运动。

**优化方案（V3）**:
1. 添加手部 velocity loss（权重 1.0），鼓励模型保留手部动态
2. 提高手部分类权重 `ch: 1 → 2`，更准确匹配 VQ-VAE codebook entry

---

## 四、版本演进

### V1: MHR 基础适配
- 323d 布局：body(130) + hand(108) + face(75) + global(10)
- 4 个独立 VQ-VAE（face/body/hand + global VAE）
- 基础 MSE + NLLLoss 训练

### V2: Bug 修复 + velocity loss
- 修复 speaker ID 越界（vid=-1 clamp）
- 修复 Quantizer NaN 防护
- 修复 _g_test clone bug
- 添加 global + body 的 velocity loss
- 推理输出保存 scale_params/focal_length 等渲染常量

### V3: cam_t 移除 + 手部改进（当前）
- 320d 布局：body(130) + hand(108) + face(75) + global(7)
- cam_t 从 global 分支移除，作为序列级常量处理
- 添加手部 velocity loss（权重 1.0）
- 提高手部分类权重 ch: 1 → 2
- render.py 完全自包含，不依赖原始数据目录

---

## 五、当前配置参数一览

```yaml
# 维度
pose_dims: 320          # body(130)+hand(108)+face(75)+global(7)
global_dims: 7          # global_rot(3)+contact(4)，无 cam_t

# Loss 权重
ll: 3                   # global latent MSE
lf: 3                   # face latent MSE
lu: 3                   # body latent MSE
lh: 3                   # hand latent MSE
cu: 1                   # body classification
ch: 2                   # hand classification (V3 提高)
cf: 0                   # face classification (禁用)
vel_global_weight: 2.0  # global velocity loss
vel_body_weight: 0.5    # body velocity loss
vel_hands_weight: 1.0   # hand velocity loss (V3 新增)
```

---

## 六、文件修改清单

| 文件 | 改动轮次 | 改动内容 |
|------|---------|---------|
| `dataloaders/beat_sep_lower.py` | V1+V3 | MHR 加载，cam_t 移出 global，pyarrow copy |
| `mambatalk_trainer.py` | V1+V2+V3 | MHR 切片，vel loss，clone fix，cam_t 常量化 |
| `models/mambatalk.py` | V1 | 4 分支输出，global_dims 参数化 |
| `models/quantizer.py` | V2 | NaN/Inf 防护，索引 clamp |
| `configs/mambatalk.yaml` | V1+V2+V3 | 维度、loss 权重、vel 参数 |
| `utils/config.py` | V2+V3 | 注册 vel_*_weight 参数 |
| `utils/other_tools.py` | V2 | checkpoint CPU 加载 |
| `train.py` | V1+V2 | 移除 smplx，eval_model 容错 |
| `test.py` | V1+V2 | 移除 smplx，eval_model 容错 |
| `render.py` | V2+V3 | 自包含渲染，从 npz 读取全部参数 |

---

## 七、训练与渲染命令

```bash
# 训练（首次需 new_cache: True 重建缓存）
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/mambatalk.yaml \
    --test_start_epoch 80 --test_period 10

# 推理
python test.py --config configs/mambatalk.yaml

# 渲染（需 sam_3d_body conda 环境）
conda activate sam_3d_body
export PYOPENGL_PLATFORM=osmesa
xvfb-run -a python render.py \
    --npy_path <result.npz> \
    --wav_path <audio.wav> \
    --save_dir outputs/render
```
