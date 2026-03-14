# MambaTalk — MHR 原生参数适配改动说明

## 概述

本次改动将 MambaTalk 从 SMPL-X 关节旋转格式全面迁移到 **MHR（SAM-3D-Body）原生参数**格式。核心变化是用 Euler 角 / PCA 系数直接表征人体运动，不再需要 axis-angle → 6D rotation 转换。

### 新数据布局

| 组件 | 参数来源 | 维度 | VQ-VAE |
|------|---------|------|--------|
| Body | body_pose_params[0:130] | 130 | VQVAEConvZero |
| Hand | hand_pose_params (左54+右54) | 108 | VQVAEConvZero |
| Face | expr_params(72) + jaw(3) | 75 | VQVAEConvZero |
| Global | global_rot(3) + cam_t(3) + contact(4) | 10 | 无（直接预测） |

**总维度：323**（原 SMPLX 为 330 = 55关节×6D）

### 数据流对比

```
原始 (SMPLX):
  npz → axis-angle(165) → joint_mask拆分 → 6D rotation → VQ-VAE encode → MambaTalk → VQ-VAE decode → 6D → axis-angle → SMPLX渲染

新方案 (MHR):
  npz → body(130) / hand(108) / face(75) / global(10) → VQ-VAE encode → MambaTalk → VQ-VAE decode → MHR原生参数 → MHR渲染
```

---

## 一、训练框架改动

### 1.1 数据加载器 — `dataloaders/beat_sep_lower.py`

**新增 `load_mhr_native()` 函数**：直接从 SAM-3D-Body npz 加载原生参数，不做旋转格式转换。

**`cache_generation()` 重写**：
- 加载 body_pose(130) + jaw(3) + hand_pose(108) + expr(72) + global_rot(3) + cam_t(3) + contact(4)
- 拼接为 323 维向量缓存：`[body(130), hand(108), face(75), global(10)]`
- 接触标签从关节坐标计算（脚踝/脚趾速度阈值）
- 移除了 SMPLX 格式的 joint_mask 系统

**`__getitem__()` 修复**：
- 添加 `np.array(arr, copy=True)` 解决 pyarrow 反序列化数组只读问题

### 1.2 VQ-VAE 配置 — `configs/mhr_vqvae_*.yaml`

新建 4 个独立的 VQ-VAE 配置文件：

| 配置文件 | 维度 | 模型类型 | Trainer |
|---------|------|---------|---------|
| `mhr_vqvae_body.yaml` | 130 | VQVAEConvZero | ae_mhr |
| `mhr_vqvae_hand.yaml` | 108 | VQVAEConvZero | ae_mhr |
| `mhr_vqvae_face.yaml` | 75 | VQVAEConvZero | aeface_mhr |
| `mhr_vqvae_global.yaml` | 10 | VAEConvZero | aelower_mhr |

关键参数一致：`vae_length=256, vae_codebook_size=256, pose_length=64, stride=20`

### 1.3 VQ-VAE Trainer — `ae_mhr_trainer.py`, `aeface_mhr_trainer.py`, `aelower_mhr_trainer.py`

- 移除 SMPLX 依赖（`smplx.create`, `GeodesicLoss`, vertex loss）
- Loss：MSE 重建 + velocity loss + embedding loss
- 直接在 MHR 参数空间训练，不做 6D rotation 转换

### 1.4 主模型配置 — `configs/mambatalk.yaml`

```yaml
pose_dims: 323           # body(130)+hand(108)+face(75)+global(10)
face_dims: 75
body_dims: 130
hand_dims: 108
global_dims: 10
rot6d: False             # 不使用 6D rotation
cache_path: datasets/beat_cache/mhr_mambatalk/
```

---

## 二、VQ-VAE 层面的改动

### 2.1 Quantizer 鲁棒性修复 — `models/quantizer.py`

在 `forward()`, `map2index()`, `get_codebook_entry()` 三个方法中添加了 NaN/Inf 防护：

```python
# map2index() 中：
if torch.isnan(z_flattened).any() or torch.isinf(z_flattened).any():
    z_flattened = torch.nan_to_num(z_flattened, nan=0.0, posinf=1e6, neginf=-1e6)
# argmin 后 clamp 确保索引不越界：
min_encoding_indices = min_encoding_indices.clamp(0, self.n_e - 1)

# get_codebook_entry() 中：
index_flattened = indices.view(-1).clamp(0, self.n_e - 1)
```

**原因**：MHR 参数的值域与 SMPLX 旋转矩阵不同，某些边缘数据经过 VQ-VAE encoder 后可能产生 NaN，导致 argmin 行为未定义，进而在 Embedding 查找时触发 CUDA `indexSelectLargeIndex` 断言。

---

## 三、训练代码的 Bug 修复

### 3.1 Speaker ID 越界 — `mambatalk_trainer.py`

**问题根因**：数据集中部分样本的 speaker ID (`vid`) 为 `-1`（来自非 BEAT 格式数据的默认值）。当 `float(-1.0)` 转为 `.long()` 得到 `-1`，传入 `nn.Embedding(25, 768)` 时触发 CUDA 断言 `srcIndex < srcSelectDimSize`。

**修复**：
```python
# _load_data() 中：
tar_id = dict_data["id"].to(self.rank).long().clamp(0, 24)
```

### 3.2 eval_model 维度不匹配 — `train.py`

**问题**：`VAESKConv` 评估模型的预训练权重基于 SMPLX (330维)，与 MHR (323维) 不兼容，`load_state_dict` 报错。

**修复**：添加 try-except 容错，FID 评估降级为跳过：
```python
try:
    self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
    other_tools.load_checkpoints(...)
except Exception as e:
    logger.warning(f"eval_model load failed: {e}, FID evaluation will be skipped")
    self.eval_model = None
    self.eval_copy = None
```

### 3.3 checkpoint 加载设备映射 — `utils/other_tools.py`

```python
# load_checkpoints() 中使用 CPU 加载避免 CUDA 设备映射冲突：
states = torch.load(save_path, map_location='cpu')
```

---

## 四、模型结构改动

### 4.1 MambaTalk 主模型 — `models/mambatalk.py`

**GlobalScan**：
- `motion_encoder`：输入维度 323（MHR 总维度），输出 256（motion_f）

**LocalScan**：
- 新增 4 条独立分支：face / body / hands / global
- `motion_down_global`：`hidden_size → global_dims(10)`（直接输出，无 codebook）
- face / body / hands 各自有 `motion_down_*` (hidden→256) 和 `*_classifier` (256→256)

### 4.2 MambaTalk Trainer — `mambatalk_trainer.py`

**`__init__`**：加载 3 个 MHR VQ-VAE（face/body/hands），Global 直接预测无 VQ-VAE

**`_load_data`**：
- 323 维 pose 按 `[0:130, 130:238, 238:313, 313:323]` 切片
- VQ-VAE 编码（frozen）得到 latent + index targets
- 移除 axis-angle → 6D rotation 转换

**`_g_training`**：
- MSE latent loss (face/body/hands/global) + NLLLoss cls loss (face/body/hands)
- val 模式：VQ-VAE 解码回 MHR 参数
- 移除 `joint_mask` / `inverse_selection_tensor` 系统

**`_g_test`**：自回归推理循环，每步解码 body+hand+face+global 拼接反馈

---

## 五、推理代码改动

### 5.1 `test.py`

- 移除 `import smplx` 和 `self.smplx = smplx.create(...)`
- eval_model 加载添加 try-except 容错
- `self.alignmenter = None`（train_data 在推理模式不加载）

### 5.2 `mambatalk_trainer.py` — `test()` / `test_demo()`

- NPZ 输出改为 MHR 原生格式：
  - `body_pose_params`：body(130) + jaw(3) = 133 维
  - `hand_pose_params`：108 维
  - `expr_params`：72 维
  - `global_rot`：3 维
  - `pred_cam_t`：3 维
  - `shape_params`：45 维（从 GT 复制）

### 5.3 推理使用方法

```bash
# 训练完成后，更新 test_ckpt 路径
# configs/mambatalk.yaml:
#   test_ckpt: ./outputs/audio2pose/custom/<训练目录>/last_<epoch>.bin

python test.py --config configs/mambatalk.yaml
```

---

## 六、训练流程

```bash
# 1. 训练 4 个 VQ-VAE（已完成）
python train.py --config configs/mhr_vqvae_body.yaml
python train.py --config configs/mhr_vqvae_hand.yaml
python train.py --config configs/mhr_vqvae_face.yaml
python train.py --config configs/mhr_vqvae_global.yaml

# 2. 将权重复制到 pretrained_vq/
cp <body_ckpt> pretrained/pretrained_vq/mhr_body.bin
cp <hand_ckpt> pretrained/pretrained_vq/mhr_hand.bin
cp <face_ckpt> pretrained/pretrained_vq/mhr_face.bin

# 3. 训练 MambaTalk 主模型（进行中）
CUDA_VISIBLE_DEVICES=5 python train.py --config configs/mambatalk.yaml

# 4. 推理
python test.py --config configs/mambatalk.yaml
```

---

## 修改文件清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `dataloaders/beat_sep_lower.py` | 重写 | MHR 数据加载、缓存生成、pyarrow 只读修复 |
| `configs/mhr_vqvae_*.yaml` (×4) | 新建 | 4 个 VQ-VAE 配置 |
| `configs/mambatalk.yaml` | 修改 | 维度、缓存路径、MHR 参数 |
| `ae_mhr_trainer.py` | 新建 | Body/Hand VQ-VAE trainer |
| `aeface_mhr_trainer.py` | 新建 | Face VQ-VAE trainer |
| `aelower_mhr_trainer.py` | 新建 | Global VAE trainer |
| `models/mambatalk.py` | 重写 | 4 分支输出、维度适配 |
| `models/quantizer.py` | 修改 | NaN/Inf 防护、索引 clamp |
| `mambatalk_trainer.py` | 重写 | MHR 切片、VQ-VAE 加载、loss、推理 |
| `train.py` | 修改 | 移除 smplx、eval_model 容错 |
| `test.py` | 修改 | 移除 smplx、eval_model 容错 |
| `utils/other_tools.py` | 修改 | checkpoint CPU 加载 |
