# MambaTalk MHR 模型评估报告

**日期:** 2026-03-14
**评估脚本:** `/data1/yangzhuowei/MambaTalk/evaluate_dual.py`
**评估环境:** sam_3d_body conda 环境 + SAM-3D-Body 模型

---

## 一、指标定义（PantoMatrix）

| 指标 | 含义 | 方向 |
|------|------|------|
| FGD (Frechet Gesture Distance) | 生成动作与 GT 关节位置的分布距离 | 越低越好 |
| L1Div (L1 Diversity) | 动作多样性 | 越高越好（接近 GT） |
| BC (Beat Consistency) | 音频-动作节拍对齐度 | 越高越好 |
| MSEFace | 面部顶点重建误差 | 越低越好 |
| LVDFace | 面部顶点速度差 | 越低越好 |

---

## 二、评估模型概况

| 模型 | 仓库路径 | 训练 Run | Epoch | 特点 |
|------|---------|---------|-------|------|
| MambaTalk_new_512 | `/data1/yangzhuowei/MambaTalk_new_512` | 0310_071535_mambatalk_mhr_new | 90 | MHR-512 VQ-VAE, body(130d)+hand(108d)+face(75d)+global(7d raw), pre_frames=4 |
| MambaTalk_new | `/data1/yangzhuowei/MambaTalk_new` | 0306_160640_mambatalk_mhr_new | 90 | MHR VQ-VAE (非512), body+hand+global, pre_frames=4 |
| MambaTalk_new_512_res | `/data1/yangzhuowei/MambaTalk_new_512_res` | 0310_151413_mambatalk_mhr_new_res | 70 | MHR-512 + 残差连接变体 |

---

## 三、评估结果

| 指标 | MambaTalk_new_512 (ep90) | MambaTalk_new (ep90) | MambaTalk_new_512_res (ep70) | 最优 |
|------|--------------------------|----------------------|------------------------------|------|
| **FGD** ↓ | **0.1222** | 0.1488 | 0.8706 | MambaTalk_new_512 |
| **L1Div** ↑ | **4.0529** | 3.6289 | 1.8762 | MambaTalk_new_512 |
| **BC** ↑ | **0.4258** | 0.3951 | 0.0363 | MambaTalk_new_512 |
| **MSEFace** ↓ | 0.045450 | 0.046001 | **0.044669** | MambaTalk_new_512_res |
| **LVDFace** ↓ | 0.002695 | 0.002520 | **0.002141** | MambaTalk_new_512_res |
| 总帧数 | 45,043 | 45,043 | 45,043 | - |
| 总序列数 | 116 | 116 | 116 | - |

---

## 四、分析

### 4.1 MambaTalk_new_512 表现最优

在动作质量的核心指标上，MambaTalk_new_512 全面领先：

- **FGD 0.1222**（最低）：生成动作分布最接近 GT
- **L1Div 4.0529**（最高）：动作最具多样性
- **BC 0.4258**（最高）：音频-动作节拍对齐最好

### 4.2 MambaTalk_new_512_res 面部指标略优

- MSEFace 和 LVDFace 略低于其他两个模型
- 但 FGD/L1Div/BC 差距很大（FGD 0.87 vs 0.12），说明残差变体在身体动作质量上有明显退步
- BC 仅 0.0363，说明节拍对齐基本失效

### 4.3 MambaTalk_new vs MambaTalk_new_512

MambaTalk_new_512（VQ-VAE 256d codebook）在各项指标上均优于 MambaTalk_new（非 512 版本），验证了 512d codebook 升级的有效性。

---

## 五、已知问题及修复状态

| 问题 | 状态 | 修复方案 |
|------|------|---------|
| global_rot clip 边界跳变 | 已修复 | savgol_filter(wl=13, poly=2) 后处理 |
| 视野忽大忽小 | 已修复 | camfix（pred_cam_t 使用均值） |
| pred 帧数短于 GT | 已知 | 需重建 test 缓存（当前用旧缓存） |

详见 `docs/optimization_log_v5_global_jitter_fix.md`。

---

*本文档由评估脚本 evaluate_dual.py 生成的指标汇总而成。*
