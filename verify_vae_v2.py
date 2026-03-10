"""
VQ-VAE v1 vs v2 对比验证 + 渲染 npz 生成
"""
import os, sys, numpy as np, torch
sys.path.insert(0, '.')
from dataloaders.mhr_utils import compact_cont_to_model_params_body, rot6d_to_euler3

DATA_DIR = "data/smplxflame_30"
OUTPUT_DIR = "outputs/vae_verify_v2"

BODY_V1_PATH = "outputs/audio2pose/custom/0302_182728_mhr_cont_body"
BODY_V2_PATH = "outputs/audio2pose/custom/0303_082002_mhr_cont_body_v2"
HAND_V1_PATH = "outputs/audio2pose/custom/0302_183530_mhr_cont_hand"
HAND_V2_PATH = "outputs/audio2pose/custom/0303_073628_mhr_cont_hand_v2"


def find_latest_test_epoch(vae_path):
    epochs = []
    for d in os.listdir(vae_path):
        if d.isdigit() and os.path.isdir(os.path.join(vae_path, d)):
            epochs.append(int(d))
    return max(epochs) if epochs else None


def load_results(vae_path, epoch):
    test_dir = os.path.join(vae_path, str(epoch))
    results = {}
    for f in sorted(os.listdir(test_dir)):
        if f.startswith("res_") and f.endswith(".npz"):
            data = np.load(os.path.join(test_dir, f), allow_pickle=True)
            name = f[4:-4]
            results[name] = {'rec': data['rec_pose'].squeeze(), 'tar': data['tar_pose'].squeeze()}
    return results


def amplitude_stats(rec, tar):
    n = min(rec.shape[0], tar.shape[0])
    rec, tar = rec[:n], tar[:n]
    tar_range = tar.max(axis=0) - tar.min(axis=0)
    rec_range = rec.max(axis=0) - rec.min(axis=0)
    ratio = rec_range / (tar_range + 1e-8)
    active = tar_range > 0.01
    tar_vel = np.abs(np.diff(tar, axis=0)).mean(axis=0)
    rec_vel = np.abs(np.diff(rec, axis=0)).mean(axis=0)
    vel_ratio = rec_vel / (tar_vel + 1e-8)
    return {
        'amp_ratio': ratio[active].mean() if active.any() else ratio.mean(),
        'vel_ratio': vel_ratio[active].mean() if active.any() else vel_ratio.mean(),
        'mse': np.mean((rec - tar) ** 2),
    }


def save_mhr_npz(body_euler, hand_pca, gt_npz_path, save_path):
    gt = np.load(gt_npz_path, allow_pickle=True)
    n = body_euler.shape[0]
    fl = float(gt['focal_length'][0]) if 'focal_length' in gt.files else 1500.0
    np.savez(save_path,
             body_pose_params=body_euler,
             hand_pose_params=hand_pca,
             expr_params=np.zeros((n, 72), dtype=np.float32),
             global_rot=gt['global_rot'][:n],
             pred_cam_t=gt['pred_cam_t'][:n],
             shape_params=gt['shape_params'] if 'shape_params' in gt.files else np.zeros(45),
             scale_params=np.repeat(gt['scale_params'][:1], n, axis=0) if 'scale_params' in gt.files else np.zeros((n, 28)),
             focal_length=np.full(n, fl, dtype=np.float32),
             width=gt['width'] if 'width' in gt.files else np.array([1080]),
             height=gt['height'] if 'height' in gt.files else np.array([1920]),
             frame_idx=np.arange(n, dtype=np.int32),
             fps=np.array([30.0], dtype=np.float32))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    body_v1_ep = find_latest_test_epoch(BODY_V1_PATH)
    body_v2_ep = find_latest_test_epoch(BODY_V2_PATH)
    hand_v1_ep = find_latest_test_epoch(HAND_V1_PATH)
    hand_v2_ep = find_latest_test_epoch(HAND_V2_PATH)
    print(f"Body v1: epoch {body_v1_ep}, Body v2: epoch {body_v2_ep}")
    print(f"Hand v1: epoch {hand_v1_ep}, Hand v2: epoch {hand_v2_ep}")

    body_v1 = load_results(BODY_V1_PATH, body_v1_ep)
    body_v2 = load_results(BODY_V2_PATH, body_v2_ep)
    hand_v1 = load_results(HAND_V1_PATH, hand_v1_ep)
    hand_v2 = load_results(HAND_V2_PATH, hand_v2_ep)

    print("\n" + "=" * 60)
    print("Body VQ-VAE: v1 vs v2")
    print("=" * 60)
    for name in list(body_v1.keys())[:5]:
        s1 = amplitude_stats(body_v1[name]['rec'], body_v1[name]['tar'])
        if name in body_v2:
            s2 = amplitude_stats(body_v2[name]['rec'], body_v2[name]['tar'])
            print(f"  {name}:")
            print(f"    v1: amp={s1['amp_ratio']:.3f} vel={s1['vel_ratio']:.3f} mse={s1['mse']:.6f}")
            print(f"    v2: amp={s2['amp_ratio']:.3f} vel={s2['vel_ratio']:.3f} mse={s2['mse']:.6f}")

    print("\n" + "=" * 60)
    print("Hand VQ-VAE: v1 vs v2")
    print("=" * 60)
    v1_amps, v2_amps = [], []
    for name in list(hand_v1.keys())[:10]:
        s1 = amplitude_stats(hand_v1[name]['rec'], hand_v1[name]['tar'])
        v1_amps.append(s1['amp_ratio'])
        if name in hand_v2:
            s2 = amplitude_stats(hand_v2[name]['rec'], hand_v2[name]['tar'])
            v2_amps.append(s2['amp_ratio'])
            print(f"  {name}:")
            print(f"    v1: amp={s1['amp_ratio']:.3f} vel={s1['vel_ratio']:.3f} mse={s1['mse']:.6f}")
            print(f"    v2: amp={s2['amp_ratio']:.3f} vel={s2['vel_ratio']:.3f} mse={s2['mse']:.6f}")
    print(f"\n  Average hand amplitude ratio: v1={np.mean(v1_amps):.3f}, v2={np.mean(v2_amps):.3f}")

    # Export renderable npz: 3 versions for first sample
    print("\n" + "=" * 60)
    print("Exporting renderable npz")
    print("=" * 60)
    sample_name = list(body_v1.keys())[0]
    gt_path = os.path.join(DATA_DIR, sample_name + ".npz")
    if not os.path.exists(gt_path):
        print(f"GT not found: {gt_path}")
        return

    gt_data = np.load(gt_path, allow_pickle=True)
    gt_hand = gt_data['hand_pose_params']
    gt_body = gt_data['body_pose_params']
    gt_n = gt_body.shape[0]

    # Body v2 cont -> euler
    body_v2_cont = body_v2[sample_name]['rec']
    body_v2_euler = compact_cont_to_model_params_body(torch.from_numpy(body_v2_cont).float()).numpy()
    n = body_v2_euler.shape[0]

    # Hand v1
    hand_v1_rec = hand_v1[sample_name]['rec'][:n]
    # Hand v2
    hand_v2_rec = hand_v2[sample_name]['rec'][:n] if sample_name in hand_v2 else hand_v1_rec

    # 1. GT
    save_mhr_npz(gt_body[:n], gt_hand[:n], gt_path,
                 os.path.join(OUTPUT_DIR, f"gt_{sample_name}.npz"))
    # 2. Body v2 + Hand v1
    save_mhr_npz(body_v2_euler, hand_v1_rec, gt_path,
                 os.path.join(OUTPUT_DIR, f"body_v2_hand_v1_{sample_name}.npz"))
    # 3. Body v2 + Hand v2
    save_mhr_npz(body_v2_euler, hand_v2_rec, gt_path,
                 os.path.join(OUTPUT_DIR, f"body_v2_hand_v2_{sample_name}.npz"))

    print(f"  Saved GT, body_v2+hand_v1, body_v2+hand_v2 for {sample_name}")
    print(f"  Files in: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.npz'):
            print(f"    {f}")


if __name__ == "__main__":
    main()
