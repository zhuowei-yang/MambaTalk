"""
VQ-VAE 重建质量验证脚本
用法: conda activate mambatalk && python verify_vae.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from dataloaders.mhr_utils import (
    compact_model_params_to_cont_body,
    compact_cont_to_model_params_body,
    euler3_to_rot6d,
    rot6d_to_euler3,
    CONT_DIM,
)

BODY_VAE_PATH = "outputs/audio2pose/custom/0302_182728_mhr_cont_body"
HAND_VAE_PATH = "outputs/audio2pose/custom/0302_183530_mhr_cont_hand"
DATA_DIR = "data/smplxflame_30"
OUTPUT_DIR = "outputs/vae_verify"
TEST_EPOCH = 500


def load_test_results(vae_path, epoch):
    test_dir = os.path.join(vae_path, str(epoch))
    results = []
    for f in sorted(os.listdir(test_dir)):
        if f.startswith("res_") and f.endswith(".npz"):
            data = np.load(os.path.join(test_dir, f), allow_pickle=True)
            name = f[4:-4]
            results.append({
                'name': name,
                'rec_pose': data['rec_pose'],
                'tar_pose': data['tar_pose'],
                'rec_euler': data.get('rec_euler'),
                'tar_euler': data.get('tar_euler'),
            })
    return results


def compute_metrics(rec, tar):
    mse = np.mean((rec - tar) ** 2)
    rec_vel = np.diff(rec, axis=0)
    tar_vel = np.diff(tar, axis=0)
    vel_err = np.mean(np.abs(rec_vel - tar_vel))
    rec_acc = np.diff(rec, n=2, axis=0)
    tar_acc = np.diff(tar, n=2, axis=0)
    acc_err = np.mean(np.abs(rec_acc - tar_acc))
    return mse, vel_err, acc_err


def compute_jitter(pose_seq):
    """Compute mean absolute acceleration (jitter metric)."""
    acc = np.diff(pose_seq, n=2, axis=0)
    return np.mean(np.abs(acc))


def save_mhr_npz(body_euler, hand_pca, global_rot_euler, cam_t, gt_npz_path, save_path):
    """Save reconstructed motion as MHR-compatible npz for rendering."""
    gt = np.load(gt_npz_path, allow_pickle=True)
    n = body_euler.shape[0]
    fl_val = float(gt['focal_length'][0]) if 'focal_length' in gt.files else 1500.0
    np.savez(save_path,
             body_pose_params=body_euler,
             hand_pose_params=hand_pca if hand_pca is not None else gt['hand_pose_params'][:n],
             expr_params=np.zeros((n, 72), dtype=np.float32),
             global_rot=global_rot_euler if global_rot_euler is not None else gt['global_rot'][:n],
             pred_cam_t=cam_t if cam_t is not None else gt['pred_cam_t'][:n],
             shape_params=gt['shape_params'] if 'shape_params' in gt.files else np.zeros(45, dtype=np.float32),
             scale_params=np.repeat(gt['scale_params'][:1], n, axis=0) if 'scale_params' in gt.files else np.zeros((n, 28), dtype=np.float32),
             focal_length=np.full(n, fl_val, dtype=np.float32),
             width=gt['width'] if 'width' in gt.files else np.array([1080]),
             height=gt['height'] if 'height' in gt.files else np.array([1920]),
             frame_idx=np.arange(n, dtype=np.int32),
             fps=np.array([30.0], dtype=np.float32),
             mocap_frame_rate=30)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("VQ-VAE 重建质量验证")
    print("=" * 70)

    # ========== Body VQ-VAE (260d continuous) ==========
    print("\n### Body VQ-VAE (260d continuous representation)")
    print("-" * 50)
    body_results = load_test_results(BODY_VAE_PATH, TEST_EPOCH)
    if not body_results:
        body_results = load_test_results(BODY_VAE_PATH, 100)

    body_mse_list, body_vel_list, body_acc_list = [], [], []
    body_jitter_gt_list, body_jitter_rec_list = [], []

    for r in body_results:
        rec = r['rec_pose'].squeeze()
        tar = r['tar_pose'].squeeze()
        n = min(rec.shape[0], tar.shape[0])
        rec, tar = rec[:n], tar[:n]
        mse, vel_err, acc_err = compute_metrics(rec, tar)
        body_mse_list.append(mse)
        body_vel_list.append(vel_err)
        body_acc_list.append(acc_err)
        body_jitter_gt_list.append(compute_jitter(tar))
        body_jitter_rec_list.append(compute_jitter(rec))

    print(f"  Samples: {len(body_results)}")
    print(f"  MSE (continuous space): {np.mean(body_mse_list):.6f}")
    print(f"  Velocity error:         {np.mean(body_vel_list):.6f}")
    print(f"  Acceleration error:     {np.mean(body_acc_list):.6f}")
    print(f"  Jitter (GT):            {np.mean(body_jitter_gt_list):.6f}")
    print(f"  Jitter (Reconstructed): {np.mean(body_jitter_rec_list):.6f}")
    jitter_ratio = np.mean(body_jitter_rec_list) / max(np.mean(body_jitter_gt_list), 1e-10)
    print(f"  Jitter ratio (rec/gt):  {jitter_ratio:.4f}  {'OK (<1.5)' if jitter_ratio < 1.5 else 'WARNING: jitter increased!'}")

    if body_results and body_results[0].get('rec_euler') is not None:
        print("\n  Body Euler roundtrip check:")
        euler_errors = []
        for r in body_results[:5]:
            rec_euler = r['rec_euler'].squeeze()
            tar_euler = r['tar_euler'].squeeze()
            n = min(rec_euler.shape[0], tar_euler.shape[0])
            euler_errors.append(np.mean(np.abs(rec_euler[:n] - tar_euler[:n])))
        print(f"    Mean Euler error: {np.mean(euler_errors):.6f} rad")
        print(f"    Mean Euler error: {np.degrees(np.mean(euler_errors)):.4f} deg")

    # ========== Hand VQ-VAE (108d PCA) ==========
    print("\n### Hand VQ-VAE (108d PCA)")
    print("-" * 50)
    hand_results = load_test_results(HAND_VAE_PATH, TEST_EPOCH)
    if not hand_results:
        hand_results = load_test_results(HAND_VAE_PATH, 100)

    hand_mse_list, hand_vel_list, hand_acc_list = [], [], []
    hand_jitter_gt_list, hand_jitter_rec_list = [], []

    for r in hand_results:
        rec = r['rec_pose'].squeeze()
        tar = r['tar_pose'].squeeze()
        n = min(rec.shape[0], tar.shape[0])
        rec, tar = rec[:n], tar[:n]
        mse, vel_err, acc_err = compute_metrics(rec, tar)
        hand_mse_list.append(mse)
        hand_vel_list.append(vel_err)
        hand_acc_list.append(acc_err)
        hand_jitter_gt_list.append(compute_jitter(tar))
        hand_jitter_rec_list.append(compute_jitter(rec))

    print(f"  Samples: {len(hand_results)}")
    print(f"  MSE (PCA space):        {np.mean(hand_mse_list):.6f}")
    print(f"  Velocity error:         {np.mean(hand_vel_list):.6f}")
    print(f"  Acceleration error:     {np.mean(hand_acc_list):.6f}")
    print(f"  Jitter (GT):            {np.mean(hand_jitter_gt_list):.6f}")
    print(f"  Jitter (Reconstructed): {np.mean(hand_jitter_rec_list):.6f}")
    jitter_ratio_h = np.mean(hand_jitter_rec_list) / max(np.mean(hand_jitter_gt_list), 1e-10)
    print(f"  Jitter ratio (rec/gt):  {jitter_ratio_h:.4f}  {'OK (<1.5)' if jitter_ratio_h < 1.5 else 'WARNING: jitter increased!'}")

    # ========== Export renderable npz ==========
    print("\n### Exporting renderable MHR npz files")
    print("-" * 50)
    n_export = min(5, len(body_results))
    for i in range(n_export):
        name = body_results[i]['name']
        gt_npz_path = os.path.join(DATA_DIR, name + ".npz")
        if not os.path.exists(gt_npz_path):
            print(f"  Skip {name}: GT npz not found")
            continue

        rec_cont = body_results[i]['rec_pose'].squeeze()
        rec_body_cont = torch.from_numpy(rec_cont).float()
        rec_body_euler = compact_cont_to_model_params_body(rec_body_cont).numpy()
        n = rec_body_euler.shape[0]

        gt_data = np.load(gt_npz_path, allow_pickle=True)
        stride = 1
        gt_body = gt_data['body_pose_params'][:n * stride:stride][:n]
        gt_hand = gt_data['hand_pose_params'][:n * stride:stride][:n]
        gt_global = gt_data['global_rot'][:n * stride:stride][:n]
        gt_cam_t = gt_data['pred_cam_t'][:n * stride:stride][:n]

        gt_save = os.path.join(OUTPUT_DIR, f"gt_{name}.npz")
        rec_save = os.path.join(OUTPUT_DIR, f"rec_body_{name}.npz")
        save_mhr_npz(gt_body, gt_hand, gt_global, gt_cam_t, gt_npz_path, gt_save)
        save_mhr_npz(rec_body_euler, gt_hand, gt_global, gt_cam_t, gt_npz_path, rec_save)

        body_diff = np.mean(np.abs(rec_body_euler[:, :130] - gt_body[:, :130]))
        print(f"  [{i+1}] {name}")
        print(f"      Body Euler MAE: {body_diff:.6f} rad ({np.degrees(body_diff):.3f} deg)")
        print(f"      Saved: {gt_save}")
        print(f"      Saved: {rec_save}")

    print("\n" + "=" * 70)
    print("验证完成!")
    print(f"可渲染的 npz 文件保存在: {OUTPUT_DIR}/")
    print("用以下命令渲染对比视频:")
    print(f"  conda activate sam_3d_body")
    print(f"  xvfb-run -a python render.py --npy_path {OUTPUT_DIR}/gt_XXX.npz")
    print(f"  xvfb-run -a python render.py --npy_path {OUTPUT_DIR}/rec_body_XXX.npz")
    print("=" * 70)


if __name__ == "__main__":
    main()
