"""
VQ-VAE 重建质量验证脚本 (MambaTalk_new_512)
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

# 使用 512 的 body VAE 路径
BODY_VAE_PATH = "outputs/audio2pose/custom/0306_155503_mhr_cont_body"
# 使用 256 的 hand VAE 路径 (复用 MambaTalk_new 的)
HAND_VAE_PATH = "/data1/yangzhuowei/MambaTalk_new/outputs/audio2pose/custom/0305_054312_mhr_cont_hand"
DATA_DIR = "data_new/smplxflame_30"
OUTPUT_DIR = "outputs/vae_verify"
TEST_EPOCH = 1000


def load_test_results(vae_path, epoch):
    test_dir = os.path.join(vae_path, str(epoch))
    if not os.path.exists(test_dir):
        # 尝试找最近的 epoch
        if os.path.exists(vae_path):
            epochs = [int(d) for d in os.listdir(vae_path) if d.isdigit()]
            if epochs:
                epoch = max(epochs)
                test_dir = os.path.join(vae_path, str(epoch))
                print(f"Warning: Epoch {TEST_EPOCH} not found, using {epoch}")
            else:
                return []
        else:
            return []
            
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("VQ-VAE 重建质量验证 (MambaTalk_new_512)")
    print("=" * 70)

    # ========== Body VQ-VAE (260d continuous) ==========
    print("\n### Body VQ-VAE (260d continuous representation, codebook=512)")
    print("-" * 50)
    body_results = load_test_results(BODY_VAE_PATH, TEST_EPOCH)

    if body_results:
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
    else:
        print("  No body results found.")

    # ========== Hand VQ-VAE (108d PCA) ==========
    print("\n### Hand VQ-VAE (108d PCA, codebook=256)")
    print("-" * 50)
    hand_results = load_test_results(HAND_VAE_PATH, TEST_EPOCH)

    if hand_results:
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
    else:
        print("  No hand results found.")


if __name__ == "__main__":
    main()
