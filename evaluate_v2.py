#!/usr/bin/env python3
"""
MambaTalk v2 (PD-enhanced) evaluation script.

Computes metrics: FGD, L1Div, BC, MSEFace, LVDFace
using SAM-3D-Body for joint position extraction.

Usage (in sam_3d_body conda env):
  CUDA_VISIBLE_DEVICES=1 python evaluate_v2.py \
      --result_dir outputs/audio2pose/custom/0317_054525_mambatalk_mhr_new/40 \
      --data_dir ./data_new/ \
      --audio_dir ./data_new/wave16k/ \
      --device cuda
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import librosa
import math
from tqdm import tqdm
from scipy import linalg
from scipy.signal import argrelextrema

sys.path.insert(0, '/data1/yangzhuowei/PantoMatrix')
import importlib, types
_mertic_src = open('/data1/yangzhuowei/PantoMatrix/emage_evaltools/mertic.py').read()
_mertic_src = _mertic_src.replace('from .motion_encoder import VAESKConv', '')
_mertic_mod = types.ModuleType('_panto_mertic')
exec(compile(_mertic_src, 'mertic.py', 'exec'), _mertic_mod.__dict__)
PantoL1div = _mertic_mod.L1div
PantoBC = _mertic_mod.BC
PantoMSEFace = _mertic_mod.MSEFace
PantoLVDFace = _mertic_mod.LVDFace

sys.path.insert(0, '/data1/yangzhuowei/sam_3d_body')
import pyrootutils
root = pyrootutils.setup_root(
    search_from='/data1/yangzhuowei/sam_3d_body',
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True, dotenv=True,
)
from sam_3d_body import load_sam_3d_body


def load_mhr_model(device):
    checkpoint = "/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    model, _ = load_sam_3d_body(checkpoint, device=device, mhr_path=mhr_path)
    model.eval()
    return model


def mhr_forward_batch(model, npz_data, device, batch_size=64):
    N = npz_data['body_pose_params'].shape[0]
    shape = npz_data['shape_params']
    if shape.ndim == 2:
        shape = shape[0]
    scale = npz_data['scale_params']
    if scale.ndim == 2:
        scale = scale[0]

    all_joints = []
    all_verts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start

        body_pose = torch.from_numpy(npz_data['body_pose_params'][start:end]).float().to(device)
        hand_pose = torch.from_numpy(npz_data['hand_pose_params'][start:end]).float().to(device)
        expr = torch.from_numpy(npz_data['expr_params'][start:end]).float().to(device)
        g_rot = torch.from_numpy(npz_data['global_rot'][start:end]).float().to(device)
        shape_t = torch.from_numpy(shape).float().unsqueeze(0).expand(bs, -1).to(device)
        scale_t = torch.from_numpy(scale).float().unsqueeze(0).expand(bs, -1).to(device)
        g_trans = torch.zeros(bs, 3, dtype=torch.float32).to(device)

        with torch.no_grad():
            result = model.head_pose.mhr_forward(
                global_trans=g_trans, global_rot=g_rot,
                body_pose_params=body_pose, hand_pose_params=hand_pose,
                scale_params=scale_t, shape_params=shape_t, expr_params=expr,
                return_keypoints=False, return_joint_coords=True,
            )
            if isinstance(result, tuple):
                verts = result[0]
                joint_coords = result[1] if len(result) > 1 else None
            else:
                verts = result
                joint_coords = None

        all_verts.append(verts.cpu().numpy())
        if joint_coords is not None:
            all_joints.append(joint_coords.cpu().numpy())

    verts_all = np.concatenate(all_verts, axis=0)
    joints_all = np.concatenate(all_joints, axis=0) if all_joints else None
    return joints_all, verts_all


def frechet_distance(samples_A, samples_B):
    A_mu = np.mean(samples_A, axis=0)
    A_sigma = np.cov(samples_A, rowvar=False)
    B_mu = np.mean(samples_B, axis=0)
    B_sigma = np.cov(samples_B, rowvar=False)

    diff = A_mu - B_mu
    covmean, _ = linalg.sqrtm(A_sigma.dot(B_sigma), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(A_sigma.shape[0]) * eps
        covmean = linalg.sqrtm((A_sigma + offset).dot(B_sigma + offset)).real

    return diff.dot(diff) + np.trace(A_sigma) + np.trace(B_sigma) - 2 * np.trace(covmean)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MambaTalk v2 with PantoMatrix metrics')
    parser.add_argument('--result_dir', required=True, help='Directory with res_*.npz inference results')
    parser.add_argument('--data_dir', default='./data_new/', help='Data root (contains smplxflame_30/)')
    parser.add_argument('--audio_dir', default='./data_new/wave16k/', help='Audio wav directory')
    parser.add_argument('--pose_rep', default='smplxflame_30')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading SAM-3D-Body model...")
    model = load_mhr_model(device)

    res_files = sorted(glob.glob(os.path.join(args.result_dir, 'res_*.npz')))
    print(f"Found {len(res_files)} result files")

    gt_dir = os.path.join(args.data_dir, args.pose_rep)
    paired = []
    for res_f in res_files:
        basename = os.path.basename(res_f)
        name = basename.replace('res_', '').replace('.npz', '')
        gt_f = os.path.join(gt_dir, name + '.npz')
        audio_f = os.path.join(args.audio_dir, name + '.wav')
        if os.path.exists(gt_f):
            paired.append((gt_f, res_f, audio_f, name))
        else:
            print(f"  Warning: GT not found for {name}")

    print(f"Paired: {len(paired)} sequences")

    l1_calc = PantoL1div()
    bc_calc = PantoBC(download_path='/data1/yangzhuowei/PantoMatrix/emage_evaltools/', sigma=0.3, order=7)
    mse_calc = PantoMSEFace()
    lvd_calc = PantoLVDFace()
    gt_joints_flat_all = []
    pred_joints_flat_all = []
    total_frames = 0
    per_seq_results = []

    for gt_path, res_path, audio_path, spk_id in tqdm(paired, desc="Evaluating"):
        try:
            gt_data = dict(np.load(gt_path, allow_pickle=True))
            pred_data = dict(np.load(res_path, allow_pickle=True))
            gt_joints, gt_verts = mhr_forward_batch(model, gt_data, device)
            pred_joints, pred_verts = mhr_forward_batch(model, pred_data, device)
        except Exception as e:
            print(f"  Skip {spk_id}: {e}")
            continue

        if gt_joints is None or pred_joints is None:
            print(f"  Skip {spk_id}: mhr_forward did not return joint_coords")
            continue

        n = min(gt_joints.shape[0], pred_joints.shape[0])
        gt_j = gt_joints[:n, :55, :]
        pred_j = pred_joints[:n, :55, :]

        gt_flat = gt_j.reshape(n, -1)
        pred_flat = pred_j.reshape(n, -1)
        gt_joints_flat_all.append(gt_flat)
        pred_joints_flat_all.append(pred_flat)

        l1_calc.compute(pred_flat.copy())

        bc_score = None
        if os.path.exists(audio_path):
            align_mask = 60
            if n > 2 * align_mask:
                try:
                    audio_beat = bc_calc.load_audio(
                        audio_path, t_start=int(align_mask * 16000 / 30),
                        t_end=int((n - align_mask) / 30 * 16000))
                    motion_beat = bc_calc.load_motion(
                        pred_flat, t_start=align_mask, t_end=n - align_mask,
                        pose_fps=30, without_file=True)
                    bc_calc.compute(audio_beat, motion_beat, length=n - 2 * align_mask, pose_fps=30)
                except Exception as e:
                    print(f"  BC skip {spk_id}: {e}")

        n_face = min(gt_verts.shape[0], pred_verts.shape[0])
        gt_v_flat = gt_verts[:n_face].reshape(n_face, -1)
        pred_v_flat = pred_verts[:n_face].reshape(n_face, -1)
        mse_calc.compute(pred_v_flat, gt_v_flat)
        lvd_calc.compute(pred_v_flat, gt_v_flat)

        mpjpe = np.mean(np.sqrt(np.sum((gt_j - pred_j) ** 2, axis=-1)))
        per_seq_results.append({
            'name': spk_id,
            'frames': n,
            'mpjpe': mpjpe,
        })
        total_frames += n

    print("\n" + "=" * 60)
    print("MambaTalk v2 (PD-Enhanced) Evaluation Results")
    print("=" * 60)

    fgd_val = None
    if len(gt_joints_flat_all) > 0:
        gt_all = np.concatenate(gt_joints_flat_all, axis=0)
        pred_all = np.concatenate(pred_joints_flat_all, axis=0)
        fgd_val = frechet_distance(pred_all, gt_all)
        print(f"  FGD (Frechet Gesture Distance):  {fgd_val:.4f}")
    else:
        print(f"  FGD: N/A")

    l1div_val = l1_calc.avg()
    bc_val = bc_calc.avg()
    mse_val = mse_calc.avg()
    lvd_val = lvd_calc.avg()

    print(f"  L1Div (Diversity):               {l1div_val:.4f}")
    print(f"  BC (Beat Consistency):           {bc_val:.4f}")
    print(f"  MSEFace:                         {mse_val:.6f}")
    print(f"  LVDFace:                         {lvd_val:.6f}")

    if per_seq_results:
        avg_mpjpe = np.mean([r['mpjpe'] for r in per_seq_results])
        print(f"  MPJPE (Mean Per-Joint Pos Error): {avg_mpjpe:.4f}")

    print(f"\n  Total sequences: {len(paired)}")
    print(f"  Total frames: {total_frames}")
    print("=" * 60)

    out_file = os.path.join(args.result_dir, 'evaluation_metrics.txt')
    with open(out_file, 'w') as f:
        f.write("MambaTalk v2 (PD-Enhanced) Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Result dir: {args.result_dir}\n")
        f.write(f"Total sequences: {len(paired)}\n")
        f.write(f"Total frames: {total_frames}\n\n")
        f.write("Metrics:\n")
        f.write(f"  FGD:      {fgd_val:.4f}\n" if fgd_val else "  FGD:      N/A\n")
        f.write(f"  L1Div:    {l1div_val:.4f}\n")
        f.write(f"  BC:       {bc_val:.4f}\n")
        f.write(f"  MSEFace:  {mse_val:.6f}\n")
        f.write(f"  LVDFace:  {lvd_val:.6f}\n")
        if per_seq_results:
            f.write(f"  MPJPE:    {avg_mpjpe:.4f}\n")
        f.write("\nPer-sequence MPJPE:\n")
        for r in per_seq_results:
            f.write(f"  {r['name']}: {r['mpjpe']:.4f} ({r['frames']} frames)\n")
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
