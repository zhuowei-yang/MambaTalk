#!/usr/bin/env python3
"""
Dual-person interaction quality evaluation.

Computes three interaction-specific metrics that evaluate how well
generated speaker2 gestures respond to speaker1's behavior:

  1. InteractionSync (IS):  Cross-correlation peak between motion
     energies of generated speaker2 and GT speaker1.
  2. TurnResponseScore (TRS):  Ratio of speaker2 motion energy
     after vs before dialogue-state change points.
  3. PartnerMotionCorr (PMC):  Pearson correlation of joint
     velocities between generated and GT speaker2.

Usage (in sam_3d_body conda env):
  CUDA_VISIBLE_DEVICES=1 python evaluate_interaction.py \
      --result_dir outputs/audio2pose/custom/0323_092535_phase1_interaction/40 \
      --data_dir ./data_new/ \
      --state_dir ./data_new/dialogue_states/ \
      --device cuda
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

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
    scale = npz_data.get('scale_params', np.zeros(28, dtype=np.float32))
    if scale.ndim == 2:
        scale = scale[0]

    all_joints = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start

        body_pose = torch.from_numpy(npz_data['body_pose_params'][start:end]).float().to(device)
        hand_pose = torch.from_numpy(npz_data['hand_pose_params'][start:end]).float().to(device)
        expr = npz_data.get('expr_params', np.zeros((N, 72), dtype=np.float32))
        expr_t = torch.from_numpy(expr[start:end]).float().to(device)
        g_rot = torch.from_numpy(npz_data['global_rot'][start:end]).float().to(device)
        shape_t = torch.from_numpy(shape).float().unsqueeze(0).expand(bs, -1).to(device)
        scale_t = torch.from_numpy(scale).float().unsqueeze(0).expand(bs, -1).to(device)
        g_trans = torch.zeros(bs, 3, dtype=torch.float32).to(device)

        with torch.no_grad():
            result = model.head_pose.mhr_forward(
                global_trans=g_trans, global_rot=g_rot,
                body_pose_params=body_pose, hand_pose_params=hand_pose,
                scale_params=scale_t, shape_params=shape_t, expr_params=expr_t,
                return_keypoints=False, return_joint_coords=True,
            )
            joint_coords = result[1] if isinstance(result, tuple) and len(result) > 1 else None

        if joint_coords is not None:
            all_joints.append(joint_coords.cpu().numpy())

    return np.concatenate(all_joints, axis=0) if all_joints else None


def interaction_sync(gen_joints2, gt_joints1, fps=30, max_lag_sec=2.0):
    """Cross-correlation peak between speaker2 generated and speaker1 GT motion energies."""
    max_lag = int(max_lag_sec * fps)
    vel1 = np.diff(gt_joints1, axis=0)
    vel2 = np.diff(gen_joints2, axis=0)
    energy1 = np.linalg.norm(vel1.reshape(vel1.shape[0], -1), axis=1)
    energy2 = np.linalg.norm(vel2.reshape(vel2.shape[0], -1), axis=1)

    n = min(len(energy1), len(energy2))
    if n < max_lag * 2:
        return 0.0, 0.0

    e1 = energy1[:n]
    e2 = energy2[:n]
    e1_norm = (e1 - e1.mean()) / (e1.std() + 1e-8)
    e2_norm = (e2 - e2.mean()) / (e2.std() + 1e-8)

    corr = np.correlate(e1_norm, e2_norm, mode='full')
    corr /= n
    mid = len(corr) // 2
    lo, hi = mid - max_lag, mid + max_lag
    peak_val = np.max(corr[lo:hi])
    peak_lag = (np.argmax(corr[lo:hi]) - max_lag) / fps
    return float(peak_val), float(peak_lag)


def turn_response_score(gen_joints2, state_seq, fps=30, window=15):
    """Motion energy ratio after/before dialogue-state change points."""
    vel = np.diff(gen_joints2, axis=0)
    energy = np.linalg.norm(vel.reshape(vel.shape[0], -1), axis=1)

    n = min(len(energy), len(state_seq) - 1)
    if n < window * 3:
        return 1.0

    change_pts = np.where(np.diff(state_seq[:n + 1]) != 0)[0]
    scores = []
    for cp in change_pts:
        if cp < window or cp + window >= n:
            continue
        before = energy[cp - window:cp].mean()
        after = energy[cp:cp + window].mean()
        scores.append(after / (before + 1e-8))

    return float(np.mean(scores)) if scores else 1.0


def partner_motion_corr(gen_joints2, gt_joints2, fps=30):
    """Pearson correlation of joint velocities between generated and GT speaker2."""
    n = min(gen_joints2.shape[0], gt_joints2.shape[0])
    if n < 10:
        return 0.0
    vel_gen = np.diff(gen_joints2[:n], axis=0).reshape(-1)
    vel_gt = np.diff(gt_joints2[:n], axis=0).reshape(-1)
    if vel_gen.std() < 1e-8 or vel_gt.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(vel_gen, vel_gt)[0, 1])


def main():
    parser = argparse.ArgumentParser(description='Evaluate dual-person interaction quality')
    parser.add_argument('--result_dir', required=True,
                        help='Directory with res_*_speaker1.npz inference results')
    parser.add_argument('--data_dir', default='./data_new/',
                        help='Data root (contains smplxflame_30/)')
    parser.add_argument('--state_dir', default='./data_new/dialogue_states/',
                        help='Directory with *_states.npy files')
    parser.add_argument('--pose_rep', default='smplxflame_30')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--joints', type=int, default=55,
                        help='Number of joints to use for metrics')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading SAM-3D-Body model...")
    model = load_mhr_model(device)

    res_files = sorted(glob.glob(os.path.join(args.result_dir, 'res_*.npz')))
    print(f"Found {len(res_files)} result files")

    gt_dir = os.path.join(args.data_dir, args.pose_rep)

    is_vals, trs_vals, pmc_vals = [], [], []
    peak_lags = []
    per_seq = []

    for res_path in tqdm(res_files, desc="Evaluating interaction"):
        basename = os.path.basename(res_path)
        spk1_id = basename.replace('res_', '').replace('.npz', '')
        if not spk1_id.endswith('speaker1'):
            continue
        spk2_id = spk1_id.rsplit('speaker1', 1)[0] + 'speaker2'

        gt_spk1_path = os.path.join(gt_dir, spk1_id + '.npz')
        gt_spk2_path = os.path.join(gt_dir, spk2_id + '.npz')
        state_path = os.path.join(args.state_dir, spk1_id + '_states.npy')

        if not os.path.exists(gt_spk1_path):
            print(f"  Skip {spk1_id}: GT speaker1 not found")
            continue
        if not os.path.exists(gt_spk2_path):
            print(f"  Skip {spk1_id}: GT speaker2 not found")
            continue

        try:
            pred_data = dict(np.load(res_path, allow_pickle=True))
            gt1_data = dict(np.load(gt_spk1_path, allow_pickle=True))
            gt2_data = dict(np.load(gt_spk2_path, allow_pickle=True))

            pred_joints = mhr_forward_batch(model, pred_data, device)
            gt1_joints = mhr_forward_batch(model, gt1_data, device)
            gt2_joints = mhr_forward_batch(model, gt2_data, device)
        except Exception as e:
            print(f"  Skip {spk1_id}: {e}")
            continue

        if pred_joints is None or gt1_joints is None or gt2_joints is None:
            print(f"  Skip {spk1_id}: mhr_forward returned None")
            continue

        j = args.joints
        n = min(pred_joints.shape[0], gt1_joints.shape[0], gt2_joints.shape[0])
        pred_j = pred_joints[:n, :j, :]
        gt1_j = gt1_joints[:n, :j, :]
        gt2_j = gt2_joints[:n, :j, :]

        is_val, lag = interaction_sync(pred_j, gt1_j)
        pmc_val = partner_motion_corr(pred_j, gt2_j)

        trs_val = 1.0
        if os.path.exists(state_path):
            states = np.load(state_path)
            trs_val = turn_response_score(pred_j, states)

        is_vals.append(is_val)
        peak_lags.append(lag)
        trs_vals.append(trs_val)
        pmc_vals.append(pmc_val)
        per_seq.append({
            'name': spk1_id, 'IS': is_val, 'PeakLag': lag,
            'TRS': trs_val, 'PMC': pmc_val, 'frames': n,
        })

    print("\n" + "=" * 60)
    print("Dual-Person Interaction Evaluation Results")
    print("=" * 60)

    avg_is = np.mean(is_vals) if is_vals else 0.0
    avg_lag = np.mean(peak_lags) if peak_lags else 0.0
    avg_trs = np.mean(trs_vals) if trs_vals else 1.0
    avg_pmc = np.mean(pmc_vals) if pmc_vals else 0.0

    print(f"  InteractionSync (IS):       {avg_is:.4f}")
    print(f"  PeakLag (seconds):          {avg_lag:.4f}")
    print(f"  TurnResponseScore (TRS):    {avg_trs:.4f}")
    print(f"  PartnerMotionCorr (PMC):    {avg_pmc:.4f}")
    print(f"\n  Total sequences evaluated:  {len(per_seq)}")
    print("=" * 60)

    out_file = os.path.join(args.result_dir, 'interaction_metrics.txt')
    with open(out_file, 'w') as f:
        f.write("Dual-Person Interaction Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Result dir: {args.result_dir}\n")
        f.write(f"Total sequences: {len(per_seq)}\n\n")
        f.write("Metrics:\n")
        f.write(f"  IS:       {avg_is:.4f}\n")
        f.write(f"  PeakLag:  {avg_lag:.4f}\n")
        f.write(f"  TRS:      {avg_trs:.4f}\n")
        f.write(f"  PMC:      {avg_pmc:.4f}\n")
        f.write("\nPer-sequence:\n")
        for r in per_seq:
            f.write(f"  {r['name']}: IS={r['IS']:.4f} Lag={r['PeakLag']:.3f}s "
                    f"TRS={r['TRS']:.4f} PMC={r['PMC']:.4f} ({r['frames']}f)\n")
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
