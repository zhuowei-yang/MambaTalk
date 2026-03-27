#!/usr/bin/env python3
"""
Extract and analyze InteractionFusion sigma weights from inference results.

Produces:
  - Global average sigma distribution (sigma_self, sigma_partner, sigma_state)
  - Per-dialogue-state breakdown (idle/nonidle/backchannel/complete/incomplete)
  - Per-sequence summary table

Usage:
  python tools/extract_sigma_weights.py \
      --result_dir outputs/audio2pose/custom/0323_092535_phase1_interaction/100 \
      --state_dir ./data_new/dialogue_states/ \
      --output sigma_analysis.txt
"""

import os
import argparse
import glob
import numpy as np

STATE_NAMES = {
    0: "idle",
    1: "nonidle",
    2: "backchannel",
    3: "complete",
    4: "incomplete",
}
SIGMA_NAMES = ["sigma_self", "sigma_partner", "sigma_state"]


def load_states(state_dir, spk_id, n_frames, pose_fps=30):
    """Load and temporally align state sequence to n_frames at pose_fps."""
    state_path = os.path.join(state_dir, spk_id + '_states.npy')
    if not os.path.exists(state_path):
        return None
    states = np.load(state_path)
    if len(states) == 0:
        return None
    indices = np.linspace(0, len(states) - 1, n_frames).astype(int)
    return states[indices]


def analyze_result_dir(result_dir, state_dir):
    """Analyze all res_*.npz files in result_dir."""
    res_files = sorted(glob.glob(os.path.join(result_dir, 'res_*.npz')))

    global_sigmas = []
    per_state = {s: [] for s in range(5)}
    per_seq = []

    for res_path in res_files:
        data = np.load(res_path, allow_pickle=True)
        if 'sigma_weights' not in data.files:
            continue

        sigma = data['sigma_weights']  # (T, 3)
        if sigma.ndim != 2 or sigma.shape[1] != 3:
            continue

        basename = os.path.basename(res_path)
        spk_id = basename.replace('res_', '').replace('.npz', '')
        n_frames = sigma.shape[0]

        seq_mean = sigma.mean(axis=0)
        global_sigmas.append(seq_mean)
        per_seq.append({'name': spk_id, 'mean': seq_mean, 'frames': n_frames})

        if state_dir:
            states = load_states(state_dir, spk_id, n_frames)
            if states is not None:
                for s_id in range(5):
                    mask = (states == s_id)
                    if mask.sum() > 0:
                        per_state[s_id].append(sigma[mask].mean(axis=0))

    return global_sigmas, per_state, per_seq


def format_report(global_sigmas, per_state, per_seq, result_dir):
    """Format analysis results as text report."""
    lines = []
    lines.append("InteractionFusion Sigma Weight Analysis")
    lines.append("=" * 60)
    lines.append(f"Result dir: {result_dir}")
    lines.append(f"Sequences with sigma: {len(per_seq)}")
    lines.append("")

    if global_sigmas:
        gm = np.array(global_sigmas).mean(axis=0)
        lines.append("Global Average Sigma Weights:")
        for i, name in enumerate(SIGMA_NAMES):
            lines.append(f"  {name:20s}: {gm[i]:.4f}")
        lines.append("")

    lines.append("Per Dialogue-State Breakdown:")
    lines.append(f"  {'State':15s} | {'sigma_self':12s} | {'sigma_partner':14s} | {'sigma_state':12s} | {'N_seqs':6s}")
    lines.append("  " + "-" * 70)
    for s_id in range(5):
        vals = per_state[s_id]
        if vals:
            m = np.array(vals).mean(axis=0)
            lines.append(f"  {STATE_NAMES[s_id]:15s} | {m[0]:12.4f} | {m[1]:14.4f} | {m[2]:12.4f} | {len(vals):6d}")
        else:
            lines.append(f"  {STATE_NAMES[s_id]:15s} |          N/A |            N/A |          N/A |      0")
    lines.append("")

    lines.append("Per-Sequence Summary:")
    for r in per_seq:
        m = r['mean']
        lines.append(f"  {r['name']}: self={m[0]:.3f} partner={m[1]:.3f} "
                     f"state={m[2]:.3f} ({r['frames']}f)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze InteractionFusion sigma weights')
    parser.add_argument('--result_dir', required=True,
                        help='Directory with res_*.npz containing sigma_weights')
    parser.add_argument('--state_dir', default='./data_new/dialogue_states/',
                        help='Directory with *_states.npy files')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: {result_dir}/sigma_analysis.txt)')
    args = parser.parse_args()

    global_sigmas, per_state, per_seq = analyze_result_dir(args.result_dir, args.state_dir)

    if not per_seq:
        print(f"No sigma_weights found in {args.result_dir}")
        print("(Sigma weights are only saved when the model code includes sigma export support)")
        return

    report = format_report(global_sigmas, per_state, per_seq, args.result_dir)
    print(report)

    out_path = args.output or os.path.join(args.result_dir, 'sigma_analysis.txt')
    with open(out_path, 'w') as f:
        f.write(report + "\n")
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
