"""
Render GT vs VAE reconstruction side-by-side comparison video.
Left = GT (original), Right = VAE reconstructed (Body + Hand).

Two-stage pipeline:
  Stage 1 (mambatalk env): VAE inference -> save GT & reconstructed npz
  Stage 2 (sam_3d_body env): render npz -> mesh mp4 -> ffmpeg concat

Usage:
  /data1/yangzhuowei/miniconda3/envs/mambatalk/bin/python render_vae_compare.py \
      --gpu 4 --n_samples 3 [--sample_name XXX]
"""
import os, sys, argparse, random, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders.mhr_utils import (
    compact_model_params_to_cont_body,
    compact_cont_to_model_params_body,
)

DATA_DIR = "./data/smplxflame_30"
OUT_DIR = "./outputs/vae_compare"

BODY_CKPT = "./outputs/audio2pose/custom/0310_121753_mhr_cont_body_res/rec.bin"
HAND_CKPT = "./outputs/audio2pose/custom/0310_121754_mhr_cont_hand_res/rec.bin"

RENDER_SCRIPT = "/data1/yangzhuowei/Gesture/data_process/render_mesh_video_from_npz.py"
SAM_PYTHON = "/data1/yangzhuowei/miniconda3/envs/sam_3d_body/bin/python"

BODY_CONT_DIM = 260
HAND_DIM = 108


def load_vae(ckpt_path, vae_test_dim, vae_length=256, vae_codebook_size=256, vae_layer=2, model_class="VQVAEConvZeroRes", **kwargs):
    from types import SimpleNamespace
    args = SimpleNamespace(
        vae_test_dim=vae_test_dim, vae_length=vae_length,
        vae_codebook_size=vae_codebook_size, vae_layer=vae_layer,
        vae_grow=[1, 1, 2, 1], variational=False,
        vae_quantizer_lambda=1.0, res_scale=1.0,
    )
    mod = __import__("models.motion_representation", fromlist=[model_class])
    model = getattr(mod, model_class)(args).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def vae_reconstruct(sample_path, body_vae, hand_vae, max_frames=600):
    gt = np.load(sample_path, allow_pickle=True)
    body_euler = gt['body_pose_params'].astype(np.float32)
    hand_pca = gt['hand_pose_params'].astype(np.float32)
    n = min(body_euler.shape[0], max_frames)
    body_euler = body_euler[:n]
    hand_pca = hand_pca[:n]

    body_cont = compact_model_params_to_cont_body(torch.from_numpy(body_euler).float())
    with torch.no_grad():
        rec_body_cont = body_vae(body_cont.unsqueeze(0).cuda())["rec_pose"].squeeze(0).cpu()
    rec_body_euler = compact_cont_to_model_params_body(rec_body_cont).numpy()

    with torch.no_grad():
        rec_hand = hand_vae(torch.from_numpy(hand_pca).float().unsqueeze(0).cuda()
                            )["rec_pose"].squeeze(0).cpu().numpy()

    return gt, n, rec_body_euler, rec_hand


def save_mhr_npz(gt, n, body_euler, hand_pca, out_path):
    """Save in MHR npz format compatible with render_mesh_video_from_npz.py."""
    fl = float(gt['focal_length'][0]) if 'focal_length' in gt.files else 1500.0
    sp = gt['scale_params']
    sp = np.repeat(sp[:1], n, axis=0) if sp.ndim == 2 else np.repeat(sp.reshape(1, -1), n, axis=0)
    shp = gt['shape_params']
    shp = np.repeat(shp[:1], n, axis=0) if shp.ndim == 2 else np.repeat(shp.reshape(1, -1), n, axis=0)
    expr = gt['expr_params'][:n] if 'expr_params' in gt.files else np.zeros((n, 72), dtype=np.float32)

    np.savez(out_path,
             body_pose_params=body_euler[:n],
             hand_pose_params=hand_pca[:n],
             expr_params=expr,
             global_rot=gt['global_rot'][:n],
             pred_cam_t=gt['pred_cam_t'][:n],
             shape_params=shp[:n],
             scale_params=sp[:n],
             focal_length=np.full(n, fl, dtype=np.float32),
             width=gt['width'] if 'width' in gt.files else np.array([1080]),
             height=gt['height'] if 'height' in gt.files else np.array([1920]),
             frame_idx=np.arange(n, dtype=np.int32),
             fps=np.array([30.0], dtype=np.float32))


def render_npz(npz_path, gpu):
    """Render MHR npz using sam_3d_body env + render_mesh_video_from_npz.py."""
    abs_npz = os.path.abspath(npz_path)
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu} "
           f"PYOPENGL_PLATFORM=osmesa "
           f"xvfb-run -a {SAM_PYTHON} {RENDER_SCRIPT} "
           f"--npz_path {abs_npz} --mode mesh")
    print(f"    cmd: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True,
                            cwd="/data1/yangzhuowei/Gesture/data_process",
                            stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"    RENDER FAILED (exit={result.returncode})")
    mp4 = abs_npz.replace('.npz', '_mesh.mp4')
    return mp4 if os.path.exists(mp4) else None


def concat_lr(left, right, out):
    """Crop mesh-only part (right half) from each video, then put side by side."""
    cmd = (f'ffmpeg -y -i {left} -i {right} '
           f'-filter_complex "'
           f'[0:v]crop=iw/2:ih:iw/2:0[gt];'
           f'[1:v]crop=iw/2:ih:iw/2:0[rec];'
           f'[gt][rec]hstack=inputs=2[out]" '
           f'-map "[out]" -c:v libx264 -crf 23 {out}')
    subprocess.run(cmd, shell=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--sample_name', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Max frames per sample (300 = 10s @ 30fps)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--body_ckpt', type=str, default=BODY_CKPT)
    parser.add_argument('--hand_ckpt', type=str, default=HAND_CKPT)
    parser.add_argument('--body_codebook_size', type=int, default=256,
                        help='Body VAE codebook size (512 for MambaTalk_new_512)')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override DATA_DIR (e.g. ./data_new/smplxflame_30)')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip VAE inference, only render existing npz')
    args = parser.parse_args()

    out_dir = args.out_dir
    data_dir = (args.data_dir if args.data_dir else DATA_DIR).rstrip("/")
    if not data_dir.endswith("smplxflame_30"):
        data_dir = os.path.join(data_dir, "smplxflame_30") if os.path.isdir(data_dir) else data_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Stage 1: VAE inference (mambatalk env) ──
    if not args.skip_inference:
        print("=" * 60)
        print("Stage 1: VAE Inference")
        print("=" * 60)
        print(f"Body VAE: {args.body_ckpt} (codebook={args.body_codebook_size})")
        print(f"Hand VAE: {args.hand_ckpt}")

        body_vae = load_vae(args.body_ckpt, vae_test_dim=BODY_CONT_DIM, vae_codebook_size=args.body_codebook_size)
        hand_vae = load_vae(args.hand_ckpt, vae_test_dim=HAND_DIM)
        print("VAE models loaded.\n")

    if args.sample_name:
        samples = [s.strip() for s in args.sample_name.split(",") if s.strip()]
    else:
        import pandas as pd
        split_path = "./data/train_test_split.csv" if os.path.exists("./data/train_test_split.csv") else "./data_new/train_test_split.csv"
        split = pd.read_csv(split_path)
        test_ids = split.loc[split['type'] == 'test', 'id'].tolist()
        valid = [s for s in test_ids
                 if os.path.exists(os.path.join(data_dir, s + ".npz"))]
        random.seed(args.seed)
        samples = random.sample(valid, min(args.n_samples, len(valid)))

    print(f"Samples ({len(samples)}): {samples}\n")

    for i, name in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {name}")
        sample_path = os.path.join(data_dir, name + ".npz")
        gt_npz = os.path.join(out_dir, f"gt_{name}.npz")
        rec_npz = os.path.join(out_dir, f"rec_{name}.npz")

        # Stage 1: inference
        if not args.skip_inference:
            gt, n, rec_body, rec_hand = vae_reconstruct(
                sample_path, body_vae, hand_vae, args.max_frames)
            print(f"  Frames={n}, body_rec={rec_body.shape}, hand_rec={rec_hand.shape}")

            save_mhr_npz(gt, n, gt['body_pose_params'], gt['hand_pose_params'], gt_npz)
            save_mhr_npz(gt, n, rec_body, rec_hand, rec_npz)
            print(f"  Saved: {gt_npz}")
            print(f"  Saved: {rec_npz}")

        # Stage 2: render with sam_3d_body
        print(f"  Rendering GT...")
        gt_mp4 = render_npz(gt_npz, args.gpu)
        print(f"  Rendering VAE reconstruction...")
        rec_mp4 = render_npz(rec_npz, args.gpu)

        if gt_mp4 and rec_mp4:
            compare_mp4 = os.path.join(out_dir, f"compare_{name}.mp4")
            print(f"  Creating side-by-side (Left=GT | Right=VAE)...")
            concat_lr(gt_mp4, rec_mp4, compare_mp4)
            if os.path.exists(compare_mp4):
                size_mb = os.path.getsize(compare_mp4) / 1024 / 1024
                print(f"  -> {compare_mp4} ({size_mb:.1f}MB)")
            else:
                print(f"  ffmpeg concat failed. Individual: {gt_mp4} | {rec_mp4}")
        else:
            print(f"  Render failed: gt_mp4={gt_mp4}, rec_mp4={rec_mp4}")
        print()

    print("=" * 60)
    print(f"All done! Videos in: {out_dir}/")
    print(f"  compare_*.mp4 = Left(GT) vs Right(VAE reconstruction)")
    print("=" * 60)


if __name__ == "__main__":
    main()
