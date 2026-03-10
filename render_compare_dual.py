"""
Render GT vs MambaTalk_new inference for dual-speaker conversation.
Output: side-by-side video, left=GT, right=Predicted, for both speakers.
Usage: conda activate sam_3d_body && python render_compare_dual.py --sample 47HlvCfubE4_0018 --epoch 140 --gpu 6
"""
import os, sys, argparse, subprocess, glob
import numpy as np

RENDER_SCRIPT = "/data1/yangzhuowei/Gesture/data_process/render_mesh_video_from_npz.py"

def fix_npz_for_render(src_path, dst_path, ref_gt_path=None):
    d = np.load(src_path, allow_pickle=True)
    n = d['body_pose_params'].shape[0]

    fl_val = 1500.0
    if 'focal_length' in d.files:
        fl = d['focal_length']
        fl_val = float(fl) if fl.ndim == 0 else float(fl[0])

    sp = d['scale_params']
    if sp.ndim == 1:
        sp = np.repeat(sp.reshape(1, -1), n, axis=0)
    elif sp.shape[0] != n:
        sp = np.repeat(sp[:1], n, axis=0)

    shp = d['shape_params']
    if shp.ndim == 1:
        shp = np.repeat(shp.reshape(1, -1), n, axis=0)
    elif shp.shape[0] != n:
        shp = np.repeat(shp[:1], n, axis=0)

    w = d['width'] if 'width' in d.files else np.array([1080])
    h = d['height'] if 'height' in d.files else np.array([1920])

    np.savez(dst_path,
             body_pose_params=d['body_pose_params'][:n],
             hand_pose_params=d['hand_pose_params'][:n],
             expr_params=d['expr_params'][:n] if 'expr_params' in d.files else np.zeros((n, 72), dtype=np.float32),
             global_rot=d['global_rot'][:n],
             pred_cam_t=d['pred_cam_t'][:n],
             shape_params=shp[:n],
             scale_params=sp[:n],
             focal_length=np.full(n, fl_val, dtype=np.float32),
             width=w, height=h,
             frame_idx=np.arange(n, dtype=np.int32),
             fps=np.array([30.0], dtype=np.float32))


def make_gt_npz(gt_src, dst_path, n_frames):
    gt = np.load(gt_src, allow_pickle=True)
    n = min(n_frames, gt['body_pose_params'].shape[0])

    fl_val = float(gt['focal_length'][0]) if 'focal_length' in gt.files else 1500.0
    sp = gt['scale_params']
    sp = np.repeat(sp[:1], n, axis=0) if sp.ndim == 2 else np.repeat(sp.reshape(1, -1), n, axis=0)
    shp = gt['shape_params']
    shp = np.repeat(shp[:1], n, axis=0) if shp.ndim == 2 else np.repeat(shp.reshape(1, -1), n, axis=0)

    np.savez(dst_path,
             body_pose_params=gt['body_pose_params'][:n],
             hand_pose_params=gt['hand_pose_params'][:n],
             expr_params=gt['expr_params'][:n] if 'expr_params' in gt.files else np.zeros((n, 72), dtype=np.float32),
             global_rot=gt['global_rot'][:n],
             pred_cam_t=gt['pred_cam_t'][:n],
             shape_params=shp[:n], scale_params=sp[:n],
             focal_length=np.full(n, fl_val, dtype=np.float32),
             width=gt['width'] if 'width' in gt.files else np.array([1080]),
             height=gt['height'] if 'height' in gt.files else np.array([1920]),
             frame_idx=np.arange(n, dtype=np.int32),
             fps=np.array([30.0], dtype=np.float32))


def render_npz(npz_path, gpu_id=6):
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} xvfb-run -a python {RENDER_SCRIPT} --npz_path {npz_path} --mode mesh"
    print(f"    Rendering {os.path.basename(npz_path)}...")
    subprocess.run(cmd, shell=True, cwd="/data1/yangzhuowei/Gesture/data_process")
    video = npz_path.replace('.npz', '_mesh.mp4')
    return video if os.path.exists(video) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='47HlvCfubE4_0018')
    parser.add_argument('--epoch', type=int, default=140)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--out_root', type=str, default='/data1/yangzhuowei/MambaTalk_new/outputs/audio2pose/custom/0306_160640_mambatalk_mhr_new')
    parser.add_argument('--data_dir', type=str, default='/data1/yangzhuowei/MambaTalk_new/data_new/smplxflame_30')
    args = parser.parse_args()

    render_dir = os.path.join(args.out_root, 'render_compare')
    os.makedirs(render_dir, exist_ok=True)

    for speaker in ['speaker1', 'speaker2']:
        full_name = f"{args.sample}_{speaker}"
        print(f"\n{'='*60}")
        print(f"Processing: {full_name}")
        print(f"{'='*60}")

        res_npz = os.path.join(args.out_root, str(args.epoch), f"res_{full_name}.npz")
        gt_src = os.path.join(args.data_dir, f"{full_name}.npz")

        if not os.path.exists(res_npz):
            print(f"  SKIP: {res_npz} not found")
            continue
        if not os.path.exists(gt_src):
            print(f"  SKIP: GT {gt_src} not found")
            continue

        res_data = np.load(res_npz, allow_pickle=True)
        n_frames = res_data['body_pose_params'].shape[0]
        print(f"  Frames: {n_frames} (~{n_frames/30:.1f}s)")

        gt_render = os.path.join(render_dir, f"gt_{full_name}.npz")
        rec_render = os.path.join(render_dir, f"rec_{full_name}.npz")

        make_gt_npz(gt_src, gt_render, n_frames)
        fix_npz_for_render(res_npz, rec_render)

        print("  Rendering GT...")
        gt_video = render_npz(gt_render, args.gpu)
        print("  Rendering Prediction...")
        rec_video = render_npz(rec_render, args.gpu)

        if gt_video and rec_video:
            output = os.path.join(render_dir, f"compare_{full_name}_ep{args.epoch}.mp4")
            print("  Creating side-by-side (Left=GT, Right=Predicted)...")
            cmd = (f'ffmpeg -y -i {gt_video} -i {rec_video} '
                   f'-filter_complex "[0:v]pad=iw*2:ih[bg];[bg][1:v]overlay=w" '
                   f'-c:v libx264 -crf 23 {output}')
            subprocess.run(cmd, shell=True, capture_output=True)
            if os.path.exists(output):
                size_mb = os.path.getsize(output) / 1024 / 1024
                print(f"  Done! {output} ({size_mb:.1f}MB)")
            else:
                print(f"  ffmpeg failed. Individual videos: {gt_video} | {rec_video}")
        else:
            print(f"  Rendering failed for {speaker}!")

    print(f"\nAll outputs in: {render_dir}")


if __name__ == "__main__":
    main()
