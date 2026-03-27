#!/usr/bin/env python3
"""
MambaTalk render (SAM-3D-Body MHR)

渲染MambaTalk输出的npz为人体Mesh视频。
npz已包含全部渲染所需参数（body/hand/expr + global_rot/cam_t + scale_params/focal_length）。
对于新数据（无原始数据时），直接使用模型预测值渲染。

Usage (sam_3d_body conda env):
    conda activate sam_3d_body
    export PYOPENGL_PLATFORM=osmesa
    xvfb-run -a python render.py --npy_path <result.npz> --wav_path <audio.wav> --save_dir <dir>
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, '/data1/yangzhuowei/sam_3d_body')
import pyrootutils
root = pyrootutils.setup_root(
    search_from='/data1/yangzhuowei/sam_3d_body',
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True, dotenv=True,
)
from sam_3d_body import load_sam_3d_body
from sam_3d_body.visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def load_model(checkpoint_path, mhr_path, device):
    print("Loading SAM-3D-Body model...")
    model, _ = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    faces = model.head_pose.faces.cpu().numpy()
    return model, faces


def reconstruct_vertices(model, outputs, device):
    body_pose = torch.tensor(outputs['body_pose_params'], dtype=torch.float32).unsqueeze(0).to(device)
    hand_pose = torch.tensor(outputs['hand_pose_params'], dtype=torch.float32).unsqueeze(0).to(device)
    shape = torch.tensor(outputs['shape_params'], dtype=torch.float32).unsqueeze(0).to(device)
    scale = torch.tensor(outputs['scale_params'], dtype=torch.float32).unsqueeze(0).to(device)
    face = torch.tensor(outputs['expr_params'], dtype=torch.float32).unsqueeze(0).to(device)
    global_rot = torch.tensor(outputs['global_rot'], dtype=torch.float32).unsqueeze(0).to(device)
    global_trans = torch.zeros(1, 3, dtype=torch.float32).to(device)

    with torch.no_grad():
        verts = model.head_pose.mhr_forward(
            global_trans=global_trans, global_rot=global_rot,
            body_pose_params=body_pose, hand_pose_params=hand_pose,
            scale_params=scale, shape_params=shape, expr_params=face,
            return_keypoints=False, return_joint_coords=False,
        )
        if isinstance(verts, tuple):
            verts = verts[0]
        verts[..., [1, 2]] *= -1
    return verts.cpu().numpy()[0]


def load_npz(npz_path):
    """Load npz and build per-frame param dicts. Works for both MambaTalk output and raw SAM-3D-Body."""
    data = np.load(npz_path, allow_pickle=True)
    N = len(data['body_pose_params'])

    # Resolution and focal_length
    ow = int(data['width'][0]) if 'width' in data.files else 1080
    oh = int(data['height'][0]) if 'height' in data.files else 1920
    render_size = min(ow, oh, 540)
    scale_factor = render_size / max(ow, oh)
    base_fl = float(data['focal_length'][0]) * scale_factor if 'focal_length' in data.files else 500.0

    # Per-video constants
    if 'scale_params' in data.files:
        sp = data['scale_params']
        scale_shared = sp[0].astype(np.float32) if sp.ndim == 2 else sp.astype(np.float32)
    else:
        scale_shared = np.zeros(28, np.float32)

    if 'shape_params' in data.files:
        shp = data['shape_params']
        shape_shared = shp[0].astype(np.float32) if shp.ndim == 2 else shp.astype(np.float32)
    else:
        shape_shared = np.zeros(45, np.float32)

    frames = []
    for i in range(N):
        f = {}
        f['body_pose_params'] = data['body_pose_params'][i].astype(np.float32)
        f['hand_pose_params'] = data['hand_pose_params'][i].astype(np.float32) if 'hand_pose_params' in data.files else np.zeros(108, np.float32)
        f['expr_params'] = data['expr_params'][i].astype(np.float32) if 'expr_params' in data.files else np.zeros(72, np.float32)
        f['global_rot'] = data['global_rot'][i].astype(np.float32) if 'global_rot' in data.files else np.zeros(3, np.float32)
        f['pred_cam_t'] = data['pred_cam_t'][i].astype(np.float32) if 'pred_cam_t' in data.files else np.zeros(3, np.float32)
        f['shape_params'] = shape_shared
        f['scale_params'] = scale_shared.copy()

        if 'focal_length' in data.files and data['focal_length'].shape[0] > 1:
            f['focal_length'] = float(data['focal_length'][i]) * scale_factor
        else:
            f['focal_length'] = base_fl
        frames.append(f)

    fps = float(data['mocap_frame_rate']) if 'mocap_frame_rate' in data.files else (float(data['fps'][0]) if 'fps' in data.files else 30.0)
    data.close()
    return frames, fps, render_size, render_size


def render_sequence(
    npz_path, output_dir, audio_path=None,
    checkpoint_path="/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt",
    mhr_path="/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, faces = load_model(checkpoint_path, mhr_path, device)

    print(f"Loading: {npz_path}")
    frames, fps, width, height = load_npz(npz_path)
    N = len(frames)
    print(f"Frames: {N}, FPS: {fps}, Size: {width}x{height}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    video_path = os.path.join(output_dir, f"{base_name}.mp4")
    temp_path = os.path.join(output_dir, f"{base_name}_temp.avi")

    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width * 2, height))

    for i in tqdm(range(N), desc="Rendering"):
        try:
            renderer = Renderer(focal_length=frames[i]['focal_length'], faces=faces)
            verts = reconstruct_vertices(model, frames[i], device)
            cam_t = frames[i]['pred_cam_t']
            img_front = (renderer(verts, cam_t, white_bg.copy(),
                         mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1)) * 255).astype(np.uint8)
            img_side = (renderer(verts, cam_t, white_bg.copy(),
                        mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), side_view=True) * 255).astype(np.uint8)
        except Exception as e:
            print(f"Frame {i} failed: {e}")
            img_front = white_bg.copy()
            img_side = white_bg.copy()
        out.write(np.concatenate([img_front, img_side], axis=1))

    out.release()

    if audio_path and os.path.exists(audio_path):
        cmd = ['ffmpeg', '-y', '-i', temp_path, '-i', audio_path,
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-c:a', 'aac', '-b:a', '128k', '-pix_fmt', 'yuv420p',
               '-shortest', video_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', temp_path,
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-pix_fmt', 'yuv420p', video_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_path)
        print(f"Saved: {video_path}")
    except subprocess.CalledProcessError:
        print(f"ffmpeg failed, temp: {temp_path}")
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", required=True)
    parser.add_argument("--wav_path", default="")
    parser.add_argument("--save_dir", default="outputs/render")
    parser.add_argument("--checkpoint_path",
        default="/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path",
        default="/data1/yangzhuowei/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    render_sequence(args.npy_path, args.save_dir,
                    audio_path=args.wav_path if args.wav_path else None,
                    checkpoint_path=args.checkpoint_path, mhr_path=args.mhr_path)
