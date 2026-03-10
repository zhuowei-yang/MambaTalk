#!/usr/bin/env python3
"""
简单的 MambaTalk 渲染脚本
使用 SMPLX 模型将生成的动作序列渲染为视频

【重要】为何 GT 视频会扭曲？
- 测试阶段保存的 gt_xxx.npz 里的 poses 来自 SAM-3D-Body (MHR)：
  是 127 关节中前 55 个的「全局旋转」转成的 axis-angle。
- SMPL-X 期望的是「根节点全局 + 身体局部旋转」且关节顺序与 MHR 不同。
- 用 SMPLX 渲染这类数据 = 把“全局旋转 + 错误关节顺序”当 SMPL-X 用 → 姿态扭曲。

【正确渲染 GT】请用「原始」SAM-3D-Body npz + MHR 渲染，例如：
  python render.py --npy_path ./data/smplxflame_30/1_custom_0_1_1.npz --wav_path ./data/wave16k/1_custom_0_1_1.wav --save_dir ./outputs/gt_render/
（需配置 render.py 中的 checkpoint_path 与 mhr_path）
"""

import os
import sys
import argparse
import numpy as np 
import torch
import cv2
from tqdm import tqdm

import smplx
import pyrender
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def create_pose_camera(angle_deg=-2):
    """创建相机姿态矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def render_frame(renderer, vertices, faces):
    """渲染单帧"""
    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(np.deg2rad(-30)), -np.sin(np.deg2rad(-30)), 0.0],
        [0.0, np.sin(np.deg2rad(-30)), np.cos(np.deg2rad(-30)), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=uniform_color)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
    
    scene = pyrender.Scene()
    scene.add(mesh)
    
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    
    color, _ = renderer.render(scene)
    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True, help="推理结果 NPZ 文件路径")
    parser.add_argument("--wav_path", type=str, default=None, help="音频文件路径")
    parser.add_argument("--save_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--smplx_path", type=str, default="./pretrained/smplx_models/", help="SMPLX 模型路径")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 SMPLX 模型
    print("加载 SMPLX 模型...")
    smplx_model = smplx.create(
        args.smplx_path, 
        model_type='smplx',
        gender='NEUTRAL_2020', 
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100, 
        ext='npz',
        use_pca=False,
    ).to(device).eval()
    faces = smplx_model.faces
    
    # 加载 NPZ 数据
    print(f"加载数据: {args.npy_path}")
    data = np.load(args.npy_path)
    if "poses" in data.files and "pred_global_rots" not in data.files:
        print("\n[注意] 当前 npz 为训练/测试用格式（由 MHR 转成的 axis-angle）。")
        print("      用 SMPLX 渲染可能导致 GT 视频扭曲；正确 GT 请用「原始」npz 配合 render.py (MHR) 渲染。\n")
    poses = data['poses']  # (N, 165)
    expressions = data['expressions']  # (N, 100)
    betas = data['betas']  # (300,)
    trans = data['trans']  # (N, 3)
    
    num_frames = poses.shape[0]
    print(f"帧数: {num_frames}")
    
    # 转换为 tensor
    poses_tensor = torch.tensor(poses, dtype=torch.float32).to(device)
    expressions_tensor = torch.tensor(expressions, dtype=torch.float32).to(device)
    betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(num_frames, -1).to(device)
    trans_tensor = torch.tensor(trans, dtype=torch.float32).to(device)
    
    # 计算顶点
    print("计算 SMPLX 顶点...")
    with torch.no_grad():
        output = smplx_model(
            betas=betas_tensor,
            transl=trans_tensor,
            expression=expressions_tensor,
            jaw_pose=poses_tensor[:, 66:69],
            global_orient=poses_tensor[:, :3],
            body_pose=poses_tensor[:, 3:66],
            left_hand_pose=poses_tensor[:, 75:120],
            right_hand_pose=poses_tensor[:, 120:165],
            leye_pose=poses_tensor[:, 69:72],
            reye_pose=poses_tensor[:, 72:75],
            return_verts=True,
        )
    vertices = output.vertices.cpu().numpy()  # (N, V, 3)
    print(f"顶点形状: {vertices.shape}")
    
    # 渲染
    print("开始渲染...")
    renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=720)
    
    frames = []
    for i in tqdm(range(num_frames), desc="渲染帧"):
        frame = render_frame(renderer, vertices[i], faces)
        frames.append(frame)
    
    renderer.delete()
    
    # 保存视频
    os.makedirs(args.save_dir, exist_ok=True)
    video_name = os.path.basename(args.npy_path).replace('.npz', '.mp4')
    video_path = os.path.join(args.save_dir, video_name)
    
    print(f"保存视频: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (720, 720))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    
    # 添加音频
    if args.wav_path and os.path.exists(args.wav_path):
        print(f"添加音频: {args.wav_path}")
        final_video_path = video_path.replace('.mp4', '_audio.mp4')
        os.system(f'ffmpeg -y -i {video_path} -i {args.wav_path} -c:v copy -c:a aac -shortest {final_video_path} -loglevel error')
        if os.path.exists(final_video_path):
            os.remove(video_path)
            os.rename(final_video_path, video_path)
            print(f"最终视频: {video_path}")
    
    print("渲染完成!")


if __name__ == "__main__":
    main()
