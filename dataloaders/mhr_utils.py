# Extracted from sam_3d_body/models/modules/mhr_utils.py
# Pure PyTorch functions for MHR continuous <-> Euler conversion

import torch
import torch.nn.functional as F


def batch6DFromXYZ(r, return_9D=False):
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx, cy, cz = rc[..., 0], rc[..., 1], rc[..., 2]
    sx, sy, sz = rs[..., 0], rs[..., 1], rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype).to(r.device)
    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result


def batchXYZfrom6D(poses):
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)

    sy = torch.sqrt(
        matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0]
    )
    singular = (sy < 1e-6).float()

    x_angle = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y_angle = torch.atan2(-matrix[..., 2, 0], sy)
    z_angle = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0

    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x_angle * (1 - singular) + xs * singular
    out_euler[..., 1] = y_angle * (1 - singular) + ys * singular
    out_euler[..., 2] = z_angle * (1 - singular) + zs * singular

    return out_euler


# fmt: off
_ALL_3DOF_IDXS = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
_ALL_1DOF_IDXS = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
_ALL_TRANS_IDXS = torch.LongTensor([124, 125, 126, 127, 128, 129])
# fmt: on

_NUM_3DOF = len(_ALL_3DOF_IDXS) * 3   # 69
_NUM_1DOF = len(_ALL_1DOF_IDXS)        # 58
_NUM_TRANS = len(_ALL_TRANS_IDXS)       # 6
CONT_DIM = 2 * _NUM_3DOF + 2 * _NUM_1DOF + _NUM_TRANS  # 260


def compact_model_params_to_cont_body(body_pose_params):
    """Convert 133d Euler body params -> 260d continuous (6D rot + sin/cos)."""
    assert body_pose_params.shape[-1] == 133
    params_3dof = body_pose_params[..., _ALL_3DOF_IDXS.flatten()]
    params_1dof = body_pose_params[..., _ALL_1DOF_IDXS]
    params_trans = body_pose_params[..., _ALL_TRANS_IDXS]

    cont_3dof = batch6DFromXYZ(params_3dof.unflatten(-1, (-1, 3))).flatten(-2, -1)
    cont_1dof = torch.stack([params_1dof.sin(), params_1dof.cos()], dim=-1).flatten(-2, -1)

    return torch.cat([cont_3dof, cont_1dof, params_trans], dim=-1)


def compact_cont_to_model_params_body(body_pose_cont):
    """Convert 260d continuous -> 133d Euler body params."""
    assert body_pose_cont.shape[-1] == CONT_DIM
    cont_3dof = body_pose_cont[..., :2 * _NUM_3DOF]
    cont_1dof = body_pose_cont[..., 2 * _NUM_3DOF:2 * _NUM_3DOF + 2 * _NUM_1DOF]
    cont_trans = body_pose_cont[..., 2 * _NUM_3DOF + 2 * _NUM_1DOF:]

    params_3dof = batchXYZfrom6D(cont_3dof.unflatten(-1, (-1, 6))).flatten(-2, -1)
    params_1dof = torch.atan2(cont_1dof.unflatten(-1, (-1, 2))[..., -2],
                              cont_1dof.unflatten(-1, (-1, 2))[..., -1])

    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133, device=body_pose_cont.device, dtype=body_pose_cont.dtype)
    body_pose_params[..., _ALL_3DOF_IDXS.flatten()] = params_3dof
    body_pose_params[..., _ALL_1DOF_IDXS] = params_1dof
    body_pose_params[..., _ALL_TRANS_IDXS] = cont_trans
    return body_pose_params


def euler3_to_rot6d(euler):
    """Convert (*, 3) Euler XYZ -> (*, 6) rotation 6D."""
    return batch6DFromXYZ(euler.unsqueeze(-2)).squeeze(-2)


def rot6d_to_euler3(rot6d):
    """Convert (*, 6) rotation 6D -> (*, 3) Euler XYZ."""
    return batchXYZfrom6D(rot6d.unsqueeze(-2)).squeeze(-2)
