"""
Standalone pre-training script for the FreqPD (FMLP + PD) composite module.

Same data pipeline as pd_trainer.py (joint velocity from pred_joint_coords),
but trains FreqPeriodicBlock which includes the learnable filter layer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python freq_pd_trainer.py --config configs/freq_pd_config.yaml
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import pandas as pd

from models.freq_periodic_block import FreqPeriodicBlock
from dataloaders.beat_mhr import JOINT_VEL_DIM, SELECTED_JOINT_IDXS


class JointVelocityDataset(Dataset):
    """
    Reads pred_joint_coords from npz files, selects upper-body + hand joints,
    computes frame-to-frame velocity, and returns sliding-window clips.
    """

    def __init__(self, data_dir: str, pose_rep: str, split_csv: str,
                 split_type: str = "train", pose_fps: int = 30,
                 pose_length: int = 64, stride: int = 20):
        self.pose_length = pose_length
        self.clips = []

        split_rule = pd.read_csv(split_csv)
        selected = split_rule.loc[split_rule['type'] == split_type]
        if split_type == 'train':
            extra = split_rule.loc[split_rule['type'] == 'additional']
            if not extra.empty:
                selected = pd.concat([selected, extra])

        npz_dir = os.path.join(data_dir, pose_rep)
        n_files = 0
        for _, row in selected.iterrows():
            fpath = os.path.join(npz_dir, row["id"] + ".npz")
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath, allow_pickle=True)
            if 'pred_joint_coords' not in data.files:
                continue

            joints = data['pred_joint_coords'].astype(np.float32)
            raw_fps = float(data['fps'][0]) if 'fps' in data.files else 30.0
            ds = int(raw_fps / pose_fps) if raw_fps >= pose_fps else 1
            joints = joints[::ds]

            selected_j = joints[:, SELECTED_JOINT_IDXS, :]
            vel = np.zeros_like(selected_j)
            vel[1:] = selected_j[1:] - selected_j[:-1]
            vel_flat = vel.reshape(len(vel), -1).astype(np.float32)

            n_frames = len(vel_flat)
            for start in range(0, n_frames - pose_length + 1, stride):
                self.clips.append(vel_flat[start:start + pose_length])
            n_files += 1

        logger.info(f"JointVelocityDataset [{split_type}]: {n_files} files, {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return torch.from_numpy(self.clips[idx])


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    data_dir = cfg.get("data_path", "./data_new/")
    split_csv = os.path.join(data_dir, "train_test_split.csv")

    dataset = JointVelocityDataset(
        data_dir=data_dir,
        pose_rep=cfg["pose_rep"],
        split_csv=split_csv,
        split_type="train",
        pose_fps=cfg.get("pose_fps", 30),
        pose_length=cfg.get("pose_length", 64),
        stride=cfg.get("stride", 20),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    model = FreqPeriodicBlock(
        input_dim=cfg.get("pd_input_dim", JOINT_VEL_DIM),
        n_channels=cfg.get("pd_n_channels", 10),
        max_T=cfg.get("max_T", 128),
        dropout=cfg.get("dropout", 0.1),
        use_ffn=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 5e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    lambda_vel = cfg.get("pd_lambda_vel", 0.5)

    n_params = sum(p.numel() for p in model.parameters())
    n_filter = sum(p.numel() for n, p in model.named_parameters() if 'filter_layer' in n)
    logger.info(f"FreqPD params: {n_params:,} (filter_layer: {n_filter:,})")
    logger.info(f"Training for {cfg['epochs']} epochs, batch_size={cfg['batch_size']}, lr={cfg['lr']}")

    best_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = {"rec": 0, "vel": 0, "total": 0}
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            loss, loss_dict = model.compute_loss(batch, lambda_vel=lambda_vel)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_norm", 1.0))
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        if epoch % cfg.get("log_period", 10) == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{cfg['epochs']} | "
                f"rec={epoch_losses['rec']:.6f} vel={epoch_losses['vel']:.6f} "
                f"total={epoch_losses['total']:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save(model.state_dict(), os.path.join(out_dir, "freq_pd_best.bin"))

        if epoch % cfg.get("save_period", 20) == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"freq_pd_epoch_{epoch}.bin"))

    torch.save(model.state_dict(), os.path.join(out_dir, "freq_pd_final.bin"))
    logger.info(f"Training complete. Best loss: {best_loss:.6f}. Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreqPD Module Pre-training")
    parser.add_argument("--config", type=str, default="configs/freq_pd_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
