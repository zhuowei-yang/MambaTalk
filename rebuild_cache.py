"""
Standalone LMDB cache rebuild script.

Rebuilds train/test/val caches with joint_velocity (10th field) included,
without requiring the full training pipeline (no model, no DDP).

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=<GPU> python rebuild_cache.py
"""

import os
import sys
import argparse
import shutil

import torch
import torch.distributed as dist

from utils import config
from dataloaders.build_vocab import Vocab  # needed for pickle deserialization


def main():
    sys.argv = [
        "rebuild_cache.py",
        "--config", "configs/mambatalk_mhr_new.yaml",
        "--new_cache", "True",
    ]

    args = config.parse_args()
    args.is_train = True

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9599"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from dataloaders.beat_mhr import CustomDataset

    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"Rebuilding cache for split: {split}")
        print(f"{'='*60}")

        cache_dir = args.root_path + args.cache_path + split + f"/{args.pose_rep}_cache"
        if os.path.exists(cache_dir):
            print(f"Removing old cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        ds = CustomDataset(args, split, build_cache=True)
        print(f"  -> {len(ds)} samples")

    dist.destroy_process_group()
    print("\nCache rebuild complete!")


if __name__ == "__main__":
    main()
