"""
Standalone dual-person LMDB cache rebuild script.

Rebuilds train/test caches for the dual-person dataloader (beat_mhr_dual)
with 381d pose format and optional dialogue state sequences.

Usage:
    CUDA_VISIBLE_DEVICES=<GPU> python rebuild_cache_dual.py
"""

import os
import sys
import shutil

import torch
import torch.distributed as dist

from utils import config
from dataloaders.build_vocab import Vocab  # needed for pickle deserialization


def main():
    sys.argv = [
        "rebuild_cache_dual.py",
        "-c", "configs/dualmambatalk_v8.yaml",
    ]

    args = config.parse_args()
    args.is_train = True

    import random
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random.randint(40000, 50000))
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from dataloaders.beat_mhr_dual import CustomDataset

    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"Rebuilding dual cache for split: {split}")
        print(f"{'='*60}")

        cache_dir = (args.root_path + args.cache_path
                     + split + f"/{args.pose_rep}_dual_cache")
        if os.path.exists(cache_dir):
            print(f"Removing old cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        ds = CustomDataset(args, split, build_cache=True)
        print(f"  -> {len(ds)} samples")

    dist.destroy_process_group()
    print("\nDual cache rebuild complete!")


if __name__ == "__main__":
    main()
