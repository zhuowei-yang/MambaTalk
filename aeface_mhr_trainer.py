import train
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class CustomTrainer(train.BaseTrainer):
    """
    MHR Face VQ-VAE trainer.
    Input: expr_params(72) + jaw(3) = 75 dims.
    Cache layout: body(130) + hand(108) + face(75) + global(10) = 323
    Face slice: [238:313]
    """
    def __init__(self, args):
        super().__init__(args)
        self.tracker = other_tools.EpochTracker(
            ["rec", "vel", "acc", "com", "expr", "jaw"], [False]*6
        )
        self.rec_loss = torch.nn.MSELoss(reduction='mean')
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.face_slice = (238, 313)  # face(75) in the cached pose

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose_full = dict_data["pose"].cuda()
            bs, n, _ = tar_pose_full.shape
            tar_face = tar_pose_full[:, :, self.face_slice[0]:self.face_slice[1]]  # (bs, n, 75)
            t_data = time.time() - t_start

            self.opt.zero_grad()
            net_out = self.model(tar_face)
            rec_face = net_out["rec_pose"]

            # Expression loss (first 72 dims)
            loss_expr = self.rec_loss(rec_face[:, :, :72], tar_face[:, :, :72])
            self.tracker.update_meter("expr", "train", loss_expr.item())

            # Jaw loss (last 3 dims)
            loss_jaw = self.rec_loss(rec_face[:, :, 72:], tar_face[:, :, 72:])
            self.tracker.update_meter("jaw", "train", loss_jaw.item())

            loss_rec = (loss_expr + loss_jaw) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss = loss_rec

            vel_loss = self.vel_loss(rec_face[:, 1:] - rec_face[:, :-1],
                                     tar_face[:, 1:] - tar_face[:, :-1]) * self.args.rec_weight
            acc_loss = self.vel_loss(rec_face[:, 2:] + rec_face[:, :-2] - 2 * rec_face[:, 1:-1],
                                     tar_face[:, 2:] + tar_face[:, :-2] - 2 * tar_face[:, 1:-1]) * self.args.rec_weight
            self.tracker.update_meter("vel", "train", vel_loss.item())
            self.tracker.update_meter("acc", "train", acc_loss.item())
            g_loss += vel_loss + acc_loss

            if "VQVAE" in self.args.g_name:
                loss_emb = net_out["embedding_loss"]
                g_loss += loss_emb
                self.tracker.update_meter("com", "train", loss_emb.item())

            g_loss.backward()
            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.args.debug and its == 1:
                break
        self.opt_s.step(epoch)

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                tar_pose_full = dict_data["pose"].cuda()
                tar_face = tar_pose_full[:, :, self.face_slice[0]:self.face_slice[1]]
                net_out = self.model(tar_face)
                rec_face = net_out["rec_pose"]
                loss_rec = self.rec_loss(rec_face, tar_face) * self.args.rec_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
                if "VQVAE" in self.args.g_name:
                    self.tracker.update_meter("com", "val", net_out["embedding_loss"].item())
            self.val_recording(epoch)

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path):
            return 0
        os.makedirs(results_save_path)
        self.model.eval()
        test_seq_list = self.test_data.selected_file
        with torch.no_grad():
            for its, dict_data in enumerate(self.test_loader):
                tar_pose_full = dict_data["pose"].cuda()
                bs, n, _ = tar_pose_full.shape
                tar_face = tar_pose_full[:, :, self.face_slice[0]:self.face_slice[1]]
                remain = n % self.args.pose_length
                if remain > 0:
                    tar_face = tar_face[:, :n - remain, :]
                net_out = self.model(tar_face)
                rec_face = net_out["rec_pose"]
                np.savez(
                    results_save_path + "res_" + test_seq_list.iloc[its]['id'] + '.npz',
                    rec_face=rec_face.cpu().numpy(),
                    tar_face=tar_face.cpu().numpy(),
                )
        logger.info(f"Test results saved to {results_save_path}")
