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
    MHR Global motion AE trainer (non-quantized VAEConvZero).
    Input: global_rot(3) + cam_t(3) + contact(4) = 10 dims.
    Cache layout: body(130) + hand(108) + face(75) + global(10) = 323
    Global slice: [313:323]
    """
    def __init__(self, args):
        super().__init__(args)
        self.tracker = other_tools.EpochTracker(
            ["rec", "vel", "contact", "trans"], [False]*4
        )
        self.rec_loss = torch.nn.MSELoss(reduction='mean')
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.global_slice = (313, 323)  # global(10) in the cached pose

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose_full = dict_data["pose"].cuda()
            bs, n, _ = tar_pose_full.shape
            tar_global = tar_pose_full[:, :, self.global_slice[0]:self.global_slice[1]]  # (bs, n, 10)
            t_data = time.time() - t_start

            self.opt.zero_grad()
            net_out = self.model(tar_global)
            rec_global = net_out["rec_pose"]

            # Rotation + translation loss (first 6 dims)
            loss_rot_trans = self.rec_loss(rec_global[:, :, :6], tar_global[:, :, :6])
            self.tracker.update_meter("trans", "train", loss_rot_trans.item())

            # Contact loss (last 4 dims)
            loss_contact = self.rec_loss(rec_global[:, :, 6:], tar_global[:, :, 6:])
            self.tracker.update_meter("contact", "train", loss_contact.item())

            loss_rec = (loss_rot_trans + loss_contact) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss = loss_rec

            # Translation velocity
            vel_loss = self.vel_loss(
                rec_global[:, 1:, 3:6] - rec_global[:, :-1, 3:6],
                tar_global[:, 1:, 3:6] - tar_global[:, :-1, 3:6]
            ) * self.args.rec_weight
            self.tracker.update_meter("vel", "train", vel_loss.item())
            g_loss += vel_loss

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
                tar_global = tar_pose_full[:, :, self.global_slice[0]:self.global_slice[1]]
                net_out = self.model(tar_global)
                rec_global = net_out["rec_pose"]
                loss_rec = self.rec_loss(rec_global, tar_global) * self.args.rec_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
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
                tar_global = tar_pose_full[:, :, self.global_slice[0]:self.global_slice[1]]
                remain = n % self.args.pose_length
                if remain > 0:
                    tar_global = tar_global[:, :n - remain, :]
                net_out = self.model(tar_global)
                rec_global = net_out["rec_pose"]
                np.savez(
                    results_save_path + "res_" + test_seq_list.iloc[its]['id'] + '.npz',
                    rec_global=rec_global.cpu().numpy(),
                    tar_global=tar_global.cpu().numpy(),
                )
        logger.info(f"Test results saved to {results_save_path}")
