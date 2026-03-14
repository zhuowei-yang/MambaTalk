"""
MHR Continuous Representation VQ-VAE Trainer.
Operates on 6D continuous body (260d) or hand PCA (108d).
No rotation conversion needed — data is already in continuous space from DataLoader.
"""
import train
import os
import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.mhr_utils import compact_cont_to_model_params_body

BODY_CONT_DIM = 260
HAND_DIM = 108
HAND_OFFSET = 260


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # BaseTrainer creates smplx + alignment which we don't need; override with None
        self.smplx = None
        self.alignmenter = None
        self.l1_calculator = None
        self.tracker = other_tools.EpochTracker(
            ["rec", "vel", "acc", "com", "res"], [False, False, False, False, False]
        )
        self.rec_loss = nn.MSELoss(reduction='mean')
        self.vel_loss = nn.L1Loss(reduction='mean')

        pd = args.pose_dims
        if pd == BODY_CONT_DIM:
            self.pose_slice = (0, BODY_CONT_DIM)
        elif pd == HAND_DIM:
            self.pose_slice = (HAND_OFFSET, HAND_OFFSET + HAND_DIM)
        else:
            raise ValueError(f"ae_mhr_cont_trainer: unsupported pose_dims={pd}, expected {BODY_CONT_DIM} or {HAND_DIM}")

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose_full = dict_data["pose"].cuda()
            bs, n, _ = tar_pose_full.shape
            tar_pose = tar_pose_full[:, :, self.pose_slice[0]:self.pose_slice[1]]
            t_data = time.time() - t_start

            self.opt.zero_grad()
            net_out = self.model(tar_pose)
            rec_pose = net_out["rec_pose"]

            loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss = loss_rec

            vw = getattr(self.args, 'vel_weight', 1.0)
            aw = getattr(self.args, 'acc_weight', 1.0)
            vel_loss = self.vel_loss(
                rec_pose[:, 1:] - rec_pose[:, :-1],
                tar_pose[:, 1:] - tar_pose[:, :-1]
            ) * self.args.rec_weight * vw
            acc_loss = self.vel_loss(
                rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1],
                tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]
            ) * self.args.rec_weight * aw
            self.tracker.update_meter("vel", "train", vel_loss.item())
            self.tracker.update_meter("acc", "train", acc_loss.item())
            g_loss += vel_loss + acc_loss

            if "VQVAE" in self.args.g_name:
                loss_emb = net_out["embedding_loss"]
                g_loss += loss_emb
                self.tracker.update_meter("com", "train", loss_emb.item())

            if "z_res" in net_out:
                res_reg_w = getattr(self.args, 'res_reg_weight', 0.1)
                loss_res_reg = res_reg_w * torch.mean(net_out["z_res"] ** 2)
                g_loss += loss_res_reg
                self.tracker.update_meter("res", "train", loss_res_reg.item())

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
                tar_pose = tar_pose_full[:, :, self.pose_slice[0]:self.pose_slice[1]]
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
                if "VQVAE" in self.args.g_name:
                    self.tracker.update_meter("com", "val", net_out["embedding_loss"].item())
                if "z_res" in net_out:
                    res_reg_w = getattr(self.args, 'res_reg_weight', 0.1)
                    self.tracker.update_meter("res", "val", (res_reg_w * torch.mean(net_out["z_res"] ** 2)).item())
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
                tar_pose = tar_pose_full[:, :, self.pose_slice[0]:self.pose_slice[1]]
                remain = n % self.args.pose_length
                if remain > 0:
                    tar_pose = tar_pose[:, :n - remain, :]
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]

                rec_np = rec_pose.cpu().numpy()
                tar_np = tar_pose[:, :rec_pose.shape[1], :].cpu().numpy()

                save_dict = {'rec_pose': rec_np, 'tar_pose': tar_np}
                if self.pose_slice == (0, BODY_CONT_DIM):
                    rec_euler = compact_cont_to_model_params_body(rec_pose.reshape(-1, BODY_CONT_DIM))
                    tar_euler = compact_cont_to_model_params_body(tar_pose[:, :rec_pose.shape[1]].reshape(-1, BODY_CONT_DIM))
                    save_dict['rec_euler'] = rec_euler.cpu().numpy()
                    save_dict['tar_euler'] = tar_euler.cpu().numpy()

                np.savez(
                    results_save_path + "res_" + test_seq_list.iloc[its]['id'] + '.npz',
                    **save_dict,
                )
        logger.info(f"Test results saved to {results_save_path}")
