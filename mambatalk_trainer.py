import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
import librosa

# MHR cache layout: body(130) + hand(108) + face(75) + global(7) = 320
# global = global_rot(3) + contact(4); cam_t stored separately in trans field
MHR_BODY_SLICE = (0, 130)
MHR_HAND_SLICE = (130, 238)
MHR_FACE_SLICE = (238, 313)
MHR_GLOBAL_SLICE = (313, 320)

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.joints = 55

        self.tracker = other_tools.EpochTracker(
            ["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc',
             'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent",
             "cls_full", "cls_self", "cls_word", "latent_word", "latent_self", "accel"],
            [False, True, True] + [False]*20
        )

        # Load MHR VQ-VAE models (new dimensions)
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        
        self.args.vae_layer = 2
        self.args.vae_length = 256
        
        self.args.vae_test_dim = 75  # face: expr(72) + jaw(3)
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_face, self.args.data_path_1 + "pretrained_vq/mhr_face.bin", args.e_name)
        
        self.args.vae_test_dim = 130  # body pose (non-hand, non-jaw)
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, self.args.data_path_1 + "pretrained_vq/mhr_body.bin", args.e_name)
        
        self.args.vae_test_dim = 108  # hand PCA 6D
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 + "pretrained_vq/mhr_hand.bin", args.e_name)
        
        # Global: no VQ-VAE, MambaTalk predicts raw 10d directly
        
        # Restore main model args
        self.args.vae_test_dim = args.pose_dims
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_face.eval()
        self.vq_model_body.eval()
        self.vq_model_hands.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.cls_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        # MHR cached pose: body(130) + hand(108) + face(75) + global(10) = 323
        tar_pose_raw = dict_data["pose"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long().clamp(0, 24)
        bs, n, _ = tar_pose_raw.shape

        # Direct slicing — no rotation conversion needed
        tar_pose_body = tar_pose_raw[:, :, MHR_BODY_SLICE[0]:MHR_BODY_SLICE[1]]    # (bs, n, 130)
        tar_pose_hands = tar_pose_raw[:, :, MHR_HAND_SLICE[0]:MHR_HAND_SLICE[1]]   # (bs, n, 108)
        tar_pose_face = tar_pose_raw[:, :, MHR_FACE_SLICE[0]:MHR_FACE_SLICE[1]]    # (bs, n, 75)
        tar_pose_global = tar_pose_raw[:, :, MHR_GLOBAL_SLICE[0]:MHR_GLOBAL_SLICE[1]]  # (bs, n, 7)
        
        tar_trans = dict_data["trans"].to(self.rank)  # cam_t from data (sequence-level constant)
        tar_exps = tar_pose_face[:, :, :72]     # expr_params
        tar_contact = tar_pose_global[:, :, 3:]  # contact(4), now at index 3 since cam_t removed

        # VQ-VAE encoding (frozen)
        tar_index_face = self.vq_model_face.map2index(tar_pose_face)
        tar_index_body = self.vq_model_body.map2index(tar_pose_body)
        tar_index_hands = self.vq_model_hands.map2index(tar_pose_hands)
      
        latent_face = self.vq_model_face.map2latent(tar_pose_face)
        latent_body = self.vq_model_body.map2latent(tar_pose_body)
        latent_hands = self.vq_model_hands.map2latent(tar_pose_hands)
        
        # Global: raw 10d directly (no AE)
        latent_global = tar_pose_global

        latent_in = torch.cat([latent_body, latent_hands, latent_global], dim=2)
        index_in = torch.stack([tar_index_body, tar_index_hands], dim=-1).long()
        
        # Full pose for motion input to the model
        latent_all = tar_pose_raw  # (bs, n, 323) — MHR native params directly
        
        return {
            "tar_pose_face": tar_pose_face,
            "tar_pose_body": tar_pose_body,
            "tar_pose_hands": tar_pose_hands,
            "tar_pose_global": tar_pose_global,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose_raw,
            "tar_index_face": tar_index_face,
            "tar_index_body": tar_index_body,
            "tar_index_hands": tar_index_hands,
            "latent_face": latent_face,
            "latent_body": latent_body,
            "latent_hands": latent_hands,
            "latent_global": latent_global,
            "latent_in": latent_in,
            "index_in": index_in,
            "tar_id": tar_id,
            "latent_all": latent_all,
            "tar_contact": tar_contact,
        }
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, _ = loaded_data["tar_pose"].shape
        # ------ full generation task ------ #
        mask_val = torch.ones(bs, n, self.args.pose_dims).float().cuda()
        mask_val[:, :self.args.pre_frames, :] = 0.0
        
        net_out_val = self.model(
            loaded_data['in_audio'], loaded_data['in_word'], mask=mask_val,
            in_id=loaded_data['tar_id'], in_motion=loaded_data['latent_all'],
            use_attentions=True)
        g_loss_final = 0
        loss_latent_face = self.reclatent_loss(net_out_val["rec_face"], loaded_data["latent_face"])
        loss_latent_global = self.reclatent_loss(net_out_val["rec_global"], loaded_data["latent_global"])
        loss_latent_hands = self.reclatent_loss(net_out_val["rec_hands"], loaded_data["latent_hands"])
        loss_latent_body = self.reclatent_loss(net_out_val["rec_body"], loaded_data["latent_body"])
        loss_latent = self.args.lf*loss_latent_face + self.args.ll*loss_latent_global + self.args.lh*loss_latent_hands + self.args.lu*loss_latent_body
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent
        
        # Velocity loss: penalize frame-to-frame prediction jumps for temporal smoothness
        rec_global = net_out_val["rec_global"]
        tar_global = loaded_data["latent_global"]
        loss_vel_global = self.vel_loss(rec_global[:, 1:] - rec_global[:, :-1],
                                        tar_global[:, 1:] - tar_global[:, :-1])
        rec_body_lat = net_out_val["rec_body"]
        tar_body_lat = loaded_data["latent_body"]
        loss_vel_body = self.vel_loss(rec_body_lat[:, 1:] - rec_body_lat[:, :-1],
                                      tar_body_lat[:, 1:] - tar_body_lat[:, :-1])
        rec_hands_lat = net_out_val["rec_hands"]
        tar_hands_lat = loaded_data["latent_hands"]
        loss_vel_hands = self.vel_loss(rec_hands_lat[:, 1:] - rec_hands_lat[:, :-1],
                                       tar_hands_lat[:, 1:] - tar_hands_lat[:, :-1])
        loss_vel = self.args.vel_global_weight * loss_vel_global + self.args.vel_body_weight * loss_vel_body + self.args.vel_hands_weight * loss_vel_hands
        self.tracker.update_meter("vel", "train", loss_vel.item())
        g_loss_final += loss_vel

        # Acceleration loss: penalize sudden velocity changes (second-order smoothness)
        rec_acc_global = rec_global[:, 2:] - 2 * rec_global[:, 1:-1] + rec_global[:, :-2]
        tar_acc_global = tar_global[:, 2:] - 2 * tar_global[:, 1:-1] + tar_global[:, :-2]
        loss_acc_global = self.vel_loss(rec_acc_global, tar_acc_global)
        rec_acc_body = rec_body_lat[:, 2:] - 2 * rec_body_lat[:, 1:-1] + rec_body_lat[:, :-2]
        tar_acc_body = tar_body_lat[:, 2:] - 2 * tar_body_lat[:, 1:-1] + tar_body_lat[:, :-2]
        loss_acc_body = self.vel_loss(rec_acc_body, tar_acc_body)
        rec_acc_hands = rec_hands_lat[:, 2:] - 2 * rec_hands_lat[:, 1:-1] + rec_hands_lat[:, :-2]
        tar_acc_hands = tar_hands_lat[:, 2:] - 2 * tar_hands_lat[:, 1:-1] + tar_hands_lat[:, :-2]
        loss_acc_hands = self.vel_loss(rec_acc_hands, tar_acc_hands)
        loss_acc = self.args.acc_global_weight * loss_acc_global + self.args.acc_body_weight * loss_acc_body + self.args.acc_hands_weight * loss_acc_hands
        self.tracker.update_meter("accel", "train", loss_acc.item())
        g_loss_final += loss_acc

        # Pose-space body loss: soft codebook lookup is differentiable,
        # decoder is detached to prevent gradient explosion through frozen conv layers
        body_probs = F.softmax(net_out_val["cls_body"], dim=2)
        codebook_weight = self.vq_model_body.quantizer.embedding.weight
        soft_body_latent = torch.matmul(body_probs, codebook_weight)
        with torch.no_grad():
            rec_pose_body_decoded = self.vq_model_body.decoder(soft_body_latent)
        # Soft-latent space smoothness loss (differentiable through codebook lookup)
        tar_body_codebook = loaded_data["latent_body"]
        loss_pose_body = self.reclatent_loss(soft_body_latent, tar_body_codebook)
        loss_pose_vel_body = self.vel_loss(
            soft_body_latent[:, 1:] - soft_body_latent[:, :-1],
            tar_body_codebook[:, 1:] - tar_body_codebook[:, :-1])
        g_loss_final += self.args.pose_body_weight * loss_pose_body + self.args.pose_vel_body_weight * loss_pose_vel_body

        rec_index_face_val = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_body_val = self.log_softmax(net_out_val["cls_body"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_hands_val = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
        tar_index_face = loaded_data["tar_index_face"].reshape(-1)
        tar_index_body = loaded_data["tar_index_body"].reshape(-1)
        tar_index_hands = loaded_data["tar_index_hands"].reshape(-1)
        # Body uses label-smoothed CrossEntropyLoss to reduce codebook oscillation
        loss_cls = self.args.cf*self.cls_loss(rec_index_face_val, tar_index_face)\
            + self.args.cu*self.cls_loss_smooth(net_out_val["cls_body"].reshape(-1, self.args.vae_codebook_size), tar_index_body)\
            + self.args.ch*self.cls_loss(rec_index_hands_val, tar_index_hands)
        self.tracker.update_meter("cls_full", "train", loss_cls.item())
        g_loss_final += loss_cls
        
        if mode == 'train':
            mask_ratio = (epoch / self.args.epochs) * 0.95 + 0.05
            mask = torch.rand(bs, n, self.args.pose_dims) < mask_ratio
            mask = mask.float().cuda()
            net_out_self = self.model(
                loaded_data['in_audio'], loaded_data['in_word'], mask=mask,
                in_id=loaded_data['tar_id'], in_motion=loaded_data['latent_all'],
                use_attentions=True, use_word=False)
            
            loss_latent_face_self = self.reclatent_loss(net_out_self["rec_face"], loaded_data["latent_face"])
            loss_latent_global_self = self.reclatent_loss(net_out_self["rec_global"], loaded_data["latent_global"])
            loss_latent_hands_self = self.reclatent_loss(net_out_self["rec_hands"], loaded_data["latent_hands"])
            loss_latent_body_self = self.reclatent_loss(net_out_self["rec_body"], loaded_data["latent_body"])
            loss_latent_self = self.args.lf*loss_latent_face_self + self.args.ll*loss_latent_global_self + self.args.lh*loss_latent_hands_self + self.args.lu*loss_latent_body_self
            self.tracker.update_meter("latent_self", "train", loss_latent_self.item())
            g_loss_final += loss_latent_self
            rec_index_face_self = self.log_softmax(net_out_self["cls_face"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_hands_self = self.log_softmax(net_out_self["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
            index_loss_top_self = self.cls_loss(rec_index_face_self, tar_index_face) \
                + self.cls_loss_smooth(net_out_self["cls_body"].reshape(-1, self.args.vae_codebook_size), tar_index_body) \
                + self.cls_loss(rec_index_hands_self, tar_index_hands)
            self.tracker.update_meter("cls_self", "train", index_loss_top_self.item())
            g_loss_final += index_loss_top_self
            
            net_out_word = self.model(
                loaded_data['in_audio'], loaded_data['in_word'], mask=mask,
                in_id=loaded_data['tar_id'], in_motion=loaded_data['latent_all'],
                use_attentions=True, use_word=True)
            
            loss_latent_face_word = self.reclatent_loss(net_out_word["rec_face"], loaded_data["latent_face"])
            loss_latent_global_word = self.reclatent_loss(net_out_word["rec_global"], loaded_data["latent_global"])
            loss_latent_hands_word = self.reclatent_loss(net_out_word["rec_hands"], loaded_data["latent_hands"])
            loss_latent_body_word = self.reclatent_loss(net_out_word["rec_body"], loaded_data["latent_body"])
            loss_latent_word = self.args.lf*loss_latent_face_word + self.args.ll*loss_latent_global_word + self.args.lh*loss_latent_hands_word + self.args.lu*loss_latent_body_word
            self.tracker.update_meter("latent_word", "train", loss_latent_word.item())
            g_loss_final += loss_latent_word

            rec_index_face_word = self.log_softmax(net_out_word["cls_face"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_hands_word = self.log_softmax(net_out_word["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
            index_loss_top_word = self.cls_loss(rec_index_face_word, tar_index_face) \
                + self.cls_loss_smooth(net_out_word["cls_body"].reshape(-1, self.args.vae_codebook_size), tar_index_body) \
                + self.cls_loss(rec_index_hands_word, tar_index_hands)
            self.tracker.update_meter("cls_word", "train", index_loss_top_word.item())
            g_loss_final += index_loss_top_word

        if mode != 'train':
            # Decode VQ-VAE latents back to MHR params
            if self.args.cu != 0:
                _, rec_idx_body = torch.max(rec_index_body_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_body = self.vq_model_body.decode(rec_idx_body)
            else:
                _, rec_idx_body, _, _ = self.vq_model_body.quantizer(net_out_val["rec_body"])
                rec_body = self.vq_model_body.decoder(rec_idx_body)
            if self.args.ch != 0:
                _, rec_idx_hands = torch.max(rec_index_hands_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_hands = self.vq_model_hands.decode(rec_idx_hands)
            else:
                _, rec_idx_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
                rec_hands = self.vq_model_hands.decoder(rec_idx_hands)
            if self.args.cf != 0:
                _, rec_idx_face = torch.max(rec_index_face_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_face = self.vq_model_face.decode(rec_idx_face)
            else:
                _, rec_idx_face, _, _ = self.vq_model_face.quantizer(net_out_val["rec_face"])
                rec_face = self.vq_model_face.decoder(rec_idx_face)
            rec_global = net_out_val["rec_global"]
            rec_pose = torch.cat([rec_body, rec_hands, rec_face, rec_global], dim=-1)

        if mode == 'train':
            return g_loss_final
        elif mode == 'val':
            return {
                'rec_pose': rec_pose,
                'tar_pose': loaded_data["tar_pose"],
            }
        else:
            return {
                'rec_pose': rec_pose,
                'tar_pose': loaded_data["tar_pose"],
                'tar_exps': loaded_data["tar_exps"],
                'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
            }
    

    def _g_test(self, loaded_data):
        bs, n, _ = loaded_data["tar_pose"].shape
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        in_word = loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]

        remain = n % 8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

        # MHR direct slicing
        tar_pose_body = tar_pose[:, :, MHR_BODY_SLICE[0]:MHR_BODY_SLICE[1]]
        tar_pose_hands = tar_pose[:, :, MHR_HAND_SLICE[0]:MHR_HAND_SLICE[1]]
        tar_pose_face = tar_pose[:, :, MHR_FACE_SLICE[0]:MHR_FACE_SLICE[1]]
        tar_pose_global = tar_pose[:, :, MHR_GLOBAL_SLICE[0]:MHR_GLOBAL_SLICE[1]]
        latent_all = tar_pose
        
        rec_index_all_face = []
        rec_index_all_body = []
        rec_index_all_hands = []
        rec_latent_all_global = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :].clone()
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :].clone()
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = self.model(
                in_audio=in_audio_tmp, in_word=in_word_tmp, mask=mask_val,
                in_motion=latent_all_tmp, in_id=in_id_tmp, use_attentions=True)
            
            # Body VQ-VAE decoding
            if self.args.cu != 0:
                rec_idx_body = self.log_softmax(net_out_val["cls_body"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_idx_body = torch.max(rec_idx_body.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            else:
                _, rec_idx_body, _, _ = self.vq_model_body.quantizer(net_out_val["rec_body"])
            # Hand VQ-VAE decoding
            if self.args.ch != 0:
                rec_idx_hands = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_idx_hands = torch.max(rec_idx_hands.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            else:
                _, rec_idx_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
            # Face VQ-VAE decoding
            if self.args.cf != 0:
                rec_idx_face = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_idx_face = torch.max(rec_idx_face.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            else:
                _, rec_idx_face, _, _ = self.vq_model_face.quantizer(net_out_val["rec_face"])
            # Global AE (no quantizer)
            rec_global_latent = net_out_val["rec_global"]

            if i == 0:
                rec_index_all_face.append(rec_idx_face)
                rec_index_all_body.append(rec_idx_body)
                rec_index_all_hands.append(rec_idx_hands)
                rec_latent_all_global.append(rec_global_latent)
            else:
                rec_index_all_face.append(rec_idx_face[:, self.args.pre_frames:])
                rec_index_all_body.append(rec_idx_body[:, self.args.pre_frames:])
                rec_index_all_hands.append(rec_idx_hands[:, self.args.pre_frames:])
                rec_latent_all_global.append(rec_global_latent[:, self.args.pre_frames:])

            # Decode for autoregressive feedback
            rec_body_last = self.vq_model_body.decode(rec_idx_body) if self.args.cu != 0 else self.vq_model_body.decoder(rec_idx_body)
            rec_hands_last = self.vq_model_hands.decode(rec_idx_hands) if self.args.ch != 0 else self.vq_model_hands.decoder(rec_idx_hands)
            rec_face_last = self.vq_model_face.decode(rec_idx_face) if self.args.cf != 0 else self.vq_model_face.decoder(rec_idx_face)
            # Global: raw prediction, no decoder needed
            latent_last = torch.cat([rec_body_last, rec_hands_last, rec_face_last, rec_global_latent], dim=-1)

        # Final decode all
        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_body = torch.cat(rec_index_all_body, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)
        rec_latent_global = torch.cat(rec_latent_all_global, dim=1)
        
        rec_body = self.vq_model_body.decode(rec_index_body) if self.args.cu != 0 else self.vq_model_body.decoder(rec_index_body)
        rec_hands = self.vq_model_hands.decode(rec_index_hands) if self.args.ch != 0 else self.vq_model_hands.decoder(rec_index_hands)
        rec_face = self.vq_model_face.decode(rec_index_face) if self.args.cf != 0 else self.vq_model_face.decoder(rec_index_face)
        rec_global = rec_latent_global  # Raw prediction, no decoder

        # Reconstruct MHR native params: body(130) + hand(108) + face(75) + global(10)
        rec_pose = torch.cat([rec_body, rec_hands, rec_face, rec_global], dim=-1)  # (bs, n, 323)
        n = rec_pose.shape[1]

        rec_exps = rec_face[:, :, :72]  # expr_params
        rec_trans = tar_trans[:, :n, :]  # cam_t from original data (not predicted)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }
    

    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 
    
    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'val', epoch)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                n = tar_pose.shape[1]
                remain = n % self.args.vae_test_len
                tar_pose = tar_pose[:, :n-remain, :]
                rec_pose = rec_pose[:, :n-remain, :]
                if self.eval_copy is not None:
                    latent_out = self.eval_copy.map2latent(rec_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                    latent_ori = self.eval_copy.map2latent(tar_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                    if its == 0:
                        latent_out_all = latent_out
                        latent_ori_all = latent_ori
                    else:
                        latent_out_all = np.concatenate([latent_out_all, latent_out], axis=0)                 
                        latent_ori_all = np.concatenate([latent_ori_all, latent_ori], axis=0)
                if self.args.debug:
                    if its == 1: break
        if self.eval_copy is not None:
            fid_motion = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            self.tracker.update_meter("fid", "val", fid_motion)
        self.val_recording(epoch) 
    
    
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        latent_out = []
        latent_ori = []
        self.model.eval()
        if self.eval_copy is not None:
            self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                bs, n = rec_pose.shape[0], rec_pose.shape[1]

                if self.eval_copy is not None:
                    remain = n % self.args.vae_test_len
                    latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                    latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())

                # Save MHR native format npz
                rec_np = rec_pose.detach().cpu().numpy().reshape(bs*n, -1)
                tar_np = tar_pose.detach().cpu().numpy().reshape(bs*n, -1)
                
                gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                shape = gt_npz['shape_params'] if 'shape_params' in gt_npz.files else np.zeros(45, dtype=np.float32)
                scale_params = gt_npz['scale_params'][0] if 'scale_params' in gt_npz.files else np.zeros(28, dtype=np.float32)
                focal_length = float(gt_npz['focal_length'][0]) if 'focal_length' in gt_npz.files else 1500.0
                orig_width = int(gt_npz['width'][0]) if 'width' in gt_npz.files else 1080
                orig_height = int(gt_npz['height'][0]) if 'height' in gt_npz.files else 1920
                render_consts = dict(shape_params=shape, scale_params=scale_params,
                    focal_length=np.array([focal_length]), width=np.array([orig_width]),
                    height=np.array([orig_height]), mocap_frame_rate=30)
                
                # cam_t: sequence-level constant from original data
                cam_t_seq = net_out['tar_trans'].detach().cpu().numpy().reshape(bs*n, -1)
                cam_t_mean = cam_t_seq.mean(axis=0, keepdims=True).repeat(bs*n, axis=0)
                
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    body_pose_params=np.concatenate([tar_np[:, MHR_BODY_SLICE[0]:MHR_BODY_SLICE[1]], tar_np[:, MHR_FACE_SLICE[0]+72:MHR_FACE_SLICE[1]]], axis=1),
                    hand_pose_params=tar_np[:, MHR_HAND_SLICE[0]:MHR_HAND_SLICE[1]],
                    expr_params=tar_np[:, MHR_FACE_SLICE[0]:MHR_FACE_SLICE[0]+72],
                    global_rot=tar_np[:, MHR_GLOBAL_SLICE[0]:MHR_GLOBAL_SLICE[0]+3],
                    pred_cam_t=cam_t_seq,
                    **render_consts,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    body_pose_params=np.concatenate([rec_np[:, MHR_BODY_SLICE[0]:MHR_BODY_SLICE[1]], rec_np[:, MHR_FACE_SLICE[0]+72:MHR_FACE_SLICE[1]]], axis=1),
                    hand_pose_params=rec_np[:, MHR_HAND_SLICE[0]:MHR_HAND_SLICE[1]],
                    expr_params=rec_np[:, MHR_FACE_SLICE[0]:MHR_FACE_SLICE[0]+72],
                    global_rot=rec_np[:, MHR_GLOBAL_SLICE[0]:MHR_GLOBAL_SLICE[0]+3],
                    pred_cam_t=cam_t_mean,
                    **render_consts,
                )
                total_length += n

        if self.eval_copy is not None and len(latent_out) > 0:
            latent_out_all = np.concatenate(latent_out, axis=0)
            latent_ori_all = np.concatenate(latent_ori, axis=0)
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            logger.info(f"fid score: {fid}")
            self.test_recording("fid", fid, epoch)

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")


    def test_demo(self, epoch):
        """Generate motion and save as MHR native npz (no metrics)"""
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                rec_pose = net_out['rec_pose']
                tar_pose = net_out['tar_pose']
                bs, n = rec_pose.shape[0], rec_pose.shape[1]

                rec_np = rec_pose.detach().cpu().numpy().reshape(bs*n, -1)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                shape = gt_npz['shape_params'] if 'shape_params' in gt_npz.files else np.zeros(45, dtype=np.float32)
                scale_params = gt_npz['scale_params'][0] if 'scale_params' in gt_npz.files else np.zeros(28, dtype=np.float32)
                focal_length = float(gt_npz['focal_length'][0]) if 'focal_length' in gt_npz.files else 1500.0
                orig_width = int(gt_npz['width'][0]) if 'width' in gt_npz.files else 1080
                orig_height = int(gt_npz['height'][0]) if 'height' in gt_npz.files else 1920
                
                cam_t_demo = net_out['tar_trans'].detach().cpu().numpy().reshape(bs*n, -1)
                cam_t_demo_mean = cam_t_demo.mean(axis=0, keepdims=True).repeat(bs*n, axis=0)
                
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    body_pose_params=np.concatenate([rec_np[:, MHR_BODY_SLICE[0]:MHR_BODY_SLICE[1]], rec_np[:, MHR_FACE_SLICE[0]+72:MHR_FACE_SLICE[1]]], axis=1),
                    hand_pose_params=rec_np[:, MHR_HAND_SLICE[0]:MHR_HAND_SLICE[1]],
                    expr_params=rec_np[:, MHR_FACE_SLICE[0]:MHR_FACE_SLICE[0]+72],
                    global_rot=rec_np[:, MHR_GLOBAL_SLICE[0]:MHR_GLOBAL_SLICE[0]+3],
                    pred_cam_t=cam_t_demo_mean,
                    shape_params=shape, scale_params=scale_params,
                    focal_length=np.array([focal_length]),
                    width=np.array([orig_width]), height=np.array([orig_height]),
                    mocap_frame_rate=30,
                )
                total_length += n

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
