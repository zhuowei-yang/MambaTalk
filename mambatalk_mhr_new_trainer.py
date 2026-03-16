"""
MambaTalk MHR trainer adapted for codebook=256, latent=256, layer=2 (original architecture).
Cache layout: body_cont(260) + hand(108) + global_rot_6d(6) + cam_t(3) + contact(4) = 381d
"""
import train
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.mhr_utils import compact_cont_to_model_params_body, rot6d_to_euler3

BODY_CONT_SLICE = (0, 260)
HAND_SLICE = (260, 368)
GLOBAL_ROT6D_SLICE = (368, 374)
CAM_T_SLICE = (374, 377)
CONTACT_SLICE = (377, 381)
GLOBAL_FULL_SLICE = (368, 381)

BODY_CODEBOOK = 512
HAND_CODEBOOK = 256
VAE_LATENT = 256
VAE_LAYER = 2


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.joints = 55
        self.smplx = None
        self.alignmenter = None

        self.tracker = other_tools.EpochTracker(
            ["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc',
             'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent",
             "cls_full", "cls_self", "cls_word", "latent_word", "latent_self", "accel"],
            [False, True, True] + [False] * 20
        )

        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])

        self.args.vae_layer = VAE_LAYER
        self.args.vae_length = VAE_LATENT
        self.args.vae_codebook_size = BODY_CODEBOOK
        self.args.vae_test_dim = 260
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, self.args.data_path_1 + "pretrained_vq/mhr_cont_body.bin", args.e_name)

        self.args.vae_codebook_size = HAND_CODEBOOK
        self.args.vae_test_dim = 108
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 + "pretrained_vq/mhr_cont_hand.bin", args.e_name)

        self.args.vae_test_dim = 75
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)

        self.args.vae_test_dim = args.pose_dims
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_body.eval()
        self.vq_model_hands.eval()
        self.vq_model_face.eval()

        self.proj_body = nn.Linear(BODY_CODEBOOK, VAE_LATENT).to(self.rank)
        self.proj_hands = nn.Linear(HAND_CODEBOOK, VAE_LATENT).to(self.rank)
        self.proj_face = nn.Linear(HAND_CODEBOOK, VAE_LATENT).to(self.rank)

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.cls_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)

        self.opt.add_param_group({
            'params': list(self.proj_body.parameters()) +
                      list(self.proj_hands.parameters()) +
                      list(self.proj_face.parameters())
        })

    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long().clamp(0, 24)
        bs, n, _ = tar_pose_raw.shape

        tar_pose_body = tar_pose_raw[:, :, BODY_CONT_SLICE[0]:BODY_CONT_SLICE[1]]
        tar_pose_hands = tar_pose_raw[:, :, HAND_SLICE[0]:HAND_SLICE[1]]
        tar_pose_rot6d = tar_pose_raw[:, :, GLOBAL_ROT6D_SLICE[0]:GLOBAL_ROT6D_SLICE[1]]
        tar_pose_cam_t = tar_pose_raw[:, :, CAM_T_SLICE[0]:CAM_T_SLICE[1]]
        tar_pose_contact = tar_pose_raw[:, :, CONTACT_SLICE[0]:CONTACT_SLICE[1]]
        tar_pose_face = torch.zeros(bs, n, 75, device=self.rank)

        tar_trans = dict_data["trans"].to(self.rank)
        tar_exps = torch.zeros(bs, n, 72, device=self.rank)
        tar_contact = tar_pose_contact

        tar_index_body = self.vq_model_body.map2index(tar_pose_body)
        tar_index_hands = self.vq_model_hands.map2index(tar_pose_hands)
        tar_index_face = self.vq_model_face.map2index(tar_pose_face)

        latent_body = self.vq_model_body.map2latent(tar_pose_body)
        latent_hands = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_face = self.vq_model_face.map2latent(tar_pose_face)
        latent_global = tar_pose_rot6d

        latent_in = torch.cat([latent_body, latent_hands, latent_global], dim=2)
        index_in = torch.stack([tar_index_body, tar_index_hands], dim=-1).long()
        latent_all = tar_pose_raw

        return {
            "tar_pose_face": tar_pose_face,
            "tar_pose_body": tar_pose_body,
            "tar_pose_hands": tar_pose_hands,
            "tar_pose_rot6d": tar_pose_rot6d,
            "tar_pose_cam_t": tar_pose_cam_t,
            "tar_pose_contact": tar_pose_contact,
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
        mask_val = torch.ones(bs, n, self.args.pose_dims).float().cuda()
        mask_val[:, :self.args.pre_frames, :] = 0.0

        net_out_val = self.model(
            loaded_data['in_audio'], loaded_data['in_word'], mask=mask_val,
            in_id=loaded_data['tar_id'], in_motion=loaded_data['latent_all'],
            use_attentions=True)
        g_loss_final = 0

        loss_latent_face = self.reclatent_loss(self.proj_face(net_out_val["rec_face"]), loaded_data["latent_face"])
        loss_latent_global = self.reclatent_loss(net_out_val["rec_global"], loaded_data["tar_pose_rot6d"])
        loss_latent_hands = self.reclatent_loss(self.proj_hands(net_out_val["rec_hands"]), loaded_data["latent_hands"])
        loss_latent_body = self.reclatent_loss(self.proj_body(net_out_val["rec_body"]), loaded_data["latent_body"])
        loss_latent = self.args.lf * loss_latent_face + self.args.ll * loss_latent_global + self.args.lh * loss_latent_hands + self.args.lu * loss_latent_body
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent

        rec_rot6d = net_out_val["rec_global"]
        tar_rot6d = loaded_data["tar_pose_rot6d"]
        loss_vel_global = self.vel_loss(rec_rot6d[:, 1:] - rec_rot6d[:, :-1], tar_rot6d[:, 1:] - tar_rot6d[:, :-1])
        rec_body_lat = self.proj_body(net_out_val["rec_body"])
        tar_body_lat = loaded_data["latent_body"]
        loss_vel_body = self.vel_loss(rec_body_lat[:, 1:] - rec_body_lat[:, :-1], tar_body_lat[:, 1:] - tar_body_lat[:, :-1])
        rec_hands_lat = self.proj_hands(net_out_val["rec_hands"])
        tar_hands_lat = loaded_data["latent_hands"]
        loss_vel_hands = self.vel_loss(rec_hands_lat[:, 1:] - rec_hands_lat[:, :-1], tar_hands_lat[:, 1:] - tar_hands_lat[:, :-1])
        loss_vel = self.args.vel_global_weight * loss_vel_global + self.args.vel_body_weight * loss_vel_body + self.args.vel_hands_weight * loss_vel_hands
        self.tracker.update_meter("vel", "train", loss_vel.item())
        g_loss_final += loss_vel

        rec_acc_global = rec_rot6d[:, 2:] - 2 * rec_rot6d[:, 1:-1] + rec_rot6d[:, :-2]
        tar_acc_global = tar_rot6d[:, 2:] - 2 * tar_rot6d[:, 1:-1] + tar_rot6d[:, :-2]
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

        rec_index_face_val = self.log_softmax(net_out_val["cls_face"]).reshape(-1, HAND_CODEBOOK)
        rec_index_body_val = self.log_softmax(net_out_val["cls_body"]).reshape(-1, BODY_CODEBOOK)
        rec_index_hands_val = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, HAND_CODEBOOK)
        tar_index_face = loaded_data["tar_index_face"].reshape(-1)
        tar_index_body = loaded_data["tar_index_body"].reshape(-1)
        tar_index_hands = loaded_data["tar_index_hands"].reshape(-1)
        loss_cls = self.args.cf * self.cls_loss(rec_index_face_val, tar_index_face) \
            + self.args.cu * self.cls_loss_smooth(net_out_val["cls_body"].reshape(-1, BODY_CODEBOOK), tar_index_body) \
            + self.args.ch * self.cls_loss(rec_index_hands_val, tar_index_hands)
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

            loss_latent_self = self.args.lf * self.reclatent_loss(self.proj_face(net_out_self["rec_face"]), loaded_data["latent_face"]) \
                + self.args.ll * self.reclatent_loss(net_out_self["rec_global"], loaded_data["tar_pose_rot6d"]) \
                + self.args.lh * self.reclatent_loss(self.proj_hands(net_out_self["rec_hands"]), loaded_data["latent_hands"]) \
                + self.args.lu * self.reclatent_loss(self.proj_body(net_out_self["rec_body"]), loaded_data["latent_body"])
            self.tracker.update_meter("latent_self", "train", loss_latent_self.item())
            g_loss_final += loss_latent_self
            index_loss_self = self.cls_loss_smooth(net_out_self["cls_body"].reshape(-1, BODY_CODEBOOK), tar_index_body) \
                + self.cls_loss(self.log_softmax(net_out_self["cls_hands"]).reshape(-1, HAND_CODEBOOK), tar_index_hands)
            self.tracker.update_meter("cls_self", "train", index_loss_self.item())
            g_loss_final += index_loss_self

            net_out_word = self.model(
                loaded_data['in_audio'], loaded_data['in_word'], mask=mask,
                in_id=loaded_data['tar_id'], in_motion=loaded_data['latent_all'],
                use_attentions=True, use_word=True)

            loss_latent_word = self.args.lf * self.reclatent_loss(self.proj_face(net_out_word["rec_face"]), loaded_data["latent_face"]) \
                + self.args.ll * self.reclatent_loss(net_out_word["rec_global"], loaded_data["tar_pose_rot6d"]) \
                + self.args.lh * self.reclatent_loss(self.proj_hands(net_out_word["rec_hands"]), loaded_data["latent_hands"]) \
                + self.args.lu * self.reclatent_loss(self.proj_body(net_out_word["rec_body"]), loaded_data["latent_body"])
            self.tracker.update_meter("latent_word", "train", loss_latent_word.item())
            g_loss_final += loss_latent_word
            index_loss_word = self.cls_loss_smooth(net_out_word["cls_body"].reshape(-1, BODY_CODEBOOK), tar_index_body) \
                + self.cls_loss(self.log_softmax(net_out_word["cls_hands"]).reshape(-1, HAND_CODEBOOK), tar_index_hands)
            self.tracker.update_meter("cls_word", "train", index_loss_word.item())
            g_loss_final += index_loss_word

        if mode != 'train':
            if self.args.cu != 0:
                _, rec_idx_body = torch.max(rec_index_body_val.reshape(-1, self.args.pose_length, BODY_CODEBOOK), dim=2)
                rec_body = self.vq_model_body.decode(rec_idx_body)
            else:
                _, rec_idx_body, _, _ = self.vq_model_body.quantizer(net_out_val["rec_body"])
                rec_body = self.vq_model_body.decoder(rec_idx_body)
            if self.args.ch != 0:
                _, rec_idx_hands = torch.max(rec_index_hands_val.reshape(-1, self.args.pose_length, HAND_CODEBOOK), dim=2)
                rec_hands = self.vq_model_hands.decode(rec_idx_hands)
            else:
                _, rec_idx_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
                rec_hands = self.vq_model_hands.decoder(rec_idx_hands)
            rec_rot6d_val = net_out_val["rec_global"]
            rec_pose = torch.cat([rec_body, rec_hands, rec_rot6d_val, loaded_data["tar_pose_cam_t"], loaded_data["tar_pose_contact"]], dim=-1)

        if mode == 'train':
            return g_loss_final
        elif mode == 'val':
            return {'rec_pose': rec_pose, 'tar_pose': loaded_data["tar_pose"]}
        else:
            return {
                'rec_pose': rec_pose, 'tar_pose': loaded_data["tar_pose"],
                'tar_exps': loaded_data["tar_exps"], 'tar_beta': loaded_data["tar_beta"],
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

        latent_all = tar_pose
        rec_index_all_body = []
        rec_index_all_hands = []
        rec_rot6d_all = []
        rec_cam_t_all = []
        rec_contact_all = []

        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        for i in range(roundt):
            s_e = i * round_l
            e_e = (i + 1) * round_l + self.args.pre_frames
            in_word_tmp = in_word[:, s_e:e_e]
            in_audio_tmp = in_audio[:, i * (16000 // 30 * round_l):(i + 1) * (16000 // 30 * round_l) + 16000 // 30 * self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, s_e:e_e]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0

            if i == 0:
                latent_all_tmp = latent_all[:, s_e:e_e, :].clone()
            else:
                latent_all_tmp = latent_all[:, s_e:e_e, :].clone()
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]

            net_out_val = self.model(
                in_audio=in_audio_tmp, in_word=in_word_tmp, mask=mask_val,
                in_motion=latent_all_tmp, in_id=in_id_tmp, use_attentions=True)

            if self.args.cu != 0:
                rec_idx_body = self.log_softmax(net_out_val["cls_body"]).reshape(-1, BODY_CODEBOOK)
                _, rec_idx_body = torch.max(rec_idx_body.reshape(-1, self.args.pose_length, BODY_CODEBOOK), dim=2)
            else:
                _, rec_idx_body, _, _ = self.vq_model_body.quantizer(net_out_val["rec_body"])
            if self.args.ch != 0:
                rec_idx_hands = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, HAND_CODEBOOK)
                _, rec_idx_hands = torch.max(rec_idx_hands.reshape(-1, self.args.pose_length, HAND_CODEBOOK), dim=2)
            else:
                _, rec_idx_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])

            rec_rot6d = net_out_val["rec_global"]

            gt_cam_t_chunk = tar_pose[:, s_e:e_e, CAM_T_SLICE[0]:CAM_T_SLICE[1]]
            gt_contact_chunk = tar_pose[:, s_e:e_e, CONTACT_SLICE[0]:CONTACT_SLICE[1]]

            if i == 0:
                rec_index_all_body.append(rec_idx_body)
                rec_index_all_hands.append(rec_idx_hands)
                rec_rot6d_all.append(rec_rot6d)
                rec_cam_t_all.append(gt_cam_t_chunk)
                rec_contact_all.append(gt_contact_chunk)
            else:
                rec_index_all_body.append(rec_idx_body[:, self.args.pre_frames:])
                rec_index_all_hands.append(rec_idx_hands[:, self.args.pre_frames:])
                rec_rot6d_all.append(rec_rot6d[:, self.args.pre_frames:])
                rec_cam_t_all.append(gt_cam_t_chunk[:, self.args.pre_frames:])
                rec_contact_all.append(gt_contact_chunk[:, self.args.pre_frames:])

            rec_body_last = self.vq_model_body.decode(rec_idx_body) if self.args.cu != 0 else self.vq_model_body.decoder(rec_idx_body)
            rec_hands_last = self.vq_model_hands.decode(rec_idx_hands) if self.args.ch != 0 else self.vq_model_hands.decoder(rec_idx_hands)
            latent_last = torch.cat([rec_body_last, rec_hands_last, rec_rot6d, gt_cam_t_chunk, gt_contact_chunk], dim=-1)

        rec_index_body = torch.cat(rec_index_all_body, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)
        rec_rot6d_final = torch.cat(rec_rot6d_all, dim=1)
        rec_cam_t_final = torch.cat(rec_cam_t_all, dim=1)
        rec_contact_final = torch.cat(rec_contact_all, dim=1)

        rec_body = self.vq_model_body.decode(rec_index_body) if self.args.cu != 0 else self.vq_model_body.decoder(rec_index_body)
        rec_hands = self.vq_model_hands.decode(rec_index_hands) if self.args.ch != 0 else self.vq_model_hands.decoder(rec_index_hands)

        from scipy.signal import savgol_filter
        rec_rot6d_np = rec_rot6d_final.detach().cpu().numpy()
        for b in range(rec_rot6d_np.shape[0]):
            rec_rot6d_np[b] = savgol_filter(rec_rot6d_np[b], window_length=5, polyorder=2, axis=0)
        rec_rot6d_final = torch.from_numpy(rec_rot6d_np).to(rec_rot6d_final.device)

        rec_pose = torch.cat([rec_body, rec_hands, rec_rot6d_final, rec_cam_t_final, rec_contact_final], dim=-1)
        n = rec_pose.shape[1]

        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        return {
            'rec_pose': rec_pose,
            'rec_trans': tar_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
        }

    def _cont_to_mhr_npz(self, cont_np):
        body_cont = torch.from_numpy(cont_np[:, :260]).float()
        body_euler = compact_cont_to_model_params_body(body_cont).numpy()
        hand_pca = cont_np[:, 260:368]
        global_rot_6d = torch.from_numpy(cont_np[:, 368:374]).float()
        global_rot_euler = rot6d_to_euler3(global_rot_6d).numpy()
        cam_t = cont_np[:, 374:377]
        expr = np.zeros((cont_np.shape[0], 72), dtype=np.float32)
        return body_euler, hand_pca, expr, global_rot_euler, cam_t

    def train(self, epoch):
        use_adv = bool(epoch >= self.args.no_adv_epoch)
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
            self.opt.zero_grad()
            g_loss_final = self._g_training(loaded_data, use_adv, 'train', epoch)
            g_loss_final.backward()
            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.args.debug and its == 1:
                break
        self.opt_s.step(epoch)

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_training(loaded_data, False, 'val', epoch)
                if self.args.debug and its == 1:
                    break
        self.val_recording(epoch)

    def test(self, epoch):
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
                bs, n_out = rec_pose.shape[0], rec_pose.shape[1]

                rec_np = rec_pose.detach().cpu().numpy().reshape(bs * n_out, -1)

                gt_npz = np.load(self.args.data_path + self.args.pose_rep + "/" + test_seq_list.iloc[its]['id'] + ".npz", allow_pickle=True)
                shape = gt_npz['shape_params'] if 'shape_params' in gt_npz.files else np.zeros(45, dtype=np.float32)
                scale_params = gt_npz['scale_params'][0] if 'scale_params' in gt_npz.files else np.zeros(28, dtype=np.float32)
                focal_length = float(gt_npz['focal_length'][0]) if 'focal_length' in gt_npz.files else 1500.0
                orig_width = int(gt_npz['width'][0]) if 'width' in gt_npz.files else 1080
                orig_height = int(gt_npz['height'][0]) if 'height' in gt_npz.files else 1920

                body_euler, hand_pca, expr, global_rot, cam_t = self._cont_to_mhr_npz(rec_np)

                np.savez(results_save_path + "res_" + test_seq_list.iloc[its]['id'] + '.npz',
                         body_pose_params=body_euler,
                         hand_pose_params=hand_pca,
                         expr_params=expr,
                         global_rot=global_rot,
                         pred_cam_t=cam_t,
                         shape_params=shape, scale_params=scale_params,
                         focal_length=np.array([focal_length]),
                         width=np.array([orig_width]), height=np.array([orig_height]),
                         mocap_frame_rate=30)
                total_length += n_out

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length / self.args.pose_fps)} s motion")
