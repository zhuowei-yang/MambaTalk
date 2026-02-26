import copy
import pickle
import torch
import torch.nn as nn

from .motion_encoder import *
from .layers import *
from .mamba_block import MambaModel, MambaScan
from .mambatalk import GlobalScan, LocalScan


class DualMambaTalk(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_face = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_face = nn.Linear(300, args.audio_f)
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f)

        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=3)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=2)
        self.at_attn_face = nn.Linear(args.audio_f * 2, args.audio_f * 2)
        self.at_attn_body = nn.Linear(args.audio_f * 2, args.audio_f * 2)

        self.global_scan = GlobalScan(args)
        self.local_scan = LocalScan(args)

        self.speaker_encoder_face = nn.Embedding(25, args.hidden_size)
        self.speaker_encoder_body = nn.Embedding(25, args.hidden_size)

        # speaker1 conditioning
        self.cond_audio_encoder = WavEncoder(args.audio_f, audio_in=2)
        args_cond = copy.deepcopy(args)
        args_cond.vae_layer = 3
        args_cond.vae_length = args.motion_f
        args_cond.vae_test_dim = args.pose_dims
        self.cond_motion_encoder = VQEncoderV6(args_cond)
        self.cond_proj = nn.Linear(args.motion_f + args.audio_f, args.hidden_size)

    def forward(self, in_audio=None, in_word=None, mask=None, in_id=None,
                in_motion=None, cond_audio=None, cond_motion=None,
                use_attentions=True, use_word=True):
        """
        Args:
            in_audio: speaker2's audio (bs, audio_samples, 3)
            in_word: speaker2's word indices (bs, T)
            mask: (bs, T, 320) generation mask
            in_id: speaker2's speaker ID (bs, T, 1)
            in_motion: speaker2's motion for autoregressive (bs, T, 320)
            cond_audio: speaker1's audio (bs, audio_samples, 3)
            cond_motion: speaker1's motion (bs, T, 320)
        Returns:
            dict with rec_face, rec_body, rec_hands, rec_global,
            cls_face, cls_body, cls_hands
        """
        in_word_face = self.text_pre_encoder_face(in_word)
        in_word_face = self.text_encoder_face(in_word_face)
        in_word_body = self.text_pre_encoder_body(in_word)
        in_word_body = self.text_encoder_body(in_word_body)
        bs, t, c = in_word_face.shape

        in_audio_face = self.audio_pre_encoder_face(in_audio)
        in_audio_body = self.audio_pre_encoder_body(in_audio[..., :2])

        if in_audio_face.shape[1] != in_motion.shape[1]:
            diff_length = in_motion.shape[1] - in_audio_face.shape[1]
            if diff_length < 0:
                in_audio_face = in_audio_face[:, :diff_length, :]
                in_audio_body = in_audio_body[:, :diff_length, :]
            else:
                in_audio_face = torch.cat((in_audio_face, in_audio_face[:, -diff_length:]), 1)
                in_audio_body = torch.cat((in_audio_body, in_audio_body[:, -diff_length:]), 1)

        alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c * 2)
        alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2)
        alpha_at_face = alpha_at_face.softmax(dim=-1)
        fusion_face = in_word_face * alpha_at_face[:, :, :, 1] + in_audio_face * alpha_at_face[:, :, :, 0]
        alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c * 2)
        alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        alpha_at_body = alpha_at_body.softmax(dim=-1)
        fusion_body = in_word_body * alpha_at_body[:, :, :, 1] + in_audio_body * alpha_at_body[:, :, :, 0]

        speaker_embedding_face = self.speaker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.speaker_encoder_body(in_id).squeeze(2)

        # speaker1 conditioning
        cond_audio_feat = self.cond_audio_encoder(cond_audio[..., :2])
        if cond_audio_feat.shape[1] != cond_motion.shape[1]:
            diff = cond_motion.shape[1] - cond_audio_feat.shape[1]
            if diff < 0:
                cond_audio_feat = cond_audio_feat[:, :diff, :]
            else:
                cond_audio_feat = torch.cat((cond_audio_feat, cond_audio_feat[:, -diff:]), 1)
        cond_motion_feat = self.cond_motion_encoder(cond_motion)
        cond_feat = self.cond_proj(torch.cat([cond_motion_feat, cond_audio_feat], dim=-1))

        global_motions, global_features = self.global_scan(
            in_motion, mask, speaker_embedding_face, speaker_embedding_body, fusion_face)
        global_features = global_features + cond_feat

        output = self.local_scan(
            global_motions, global_features,
            speaker_embedding_face, speaker_embedding_body, fusion_body, use_word)

        return output
