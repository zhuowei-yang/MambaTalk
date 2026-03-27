"""
DualMambaTalkV8 -- dual-person gesture generation model.

Extends the single-person MambaTalk (v8) with:
  - Partner conditioning (audio + motion) from speaker1
  - SoulX-Duplug dialogue-state embedding
  - InteractionFusion (tri-path cross-attention) replacing v2 additive injection
  - v8 FreqPeriodicPredictor / PeriodicMapper preserved
"""
import copy
import math
import pickle
import torch
import torch.nn as nn
from loguru import logger

from .motion_encoder import VQEncoderV6
from .layers import WavEncoder, MLP, PeriodicPositionalEncoding
from .mambatalk_mhr import GlobalScan, LocalScan, FreqPeriodicPredictor, PeriodicMapper
from .interaction_modules import StateSignalFilter, StateEmbedding, InteractionFusion


class DualMambaTalkV8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---------- text encoders (speaker2, target) ----------
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_face = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_face = nn.Linear(300, args.audio_f)
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f)

        # ---------- audio encoders (speaker2) ----------
        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=3)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=2)
        self.at_attn_face = nn.Linear(args.audio_f * 2, args.audio_f * 2)
        self.at_attn_body = nn.Linear(args.audio_f * 2, args.audio_f * 2)

        # ---------- speaker id embeddings ----------
        self.speaker_encoder_face = nn.Embedding(25, args.hidden_size)
        self.speaker_encoder_body = nn.Embedding(25, args.hidden_size)

        # ---------- core scan modules (shared with v8) ----------
        self.global_scan = GlobalScan(args)
        self.local_scan = LocalScan(args)

        # ---------- speaker1 condition encoders (from v2) ----------
        self.cond_audio_encoder = WavEncoder(args.audio_f, audio_in=2)
        args_cond = copy.deepcopy(args)
        args_cond.vae_layer = 3
        args_cond.vae_length = args.motion_f
        args_cond.vae_test_dim = args.pose_dims
        self.cond_motion_encoder = VQEncoderV6(args_cond)
        self.cond_proj = nn.Linear(args.motion_f + args.audio_f, args.hidden_size)

        # ---------- dialogue state modules (new) ----------
        self.state_filter = StateSignalFilter()
        self.state_embedding = StateEmbedding(embed_dim=args.hidden_size)

        # ---------- interaction fusion (new, replaces v2 additive) ----------
        self.interaction_fusion = InteractionFusion(
            hidden_size=args.hidden_size, n_heads=4,
            n_cross_layers=1, dropout=0.1,
        )

        # ---------- v8: periodic enhancement ----------
        pd_nc = getattr(args, 'pd_n_channels', 0)
        self.use_periodic = pd_nc > 0
        if self.use_periodic:
            n_fl = getattr(args, 'pd_n_filter_layers', 2)
            logger.info(f"DualMambaTalkV8: v8 FreqPeriodicPredictor with {pd_nc} channels, "
                        f"{n_fl} FMLP filter layers")
            self.periodic_predictor = FreqPeriodicPredictor(
                input_dim=args.hidden_size, hidden_dim=256, n_channels=pd_nc,
                n_experts=4, n_filter_layers=n_fl,
                max_T=args.pose_length * 2, dropout=0.1)
            self.periodic_mapper = PeriodicMapper(
                n_channels=pd_nc, body_dim=260, hand_dim=108)

    def _align_audio(self, audio_feat, target_len):
        """Pad or trim audio features to match pose sequence length."""
        if audio_feat.shape[1] == target_len:
            return audio_feat
        diff = target_len - audio_feat.shape[1]
        if diff < 0:
            return audio_feat[:, :diff, :]
        return torch.cat([audio_feat, audio_feat[:, -diff:, :]], dim=1)

    def forward(self, in_audio=None, in_word=None, mask=None,
                in_motion=None, in_id=None,
                cond_audio=None, cond_motion=None,
                state_ids=None,
                use_attentions=True, use_word=True):
        """
        Args:
            in_audio:    (B, T_audio, 3) speaker2 audio
            in_word:     (B, T) speaker2 word token indices
            mask:        (B, T, pose_dims)
            in_motion:   (B, T, 381) speaker2 motion (autoregressive input)
            in_id:       (B, T, 1) or (B,) speaker2 id
            cond_audio:  (B, T_audio, 3) speaker1 audio
            cond_motion: (B, T, 381) speaker1 motion
            state_ids:   (B, T_state) dialogue state ids [0-4]
            use_word:    whether to use text conditioning
        """
        # ===== Speaker2 text + audio encoding =====
        in_word_face = self.text_encoder_face(self.text_pre_encoder_face(in_word))
        in_word_body = self.text_encoder_body(self.text_pre_encoder_body(in_word))
        bs, t, c = in_word_face.shape

        in_audio_face = self._align_audio(
            self.audio_pre_encoder_face(in_audio), t)
        in_audio_body = self._align_audio(
            self.audio_pre_encoder_body(in_audio[..., :2]), t)

        # gate fusion (audio + text)
        alpha_face = self.at_attn_face(
            torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c * 2)
        ).reshape(bs, t, c, 2).softmax(dim=-1)
        fusion_face = in_word_face * alpha_face[..., 1] + in_audio_face * alpha_face[..., 0]

        alpha_body = self.at_attn_body(
            torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c * 2)
        ).reshape(bs, t, c, 2).softmax(dim=-1)
        fusion_body = in_word_body * alpha_body[..., 1] + in_audio_body * alpha_body[..., 0]

        # speaker embedding
        speaker_embedding_face = self.speaker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.speaker_encoder_body(in_id).squeeze(2)

        # ===== GlobalScan (speaker2) =====
        global_motions, global_features = self.global_scan(
            in_motion, mask, speaker_embedding_face,
            speaker_embedding_body, fusion_face)

        # ===== Speaker1 condition encoding =====
        cond_audio_feat = self._align_audio(
            self.cond_audio_encoder(cond_audio[..., :2]), t)
        cond_motion_feat = self.cond_motion_encoder(cond_motion)
        cond_feat = self.cond_proj(
            torch.cat([cond_motion_feat, cond_audio_feat], dim=-1))

        # ===== Dialogue state embedding =====
        if state_ids is not None:
            filtered_states = self.state_filter(state_ids)
            state_embed = self.state_embedding(filtered_states, t)
        else:
            state_embed = torch.zeros(bs, t, self.args.hidden_size,
                                      device=global_features.device)

        # ===== InteractionFusion (replaces v2 additive) =====
        if use_attentions:
            global_features, sigma_weights = self.interaction_fusion(
                global_features, cond_feat, state_embed, return_sigma=True)
        else:
            global_features = self.interaction_fusion(
                global_features, cond_feat, state_embed)
            sigma_weights = None

        # ===== LocalScan =====
        output = self.local_scan(
            global_motions, global_features,
            speaker_embedding_face, speaker_embedding_body,
            fusion_body, use_word)

        # ===== v8 periodic enhancement =====
        if self.use_periodic:
            pd_out = self.periodic_predictor(global_motions)
            body_pe, hand_pe = self.periodic_mapper(pd_out["periodic_signal"])
            output["periodic_params"] = {k: pd_out[k] for k in ("A", "F", "B", "S")}
            output["periodic_body"] = body_pe
            output["periodic_hands"] = hand_pe

        if sigma_weights is not None:
            output["sigma_weights"] = sigma_weights

        return output
