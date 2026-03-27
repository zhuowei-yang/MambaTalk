"""
DualMambaTalkV8MultiLevel -- Phase 3 model with LocalScan multi-level injection.

Extends DualMambaTalkV8 by injecting cond_feat into LocalScan's
body/hands/global branches, providing fine-grained partner conditioning
beyond the single fusion point at global_features.
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


class LocalScanMultiLevel(LocalScan):
    """LocalScan with additional cond_feat injection into body/hands/global branches."""

    def __init__(self, args):
        super().__init__(args)
        hs = args.hidden_size
        self.body_scan_proj = nn.Linear(2 * hs + hs, hs)
        self.hands_proj = nn.Linear(2 * hs + hs, hs)
        self.global_proj = nn.Linear(2 * hs + hs, hs)
        self.cond_local_proj = nn.Linear(hs, hs)

    def forward(self, global_motions, global_features, face_latent_in,
                speaker_embedding_body, body_features, use_word,
                cond_feat_local=None):
        motion_refined_embeddings_in = global_motions + speaker_embedding_body
        motion_refined_embeddings_in = self.position_embeddings(global_motions)
        learnable_querys = self.motion_cross_attn(
            tgt=motion_refined_embeddings_in,
            memory=self.body_audio_proj(body_features))
        global_motions = global_motions + learnable_querys

        body_latent = self.motion2latent_body(global_motions)
        hands_latent = self.motion2latent_hands(global_motions)
        global_latent = self.motion2latent_global(global_motions)

        body_latent_in = body_latent + speaker_embedding_body
        hands_latent_in = hands_latent + speaker_embedding_body
        global_latent_in = global_latent + speaker_embedding_body

        body_latent_in = self.position_embeddings(body_latent_in)
        hands_latent_in = self.position_embeddings(hands_latent_in)
        global_latent_in = self.position_embeddings(global_latent_in)
        face_latent_in = self.position_embeddings(face_latent_in)

        if cond_feat_local is not None:
            cond_local = self.cond_local_proj(cond_feat_local)
        else:
            cond_local = torch.zeros_like(body_latent_in)

        decoded_face = self.face_scan(
            self.face_proj(torch.cat([face_latent_in, global_features], dim=-1)))
        motion_body = self.body_scan(
            self.body_scan_proj(torch.cat([
                body_latent_in, hands_latent + global_latent, cond_local], dim=-1)))
        motion_hands = self.hands_scan(
            self.hands_proj(torch.cat([
                hands_latent_in, body_latent + global_latent, cond_local], dim=-1)))
        motion_global = self.global_scan(
            self.global_proj(torch.cat([
                global_latent_in, body_latent + hands_latent, cond_local], dim=-1)))

        face_latent = self.motion_down_face(decoded_face)
        body_latent = self.motion_down_body(motion_body + body_latent)
        hands_latent = self.motion_down_hands(motion_hands + hands_latent)
        global_latent = self.motion_down_global(motion_global + global_latent)

        cls_face = self.face_classifier(face_latent)
        cls_body = self.body_classifier(body_latent)
        cls_hands = self.hands_classifier(hands_latent)

        return {
            "rec_face": face_latent,
            "rec_body": body_latent,
            "rec_hands": hands_latent,
            "rec_global": global_latent,
            "cls_face": cls_face,
            "cls_body": cls_body,
            "cls_hands": cls_hands,
        }


class DualMambaTalkV8MultiLevel(nn.Module):
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

        self.speaker_encoder_face = nn.Embedding(25, args.hidden_size)
        self.speaker_encoder_body = nn.Embedding(25, args.hidden_size)

        self.global_scan = GlobalScan(args)
        self.local_scan = LocalScanMultiLevel(args)

        self.cond_audio_encoder = WavEncoder(args.audio_f, audio_in=2)
        args_cond = copy.deepcopy(args)
        args_cond.vae_layer = 3
        args_cond.vae_length = args.motion_f
        args_cond.vae_test_dim = args.pose_dims
        self.cond_motion_encoder = VQEncoderV6(args_cond)
        self.cond_proj = nn.Linear(args.motion_f + args.audio_f, args.hidden_size)

        self.state_filter = StateSignalFilter()
        self.state_embedding = StateEmbedding(embed_dim=args.hidden_size)
        self.interaction_fusion = InteractionFusion(
            hidden_size=args.hidden_size, n_heads=4,
            n_cross_layers=1, dropout=0.1)

        pd_nc = getattr(args, 'pd_n_channels', 0)
        self.use_periodic = pd_nc > 0
        if self.use_periodic:
            n_fl = getattr(args, 'pd_n_filter_layers', 2)
            logger.info(f"DualMambaTalkV8MultiLevel: FreqPeriodicPredictor {pd_nc}ch, {n_fl} layers")
            self.periodic_predictor = FreqPeriodicPredictor(
                input_dim=args.hidden_size, hidden_dim=256, n_channels=pd_nc,
                n_experts=4, n_filter_layers=n_fl,
                max_T=args.pose_length * 2, dropout=0.1)
            self.periodic_mapper = PeriodicMapper(
                n_channels=pd_nc, body_dim=260, hand_dim=108)

    def _align_audio(self, audio_feat, target_len):
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
        in_word_face = self.text_encoder_face(self.text_pre_encoder_face(in_word))
        in_word_body = self.text_encoder_body(self.text_pre_encoder_body(in_word))
        bs, t, c = in_word_face.shape

        in_audio_face = self._align_audio(self.audio_pre_encoder_face(in_audio), t)
        in_audio_body = self._align_audio(self.audio_pre_encoder_body(in_audio[..., :2]), t)

        alpha_face = self.at_attn_face(
            torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c * 2)
        ).reshape(bs, t, c, 2).softmax(dim=-1)
        fusion_face = in_word_face * alpha_face[..., 1] + in_audio_face * alpha_face[..., 0]

        alpha_body = self.at_attn_body(
            torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c * 2)
        ).reshape(bs, t, c, 2).softmax(dim=-1)
        fusion_body = in_word_body * alpha_body[..., 1] + in_audio_body * alpha_body[..., 0]

        speaker_embedding_face = self.speaker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.speaker_encoder_body(in_id).squeeze(2)

        global_motions, global_features = self.global_scan(
            in_motion, mask, speaker_embedding_face,
            speaker_embedding_body, fusion_face)

        cond_audio_feat = self._align_audio(
            self.cond_audio_encoder(cond_audio[..., :2]), t)
        cond_motion_feat = self.cond_motion_encoder(cond_motion)
        cond_feat = self.cond_proj(
            torch.cat([cond_motion_feat, cond_audio_feat], dim=-1))

        if state_ids is not None:
            filtered_states = self.state_filter(state_ids)
            state_embed = self.state_embedding(filtered_states, t)
        else:
            state_embed = torch.zeros(bs, t, self.args.hidden_size,
                                      device=global_features.device)

        if use_attentions:
            global_features, sigma_weights = self.interaction_fusion(
                global_features, cond_feat, state_embed, return_sigma=True)
        else:
            global_features = self.interaction_fusion(
                global_features, cond_feat, state_embed)
            sigma_weights = None

        output = self.local_scan(
            global_motions, global_features,
            speaker_embedding_face, speaker_embedding_body,
            fusion_body, use_word,
            cond_feat_local=cond_feat)

        if self.use_periodic:
            pd_out = self.periodic_predictor(global_motions)
            body_pe, hand_pe = self.periodic_mapper(pd_out["periodic_signal"])
            output["periodic_params"] = {k: pd_out[k] for k in ("A", "F", "B", "S")}
            output["periodic_body"] = body_pe
            output["periodic_hands"] = hand_pe

        if sigma_weights is not None:
            output["sigma_weights"] = sigma_weights

        return output
