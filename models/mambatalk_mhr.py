import copy
import math
import pickle
import torch
import torch.nn as nn
from loguru import logger

from .motion_encoder import * 
from .layers import *
from .mamba_block import MambaModel, MambaScan


class FreqFilterBlock(nn.Module):
    """FMLP filter block: learnable frequency filter + FFN + Add&Norm."""

    def __init__(self, dim, max_T=128, dropout=0.1):
        super().__init__()
        freq_bins = max_T // 2 + 1
        self.weight_real = nn.Parameter(torch.ones(freq_bins, dim))
        self.weight_imag = nn.Parameter(torch.zeros(freq_bins, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        K = T // 2 + 1
        learned_bins = self.weight_real.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()
            Y = torch.fft.rfft(x_f32, dim=1)
            if K <= learned_bins:
                W = torch.complex(self.weight_real[:K], self.weight_imag[:K])
            else:
                wr_pad = torch.ones(K - learned_bins, D, device=x.device)
                wi_pad = torch.zeros(K - learned_bins, D, device=x.device)
                W = torch.complex(
                    torch.cat([self.weight_real, wr_pad], dim=0),
                    torch.cat([self.weight_imag, wi_pad], dim=0),
                )
            Y_filtered = W.unsqueeze(0) * Y
            x_filtered = torch.fft.irfft(Y_filtered, n=T, dim=1)

        x_filtered = x_filtered.to(x.dtype)
        x = self.norm1(x + self.dropout(x_filtered))
        x = self.norm2(x + self.ffn(x))
        return x


class FreqPeriodicPredictor(nn.Module):
    """
    FMLP-enhanced periodic parameter predictor.
    Uses learnable frequency filter layers (replacing LSTM) for
    frequency-aware temporal modeling, with MoE prediction head.
    Read-only on input: does NOT modify global_motions.
    """

    def __init__(self, input_dim=768, hidden_dim=256, n_channels=10,
                 n_experts=4, n_filter_layers=2, max_T=128, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_experts = n_experts

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.filter_layers = nn.ModuleList([
            FreqFilterBlock(hidden_dim, max_T, dropout)
            for _ in range(n_filter_layers)
        ])
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_experts), nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_channels * 3),
            ) for _ in range(n_experts)
        ])
        self.fc_phase = nn.Linear(hidden_dim, n_channels * 2)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.input_proj(x)
        for fl in self.filter_layers:
            h = fl(h)

        gate_weights = self.gate(h)
        expert_outs = torch.stack([e(h) for e in self.experts], dim=-1)
        blended = (expert_outs * gate_weights.unsqueeze(2)).sum(dim=-1)

        h_mid = h[:, T // 2, :]
        afb_mid = blended[:, T // 2, :]
        A = afb_mid[:, :self.n_channels].abs()
        F_val = afb_mid[:, self.n_channels:2*self.n_channels].sigmoid() * 0.5
        B_val = afb_mid[:, 2*self.n_channels:]

        sx_sy = self.fc_phase(h_mid).view(B, self.n_channels, 2)
        S = torch.atan2(sx_sy[..., 1], sx_sy[..., 0])

        t_axis = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
        periodic_signal = (
            A.unsqueeze(1) *
            torch.sin(2 * math.pi * (F_val.unsqueeze(1) * t_axis - S.unsqueeze(1)))
            + B_val.unsqueeze(1)
        )
        return {"A": A, "F": F_val, "B": B_val, "S": S, "periodic_signal": periodic_signal}


class PeriodicMapper(nn.Module):
    def __init__(self, n_channels=10, body_dim=260, hand_dim=108):
        super().__init__()
        self.body_map = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, body_dim, kernel_size=3, padding=1),
        )
        self.hand_map = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, hand_dim, kernel_size=3, padding=1),
        )

    def forward(self, periodic_signal):
        x = periodic_signal.transpose(1, 2)
        body_pe = self.body_map(x).transpose(1, 2)
        hand_pe = self.hand_map(x).transpose(1, 2)
        return body_pe, hand_pe


class GlobalScan(nn.Module):
    def __init__(self, args):
        super(GlobalScan, self).__init__()
        self.args = args
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f
        args_top.vae_test_dim = args.pose_dims
        self.learnable_query = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims))
        self.motion_encoder = VQEncoderV6(args_top)
        self.motion_proj1 = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.motion_proj2 = nn.Linear(args.motion_f, args.hidden_size)

        # self-attention
        self.position_embeddings = PeriodicPositionalEncoding(args.hidden_size, period=args.pose_length, max_seq_len=args.pose_length)
        self.face_q_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        self.body_q_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        self.motion_self_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        
        # scan
        self.face_scan = MambaScan(in_dim=args.audio_f, d_intermediate=args.audio_f * 2, out_dim=args.audio_f, n_layer=1)
        self.body_scan = MambaScan(in_dim=args.motion_f, d_intermediate=args.hidden_size, out_dim=args.motion_f, n_layer=1)        
        self.out = nn.Linear(args.audio_f + args.motion_f, args.hidden_size)

    def forward(self, in_motion, mask, speaker_embedding_face, speaker_embedding_body, face_features):
        # learnable query
        learnable_query = self.learnable_query.expand_as(in_motion)
        learnable_query = torch.where(mask == 1, learnable_query, in_motion)
        body_features = self.motion_encoder(learnable_query)
        motion_features = self.motion_proj2(self.motion_proj1(body_features))
        
        # self-attention
        speaker_embedding_face = self.face_q_self_attn(speaker_embedding_face)
        speaker_embedding_body = self.body_q_self_attn(speaker_embedding_body)
        motion_features = speaker_embedding_body + motion_features
        motion_features = self.position_embeddings(motion_features)
        global_motion = self.motion_self_encoder(motion_features)         
        
        # scan
        face_features = self.face_scan(face_features)
        body_features = self.body_scan(body_features)
        global_features = self.out(torch.cat([face_features, body_features], dim=2))
        
        return global_motion, global_features


class LocalScan(nn.Module):
    def __init__(self, args):
        super(LocalScan, self).__init__()
        self.args = args

        # cross-attention
        self.position_embeddings = PeriodicPositionalEncoding(args.hidden_size, period=args.pose_length, max_seq_len=args.pose_length)
        self.motion_cross_attn = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=8)
        self.motion2latent_body = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.motion2latent_global = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.face_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.body_scan_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.hands_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.global_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.body_audio_proj = nn.Linear(args.audio_f, args.hidden_size)

        # scan
        self.face_scan = MambaScan(in_dim=args.hidden_size, d_intermediate=args.hidden_size, out_dim=args.hidden_size, n_layer=4)
        self.body_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)
        self.hands_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)
        self.global_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)

        # decoder
        body_cb = getattr(args, 'body_codebook_size', args.vae_codebook_size)
        hand_cb = getattr(args, 'hand_codebook_size', args.vae_codebook_size)
        self.motion_down_face = nn.Linear(args.hidden_size, hand_cb)
        self.motion_down_body = nn.Linear(args.hidden_size, body_cb)
        self.motion_down_hands = nn.Linear(args.hidden_size, hand_cb)
        self.motion_down_global = nn.Linear(args.hidden_size, args.global_dims)  # raw 10d output
        self.face_classifier = MLP(hand_cb, args.hidden_size, hand_cb)
        self.body_classifier = MLP(body_cb, args.hidden_size, body_cb)
        self.hands_classifier = MLP(hand_cb, args.hidden_size, hand_cb)

    def forward(self, global_motions, global_features, face_latent_in, speaker_embedding_body, body_features, use_word):
        # cross attn
        motion_refined_embeddings_in = global_motions + speaker_embedding_body
        motion_refined_embeddings_in = self.position_embeddings(global_motions)        
        learnable_querys = self.motion_cross_attn(tgt=motion_refined_embeddings_in, memory=self.body_audio_proj(body_features))
        global_motions = global_motions + learnable_querys
        
        # feedforward
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
        
        # local scan
        decoded_face = self.face_scan(self.face_proj(torch.cat([face_latent_in, global_features], dim=-1)))          
        motion_body = self.body_scan(self.body_scan_proj(torch.cat([body_latent_in, hands_latent+global_latent], dim=-1)))
        motion_hands = self.hands_scan(self.hands_proj(torch.cat([hands_latent_in, body_latent+global_latent], dim=-1)))
        motion_global = self.global_scan(self.global_proj(torch.cat([global_latent_in, body_latent+hands_latent], dim=-1)))
        
        # codebook
        face_latent = self.motion_down_face(decoded_face)
        body_latent = self.motion_down_body(motion_body+body_latent)
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        global_latent = self.motion_down_global(motion_global+global_latent)

        cls_face = self.face_classifier(face_latent)
        cls_body = self.body_classifier(body_latent)
        cls_hands = self.hands_classifier(hands_latent)

        return {
            "rec_face": face_latent,
            "rec_body": body_latent,
            "rec_hands": hands_latent,
            "rec_global": global_latent,  # raw 10d, no codebook
            "cls_face": cls_face,
            "cls_body": cls_body,
            "cls_hands": cls_hands,
        }


class MambaTalk(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args   
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_face = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_face = nn.Linear(300, args.audio_f) 
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f) 

        # audio/text fusion
        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=3)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=2)        
        self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        
        # mamba scan
        self.global_scan = GlobalScan(args)
        self.local_scan = LocalScan(args)

        # id emb
        self.speaker_encoder_face = nn.Embedding(25, args.hidden_size)
        self.speaker_encoder_body = nn.Embedding(25, args.hidden_size)

        # periodic enhancement (optional) — v8: FMLP-enhanced predictor
        pd_nc = getattr(args, 'pd_n_channels', 0)
        self.use_periodic = pd_nc > 0
        if self.use_periodic:
            n_fl = getattr(args, 'pd_n_filter_layers', 2)
            logger.info(f"MambaTalk(mhr): v8 FreqPeriodicPredictor with {pd_nc} channels, "
                        f"{n_fl} FMLP filter layers")
            self.periodic_predictor = FreqPeriodicPredictor(
                input_dim=args.hidden_size, hidden_dim=256, n_channels=pd_nc,
                n_experts=4, n_filter_layers=n_fl,
                max_T=args.pose_length * 2, dropout=0.1)
            self.periodic_mapper = PeriodicMapper(
                n_channels=pd_nc, body_dim=260, hand_dim=108)

    def forward(self, in_audio=None, in_word=None, mask=None, is_test=None, in_motion=None, use_attentions=True, use_word=True, in_id=None):
        in_word_face = self.text_pre_encoder_face(in_word)
        in_word_face = self.text_encoder_face(in_word_face)
        in_word_body = self.text_pre_encoder_body(in_word)
        in_word_body = self.text_encoder_body(in_word_body)
        bs, t, c = in_word_face.shape

        in_audio_face = self.audio_pre_encoder_face(in_audio)
        in_audio_body = self.audio_pre_encoder_body(in_audio[...,:2])
        
        if in_audio_face.shape[1] != in_motion.shape[1]:
            diff_length = in_motion.shape[1] - in_audio_face.shape[1]
            if diff_length < 0:
                in_audio_face = in_audio_face[:, :diff_length, :]
                in_audio_body = in_audio_body[:, :diff_length, :]
            else:
                in_audio_face = torch.cat((in_audio_face, in_audio_face[:,-diff_length:]),1)
                in_audio_body = torch.cat((in_audio_body, in_audio_body[:,-diff_length:]),1)

        # audio/text fusion
        alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
        alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2)
        alpha_at_face = alpha_at_face.softmax(dim=-1)
        fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
        alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
        alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        alpha_at_body = alpha_at_body.softmax(dim=-1)
        fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        
        # ID Emd
        speaker_embedding_face = self.speaker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.speaker_encoder_body(in_id).squeeze(2)
        
        # Global Scan
        global_motions, global_features = self.global_scan(in_motion, mask, speaker_embedding_face, speaker_embedding_body, fusion_face)
        
        # Local Scan
        output = self.local_scan(global_motions, global_features, speaker_embedding_face, speaker_embedding_body, fusion_body, use_word)

        if self.use_periodic:
            pd_out = self.periodic_predictor(global_motions)
            body_pe, hand_pe = self.periodic_mapper(pd_out["periodic_signal"])
            output["periodic_params"] = {k: pd_out[k] for k in ("A", "F", "B", "S")}
            output["periodic_body"] = body_pe
            output["periodic_hands"] = hand_pe

        return output
