"""
Interaction modules for dual-person gesture generation.
- StateSignalFilter: interface-level filter ensuring only allowed signal types pass through
- StateEmbedding: embeds 5 dialogue state tokens and aligns to pose frame rate
- InteractionFusion: tri-path cross-attention fusion (self + partner + state)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Signal Category Classification (fusion_analysis.md Section 0.2) ===
#
# | Category            | Allowed | Examples                                |
# |---------------------|:-------:|-----------------------------------------|
# | Head Motion         |   YES   | nodding, head shaking, head orientation |
# | Body Gesture        |   YES   | raise hand, lean forward, open arms     |
# | Facial Expression   |   NO    | smiling, crying, frowning, mouth open   |
#
# SoulX-Duplug outputs 5 dialogue-level state tokens. These describe
# turn-taking dynamics (who speaks, pause vs complete), NOT motion types.
# All 5 are allowed because they condition gesture timing/intensity,
# not facial blendshapes.  The filter below enforces this boundary and
# maps any out-of-range or future facial-expression token ids to IDLE.

STATE_NAME_TO_ID = {
    "user_idle": 0,
    "user_nonidle": 1,
    "user_backchannel": 2,
    "user_complete": 3,
    "user_incomplete": 4,
}
NUM_STATES = len(STATE_NAME_TO_ID)

# Signal categories per fusion_analysis.md Section 0.2
SIGNAL_CATEGORY = {
    0: "head_motion+body_gesture",   # idle       -> listener posture (allowed)
    1: "head_motion+body_gesture",   # nonidle    -> speaking gestures (allowed)
    2: "head_motion+body_gesture",   # backchannel-> nodding, small gestures (allowed)
    3: "head_motion+body_gesture",   # complete   -> turn-yield posture (allowed)
    4: "head_motion+body_gesture",   # incomplete -> hold/pause posture (allowed)
}

# Reserved range [5, ...) for future tokens that may carry facial expression
# semantics; these will be blocked and mapped to IDLE (id=0).
FACIAL_EXPRESSION_IDS = set()  # currently empty; populated if model is extended


class StateSignalFilter(nn.Module):
    """Interface-level filter per fusion_analysis.md Section 0.2-0.3.

    Ensures only Head Motion and Body Gesture signals reach the gesture
    generator.  Any token id in FACIAL_EXPRESSION_IDS or outside the
    valid range [0, NUM_STATES) is replaced with IDLE (0).
    """

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        filtered = state_ids.clone()
        out_of_range = (filtered < 0) | (filtered >= NUM_STATES)
        filtered[out_of_range] = 0  # map invalid -> idle
        if FACIAL_EXPRESSION_IDS:
            for fid in FACIAL_EXPRESSION_IDS:
                filtered[filtered == fid] = 0
        return filtered


class StateEmbedding(nn.Module):
    """Maps discrete dialogue-state ids to continuous embeddings and temporally
    aligns from SoulX-Duplug rate (~6.25 Hz) to the target pose frame rate."""

    def __init__(self, embed_dim: int = 768, n_states: int = NUM_STATES):
        super().__init__()
        self.embed = nn.Embedding(n_states, embed_dim)
        self.temporal_smooth = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state_ids: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            state_ids: (B, T_state) long tensor, values in [0, NUM_STATES)
            target_length: desired temporal length (pose frames)
        Returns:
            (B, target_length, embed_dim)
        """
        x = self.embed(state_ids)                           # (B, T_state, D)
        x = x.transpose(1, 2)                               # (B, D, T_state)
        x = F.interpolate(x, size=target_length, mode="nearest")  # (B, D, T)
        x = self.temporal_smooth(x)                          # (B, D, T)
        x = x.transpose(1, 2)                               # (B, T, D)
        return self.norm(x)


class InteractionFusion(nn.Module):
    """Tri-path adaptive fusion inspired by Co3Gesture TIM.

    Three information paths:
      1. self_feat    : the model's own global_features (identity path)
      2. partner_feat : cross-attention with partner condition
      3. state_feat   : cross-attention with dialogue state embedding

    An adaptive sigma network learns per-frame per-dimension weights across
    the three paths via softmax, replacing the simple additive injection of v2.
    """

    def __init__(self, hidden_size: int = 768, n_heads: int = 4,
                 n_cross_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        decoder_layer_p = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True,
        )
        self.partner_cross_attn = nn.TransformerDecoder(
            decoder_layer_p, num_layers=n_cross_layers
        )

        decoder_layer_s = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True,
        )
        self.state_cross_attn = nn.TransformerDecoder(
            decoder_layer_s, num_layers=n_cross_layers
        )

        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, global_features: torch.Tensor,
                cond_feat: torch.Tensor,
                state_embed: torch.Tensor,
                return_sigma: bool = False):
        """
        Args:
            global_features: (B, T, D) from GlobalScan
            cond_feat:       (B, T, D) partner condition projection
            state_embed:     (B, T, D) dialogue state embedding
            return_sigma:    if True, also return the (B, T, 3) sigma weights
        Returns:
            fused: (B, T, D)
            sigma: (B, T, 3)  -- only when return_sigma=True
        """
        f_partner = self.partner_cross_attn(
            tgt=global_features, memory=cond_feat
        )
        f_state = self.state_cross_attn(
            tgt=global_features, memory=state_embed
        )

        concat = torch.cat([global_features, f_partner, f_state], dim=-1)
        sigma = self.sigma_net(concat).softmax(dim=-1)      # (B, T, 3)

        fused = (sigma[..., 0:1] * global_features
                 + sigma[..., 1:2] * f_partner
                 + sigma[..., 2:3] * f_state)
        fused = self.out_norm(fused)
        if return_sigma:
            return fused, sigma
        return fused
