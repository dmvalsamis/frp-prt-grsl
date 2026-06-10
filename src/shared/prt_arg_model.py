"""PRT_ARG — PRT with Atmospheric Regime Gated mixture head (Tier 2).

Architecture (encoder identical to PRT; output differs):

    residual = g(x) * head_calm(z) + (1 - g(x)) * head_windy(z)
    pred     = expm1( log1p(persistence) + rs * residual )

where:
    z       = last-timestep encoder hidden vector  (B, hidden_dim)
    g(x)    = sigmoid( MLP( x[:, -1, gate_indices] ) ) in (0, 1)
              Gate output close to 1 -> trust calm head (more correction).
              Gate output close to 0 -> trust windy head (less correction).
    rs      = learnable scalar in [0, residual_scale_max]  (same as PRT)

The windy head is externally regularized in the training loop with an L2
penalty on its residual output:   L_total = L_ssat + lambda * E[r_windy^2]
This is what makes the windy regime default to persistence: the windy head
pays a cost for any nonzero correction. Implemented externally (not as a
Module hook) so the SSAT loss stays pure.

Default gate inputs assume the WindFeats-extended 37-feature input layout
(see wind_features.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_GATE_FEATURE_INDICES = [
    27,  # wind_speed_850 (raw)
    32,  # shear_signed (WindFeats)
    34,  # ws850_max_12h (WindFeats)
    35,  # hdw_product (WindFeats)
]


class PRT_ARG(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        n_lstm_layers: int,
        frp_mean: float,
        frp_std: float,
        residual_scale_max: float,
        frp_lag_1h_idx: int = 5,
        gate_feature_indices: list[int] | None = None,
        gate_hidden: int = 8,
        gate_init_bias: float = 1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.residual_scale_max = float(residual_scale_max)
        self.frp_lag_1h_idx = int(frp_lag_1h_idx)

        if gate_feature_indices is None:
            gate_feature_indices = DEFAULT_GATE_FEATURE_INDICES
        # Buffer so .to(device) moves the indices with the module.
        self.register_buffer(
            "gate_idx", torch.tensor(gate_feature_indices, dtype=torch.long)
        )
        n_gate = len(gate_feature_indices)

        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Encoder (identical to PRT)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.input_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)

        # Two residual heads
        self.head_calm = nn.Linear(hidden_dim, horizon)
        self.head_windy = nn.Linear(hidden_dim, horizon)

        # Gate network: small MLP biased toward calm at init.
        self.gate_net = nn.Sequential(
            nn.Linear(n_gate, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(float(gate_init_bias))

        # Stashed aux for training-loop access; reset every forward.
        self._last_r_windy: torch.Tensor | None = None
        self._last_gate: torch.Tensor | None = None
        self._last_r_calm: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frp_norm = x[:, -1, self.frp_lag_1h_idx]
        persistence_raw = torch.clamp(frp_norm * self.frp_std + self.frp_mean, min=0.0)
        persistence_log = torch.log1p(persistence_raw)

        h = self.input_drop(self.input_proj(x))
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm1(lstm_out + attn_out)
        ffn_out = self.ffn(lstm_out)
        lstm_out = self.norm2(lstm_out + ffn_out)
        last_h = self.out_drop(lstm_out[:, -1, :])

        r_calm = self.head_calm(last_h)
        r_windy = self.head_windy(last_h)

        gate_in = x[:, -1, :].index_select(1, self.gate_idx)  # (B, n_gate)
        g_logits = self.gate_net(gate_in)
        g = torch.sigmoid(g_logits)  # (B, 1)

        residual = g * r_calm + (1.0 - g) * r_windy

        rs = torch.clamp(self.residual_scale, min=0.0, max=self.residual_scale_max)
        pred_log = persistence_log.unsqueeze(1) + rs * residual
        pred_raw = torch.clamp(torch.expm1(pred_log), min=0.0)

        self._last_r_windy = r_windy
        self._last_r_calm = r_calm
        self._last_gate = g

        return pred_raw
