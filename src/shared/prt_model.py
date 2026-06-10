"""PRT (Persistence Residual Transformer) model class.

Copied from ../scripts/11_train_tft_v6.py on 2026-05-19; structure unchanged.
The "TFT" name in the parent script is mislabeled per project memory; the
model is a Persistence Residual Transformer.

In the redesign ablation track, row 0 (baseline) uses this class verbatim.
Subsequent rows (HC, ARG) will SUBCLASS or REPLACE this in their own files,
never modify this one.
"""
import torch
import torch.nn as nn


class PRT(nn.Module):
    """Persistence Residual Transformer with log-space residual decomposition.

    pred = expm1(log1p(persistence) + residual_scale * residual)
    where:
      persistence = denormalized frp_lag_1h at the last lookback timestep
      residual    = LSTM + self-attn + FFN encoder, last-timestep -> Linear(hd, horizon)
      residual_scale = learnable scalar, init 0.1, forward-clamped to [0, residual_scale_max]
    """

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
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.residual_scale_max = float(residual_scale_max)
        self.frp_lag_1h_idx = int(frp_lag_1h_idx)

        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

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
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, horizon)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> pred: (B, H) in raw MW (>=0)."""
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
        residual = self.output_proj(last_h)

        rs = torch.clamp(self.residual_scale, min=0.0, max=self.residual_scale_max)
        pred_log = persistence_log.unsqueeze(1) + rs * residual
        pred_raw = torch.clamp(torch.expm1(pred_log), min=0.0)
        return pred_raw
