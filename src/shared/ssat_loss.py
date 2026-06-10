"""SSAT — Skill-Score Aligned Training loss.

Loss formulation (per minibatch):

  L = w_ssat * mean_event_in_batch[
            sum_h w_h * MAE_fire(pred_e[:,h], target_e[:,h]) / MAE_pers[e, h]
        ]
      + w_log * LogHybridLoss_fire_weighted(pred, target)

  where:
    - "event_in_batch" iterates over events with >=1 fire-active sample in batch
    - MAE_pers[e, h] is the precomputed per-event per-horizon persistence MAE
    - w_h is a horizon weight, normalized to sum to 1 over the reporting horizons
    - w_ssat anchors the skill-aligned term (default 1.0)
    - w_log is a small log-space anchor (default 0.2) to handle all-zero windows
      and provide a smooth gradient when fire-active counts are tiny.

The first term is a direct soft form of (1 - SS_fire_t6) aggregated over the
reporting horizons. The second term keeps a gradient path for all samples.

If a batch has no fire-active samples in any event (rare but possible), the
SSAT term is zero and only the log anchor contributes.
"""
import torch
import torch.nn as nn


class SSATLoss(nn.Module):
    def __init__(
        self,
        persistence_mae: dict[str, "torch.Tensor"],
        horizon_weights: dict[int, float] | None = None,
        w_ssat: float = 1.0,
        w_log: float = 0.2,
        fire_weight: float = 7.0,
        log_alpha: float = 0.5,
    ):
        """
        Args:
            persistence_mae: dict[event_name] -> tensor of shape (H,) with MAE
                in raw MW. Built from the training split via
                persistence_mae_table.compute_persistence_mae_table.
            horizon_weights: dict from 1-based horizon to weight, e.g.
                {1: 0.10, 3: 0.20, 6: 0.50, 12: 0.20}. Weights are renormalized
                to sum to 1. Horizons not in the dict get weight 0.
            w_ssat: scale of the skill-aligned term.
            w_log: scale of the log-space anchor term.
            fire_weight: log-anchor fire weighting (matches LogHybridLoss).
            log_alpha: log-anchor MAE/MSE blend (matches LogHybridLoss).
        """
        super().__init__()
        # Default weights match the project's reporting horizons,
        # with t+6 emphasized (paper-primary).
        if horizon_weights is None:
            horizon_weights = {1: 0.10, 3: 0.20, 6: 0.50, 12: 0.20}

        # Persistence MAE: keep as a dict[str -> Tensor(H,)] registered as buffers.
        self.persistence_mae = persistence_mae
        self.horizon_weights_dict = horizon_weights
        # Pre-compute the weight vector lazily on the first call (need horizon size).
        self._h_weight_vec: torch.Tensor | None = None

        self.w_ssat = float(w_ssat)
        self.w_log = float(w_log)
        self.fire_weight = float(fire_weight)
        self.log_alpha = float(log_alpha)

    def _ensure_h_weights(self, horizon: int, device, dtype):
        if self._h_weight_vec is None or self._h_weight_vec.numel() != horizon:
            w = torch.zeros(horizon, device=device, dtype=dtype)
            for h_1based, val in self.horizon_weights_dict.items():
                if 1 <= h_1based <= horizon:
                    w[h_1based - 1] = float(val)
            total = w.sum()
            if total > 0:
                w = w / total
            self._h_weight_vec = w
        return self._h_weight_vec

    def _log_anchor(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_log = torch.log1p(pred)
        target_log = torch.log1p(target)
        diff = pred_log - target_log
        per = self.log_alpha * torch.abs(diff) + (1.0 - self.log_alpha) * diff.pow(2)
        fire_active = (target > 0).any(dim=1, keepdim=True).float()
        weight = 1.0 + (self.fire_weight - 1.0) * fire_active
        return (per * weight).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        event_names_in_batch: list[str],
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:   (B, H) raw MW from the model.
            target: (B, H) raw MW.
            event_names_in_batch: list of length B with the event name for each sample.

        Returns:
            (loss_scalar, info_dict) where info_dict has the SSAT-term value,
            log-anchor value, and number of events that contributed.
        """
        B, H = pred.shape
        device = pred.device
        dtype = pred.dtype
        w_h = self._ensure_h_weights(H, device, dtype)

        # Fire-active mask: any non-zero target step in the window.
        fire_mask = (target > 0).any(dim=1)  # (B,)

        # Group sample indices by event for samples that are fire-active.
        event_to_idx: dict[str, list[int]] = {}
        for i in range(B):
            if not fire_mask[i]:
                continue
            ev = event_names_in_batch[i]
            event_to_idx.setdefault(ev, []).append(i)

        ssat_terms = []
        for ev, idx_list in event_to_idx.items():
            if ev not in self.persistence_mae:
                continue
            idx = torch.tensor(idx_list, device=device, dtype=torch.long)
            pred_e = pred.index_select(0, idx)      # (n_e, H)
            target_e = target.index_select(0, idx)  # (n_e, H)
            mae_e = (pred_e - target_e).abs().mean(dim=0)  # (H,)
            mae_pers = self.persistence_mae[ev].to(device=device, dtype=dtype)  # (H,)
            ratio_h = mae_e / (mae_pers + 1e-6)  # (H,)
            ssat_terms.append((ratio_h * w_h).sum())

        if ssat_terms:
            ssat_term = torch.stack(ssat_terms).mean()
        else:
            ssat_term = torch.tensor(0.0, device=device, dtype=dtype)

        log_term = self._log_anchor(pred, target)

        loss = self.w_ssat * ssat_term + self.w_log * log_term

        info = {
            "ssat_term": float(ssat_term.detach().cpu()),
            "log_anchor": float(log_term.detach().cpu()),
            "n_events_in_batch": len(ssat_terms),
        }
        return loss, info
