"""Event-balanced batch sampler for SSAT training.

The SSAT loss computes a per-event MAE_ratio per minibatch, so each batch
must contain multiple samples from multiple events.

Strategy per batch:
  - Sample K events uniformly at random (with replacement across batches,
    without replacement within one batch where possible).
  - For each event, sample M samples uniformly without replacement.
  - Total batch size = K * M.

Events with too few samples are sampled with replacement. Events that have
zero fire-active samples are excluded from the event pool (they cannot
contribute to the SSAT term and would only add noise).
"""
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class EventBalancedBatchSampler(Sampler[list[int]]):
    """Yields lists of sample indices for each batch.

    Args:
        manifest: DataFrame with event_name column, len == N.
        targets: (N, H) raw MW, used to identify fire-active events.
        events_per_batch: K, number of distinct events per batch.
        samples_per_event: M, number of samples drawn per event per batch.
        num_batches: how many batches per epoch. If None, computed as
                     ceil(N_fire_active / (K*M)).
        seed: base seed for the sampler RNG.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        targets: np.ndarray,
        events_per_batch: int = 8,
        samples_per_event: int = 8,
        num_batches: int | None = None,
        seed: int = 0,
    ):
        self.events_per_batch = int(events_per_batch)
        self.samples_per_event = int(samples_per_event)
        self.batch_size = self.events_per_batch * self.samples_per_event

        # Group sample indices by event
        ev = manifest["event_name"].values
        self._indices_by_event: dict[str, np.ndarray] = {}
        fire = (targets > 0).any(axis=1)

        per_event_counts = defaultdict(int)
        for i, e in enumerate(ev):
            per_event_counts[e] += 1
        self.per_event_counts = dict(per_event_counts)

        for event in np.unique(ev):
            ev_mask = (ev == event) & fire  # restrict to fire-active samples
            idx = np.flatnonzero(ev_mask)
            if idx.size == 0:
                continue
            self._indices_by_event[event] = idx
        self.events = sorted(self._indices_by_event.keys())

        if num_batches is None:
            total_fire = sum(len(v) for v in self._indices_by_event.values())
            num_batches = max(1, total_fire // self.batch_size)
        self.num_batches = int(num_batches)

        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            chosen_events = self._rng.choice(
                self.events,
                size=min(self.events_per_batch, len(self.events)),
                replace=False,
            )
            batch: list[int] = []
            for event in chosen_events:
                pool = self._indices_by_event[event]
                if pool.size >= self.samples_per_event:
                    picks = self._rng.choice(pool, size=self.samples_per_event, replace=False)
                else:
                    picks = self._rng.choice(pool, size=self.samples_per_event, replace=True)
                batch.extend(int(i) for i in picks)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
