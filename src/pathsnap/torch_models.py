"""Neural candidate scorers in PyTorch (GPU when available).

mode="pointwise"  an MLP scores each (point, candidate) row on its own; binary cross-entropy.
mode="listwise"   the same MLP scores every candidate of a point; a softmax over that point's
                  candidates is trained with cross-entropy against the reference candidate.
                  This optimises exactly the decision being made (which of these candidates),
                  and its output is already a distribution over the candidate set, which is
                  what the HMM emission uses.

Both standardise features with statistics from the training rows only, hold out 10 % of the
training *trips* for early stopping, and record per-epoch train/validation loss (real training
histories for the report).
"""
from __future__ import annotations

import time

import numpy as np


def _groups(keys: np.ndarray):
    """Rows are contiguous per point key. Returns (padded row index (P, K) with -1, group of row)."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    counts = np.diff(np.r_[starts, len(keys)])
    grp = np.repeat(np.arange(len(starts)), counts)
    pos = np.arange(len(keys)) - np.repeat(starts, counts)
    idx = np.full((len(starts), int(counts.max())), -1, dtype=np.int64)
    idx[grp, pos] = np.arange(len(keys))
    return idx, grp


class TorchScorer:
    def __init__(self, mode: str = "listwise", hidden=(128, 64), dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_points: int = 4096, max_epochs: int = 40,
                 patience: int = 5, val_frac: float = 0.1, seed: int = 352):
        self.mode = mode
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_points = batch_points
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------ helpers
    def _device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self, n_in: int):
        import torch.nn as nn
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def _scores(self, X_t, rows_t):
        """Scores for a (B, K) block of row indices (-1 = padding -> -inf)."""
        import torch
        safe = rows_t.clamp(min=0)
        s = self.net_(X_t[safe.reshape(-1)]).reshape(rows_t.shape)
        return s.masked_fill(rows_t < 0, float("-inf"))

    def _listwise_loss(self, X_t, rows_t, target_t):
        import torch.nn.functional as F
        return F.cross_entropy(self._scores(X_t, rows_t), target_t)

    # ------------------------------------------------------------------ API
    def fit_grouped(self, X: np.ndarray, y: np.ndarray, keys: np.ndarray):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self._device()
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-6
        Xs = ((X - self.mu_) / self.sd_).astype(np.float32)

        trips = keys // 1_000_000
        ut = np.unique(trips)
        val_trips = rng.choice(ut, size=max(1, int(round(self.val_frac * len(ut)))), replace=False)
        is_val = np.isin(trips, val_trips)

        X_t = torch.from_numpy(Xs).to(dev)
        self.net_ = self._build(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.mode == "listwise":
            idx, _ = _groups(keys)
            ypad = np.where(idx >= 0, y[np.maximum(idx, 0)], 0)
            has_pos = ypad.sum(axis=1) > 0  # points whose reference is a candidate
            target = ypad.argmax(axis=1)
            g_val = is_val[idx[:, 0]]
            tr_g = np.flatnonzero(has_pos & ~g_val)
            va_g = np.flatnonzero(has_pos & g_val)
            idx_t = torch.from_numpy(idx).to(dev)
            tgt_t = torch.from_numpy(target).to(dev)
        else:
            y_t = torch.from_numpy(y.astype(np.float32)).to(dev)
            tr_r = np.flatnonzero(~is_val)
            va_r = np.flatnonzero(is_val)
            rows_per_batch = self.batch_points * 8

        best, best_state, bad = np.inf, None, 0
        t0 = time.perf_counter()
        for epoch in range(self.max_epochs):
            self.net_.train()
            tot, cnt = 0.0, 0
            if self.mode == "listwise":
                perm = rng.permutation(tr_g)
                for b in range(0, len(perm), self.batch_points):
                    g = torch.from_numpy(perm[b:b + self.batch_points]).to(dev)
                    loss = self._listwise_loss(X_t, idx_t[g], tgt_t[g])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(g)
                    cnt += len(g)
            else:
                perm = rng.permutation(tr_r)
                for b in range(0, len(perm), rows_per_batch):
                    r = torch.from_numpy(perm[b:b + rows_per_batch]).to(dev)
                    loss = F.binary_cross_entropy_with_logits(self.net_(X_t[r]).squeeze(1), y_t[r])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(r)
                    cnt += len(r)
            train_loss = tot / max(cnt, 1)

            self.net_.eval()
            with torch.no_grad():
                if self.mode == "listwise":
                    vl = [float(self._listwise_loss(X_t, idx_t[torch.from_numpy(va_g[b:b + 16384]).to(dev)],
                                                    tgt_t[torch.from_numpy(va_g[b:b + 16384]).to(dev)]))
                          * len(va_g[b:b + 16384]) for b in range(0, len(va_g), 16384)]
                    val_loss = sum(vl) / max(len(va_g), 1)
                else:
                    r = torch.from_numpy(va_r).to(dev)
                    val_loss = float(F.binary_cross_entropy_with_logits(self.net_(X_t[r]).squeeze(1), y_t[r]))
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def predict_proba_grouped(self, X: np.ndarray, keys: np.ndarray) -> np.ndarray:
        import torch

        dev = self._device()
        Xs = ((X - self.mu_) / self.sd_).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(Xs).to(dev)
            s = torch.cat([self.net_(X_t[b:b + 262144]).squeeze(1) for b in range(0, len(Xs), 262144)])
            s = s.cpu().numpy().astype(np.float64)
        if self.mode == "pointwise":
            return 1.0 / (1.0 + np.exp(-s))
        _, grp = _groups(keys)
        m = np.full(grp.max() + 1, -np.inf)
        np.maximum.at(m, grp, s)
        e = np.exp(s - m[grp])
        tot = np.bincount(grp, weights=e)
        return e / tot[grp]
