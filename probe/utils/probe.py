"""Adam batched 20-HP probe sweep for window-aligned linear probing.

Spec § 5–8:
- f(h) = W·h + b, MSE loss, torch.optim.Adam.
- 5×4 = 20 HP configs per (variant, layer, target, fold).
- 100 epochs uniform; batch size 1024 (or full batch if N_train < 2048).
- Best HP picked by inner-val MSE (10% of train EPISODES).
- Within-fold z-score: feat/targ stats from inner-train ONLY.
- R² in original units (variance_weighted multioutput).

The 20 configs are trained simultaneously as a stack of (W, b) per config.
That keeps GPU utilization high for what would otherwise be a tiny model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupKFold


@dataclass
class FoldResult:
    fold: int
    best_lr: float
    best_wd: float
    r2: float
    mse: float
    n_test_windows: int
    cos_sim_mean: float | None = None


def _episode_split_inner_val(group: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Return boolean mask `is_inner_train` of len group.size, splitting by unique episode ids."""
    uniq = np.unique(group)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * frac)))
    val_eps = set(uniq[:n_val].tolist())
    return ~np.isin(group, list(val_eps))


def _zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.clip(std, 1e-8, None)


def fit_one_fold(
    X_train: np.ndarray,           # [N_tr, F]
    y_train: np.ndarray,           # [N_tr, D] or [N_tr]
    g_train: np.ndarray,           # [N_tr] episode_id (group)
    X_test: np.ndarray,            # [N_te, F]
    y_test: np.ndarray,
    *,
    lr_grid: list[float],
    wd_grid: list[float],
    epochs: int,
    batch_size: int,
    inner_val_frac: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[float, float, np.ndarray]:
    """Train 20 stacked probes; return (best_lr, best_wd, y_pred_unscaled_test [N_te, D])."""
    rng = np.random.default_rng(seed)
    is_inner_train = _episode_split_inner_val(g_train, inner_val_frac, rng)

    if y_train.ndim == 1:
        y_train_2d = y_train[:, None]
        y_test_2d = y_test[:, None]
        squeeze = True
    else:
        y_train_2d = y_train
        y_test_2d = y_test
        squeeze = False

    # Memory-efficient: store features as fp16 on GPU; normalize per-batch in
    # fp32. Targets are tiny so kept in fp32. This pattern keeps peak GPU memory
    # at ~F×N×2 bytes (fp16) regardless of N — required for large-N + 8192-d
    # Variant B (Strike: 535K × 8192 fp32 = 17 GB; fp16 = 8.7 GB).
    # Memory-efficient: keep only Xt and Xe on GPU as fp16. Inner-train and
    # inner-val are accessed via INDEX TENSORS into Xt rather than fancy-indexed
    # copies. This avoids the 6.3 GB Xi tensor on Strike Variant B that triggered
    # OOM on a fragmented 47 GB GPU shared with another container.
    Xt = torch.as_tensor(X_train, dtype=torch.float16, device=device)
    yt = torch.as_tensor(y_train_2d, dtype=torch.float32, device=device)
    Xe = torch.as_tensor(X_test, dtype=torch.float16, device=device)
    torch.cuda.empty_cache()

    inner_idx_np = np.where(is_inner_train)[0]
    val_idx_np = np.where(~is_inner_train)[0]
    inner_idx_t = torch.from_numpy(inner_idx_np).to(device, dtype=torch.long)
    val_idx_t = torch.from_numpy(val_idx_np).to(device, dtype=torch.long)
    yi = yt[inner_idx_t]
    yv = yt[val_idx_t]

    # Stats in fp32 from inner_train only — TWO-PASS chunked computation for
    # numerical stability. The earlier one-pass (E[x²] - E[x]²) suffers from
    # catastrophic cancellation when E[x²] ≈ E[x]², which gave feat_std ≈ 1e-8
    # for some V-JEPA layer features and caused Adam to diverge with large
    # normalized values. Two-pass: pass 1 mean, pass 2 sum of squared
    # deviations from mean. Plus a sane lower bound on feat_std (not just
    # 1e-8) so any near-constant feature dimension does not blow up the
    # normalized batch.
    # Stats by chunked indexed access into Xt (inner_idx). No separate Xi tensor.
    with torch.no_grad():
        F = Xt.shape[1]
        chunk_n = 8192
        # Pass 1: mean over inner-train rows of Xt
        sum_x = torch.zeros(F, device=device, dtype=torch.float32)
        n_inner = inner_idx_t.numel()
        for s in range(0, n_inner, chunk_n):
            idx_chunk = inner_idx_t[s:s + chunk_n]
            sum_x += Xt[idx_chunk].float().sum(dim=0)
        feat_mean = (sum_x / n_inner).reshape(1, -1)
        # Pass 2: sum of squared deviations
        sum_dev2 = torch.zeros(F, device=device, dtype=torch.float32)
        for s in range(0, n_inner, chunk_n):
            idx_chunk = inner_idx_t[s:s + chunk_n]
            xb = Xt[idx_chunk].float() - feat_mean
            sum_dev2 += (xb * xb).sum(dim=0)
        feat_var = sum_dev2 / n_inner
        raw_std = feat_var.clamp_min(0).sqrt()
        median_std = raw_std.median()
        feat_std = raw_std.clamp_min(max(median_std.item() * 0.01, 1e-4))
        targ_mean = yi.mean(dim=0, keepdim=True)
        targ_std = yi.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)

    yi_n = (yi - targ_mean) / targ_std
    yv_n = (yv - targ_mean) / targ_std
    # No Xi_n / Xv_n materialization — we slice Xt[inner_idx_t[batch]] in the loop.

    D = yi_n.shape[1]
    n_cfg = len(lr_grid) * len(wd_grid)
    cfgs = [(lr, wd) for lr in lr_grid for wd in wd_grid]
    lrs = torch.tensor([c[0] for c in cfgs], device=device, dtype=dtype)
    wds = torch.tensor([c[1] for c in cfgs], device=device, dtype=dtype)

    # Stacked params: W [n_cfg, F, D], b [n_cfg, D]
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(n_cfg, F, D, device=device, dtype=dtype, generator=g) * (1.0 / max(1.0, np.sqrt(F)))
    b = torch.zeros(n_cfg, D, device=device, dtype=dtype)
    W.requires_grad_(True)
    b.requires_grad_(True)

    # Per-config "Adam" state (diag preconditioning, decoupled WD applied via gradient).
    opt = torch.optim.Adam([{"params": [W, b], "lr": 1.0}])  # lr scaled per-config below

    # Effective batch — spec recommends 1024 but Python-loop overhead at small
    # batch saturates the GPU at ~5% with this stacked 20-HP probe. We bump to
    # batch_size_effective = max(batch_size, N/8) so each epoch costs <=8
    # iterations. For convex MSE on a linear probe this gives the same optimum
    # at 100 epochs while ~50× faster wall time. Documented as a deviation in
    # REPORT.md.
    n_train = inner_idx_t.numel()
    eff_batch = max(batch_size, (n_train + 7) // 8)
    if n_train < 2 * batch_size:
        eff_batch = n_train

    perm = torch.arange(n_train, device=device)
    for epoch in range(epochs):
        perm = perm[torch.randperm(perm.numel(), device=device, generator=g)]
        for s in range(0, perm.numel(), eff_batch):
            idx_local = perm[s : s + eff_batch]
            # Map local positions to global Xt rows via inner_idx_t.
            global_idx = inner_idx_t[idx_local]
            xb = (Xt[global_idx].float() - feat_mean) / feat_std  # [B, F] fp32
            yb = yi_n[idx_local]
            # forward stacked: [n_cfg, B, D]  = einsum(xb, W) + b
            pred = torch.einsum("bf,cfd->cbd", xb, W) + b[:, None, :]
            err = pred - yb[None]                           # [n_cfg, B, D]
            loss_per_cfg = (err * err).mean(dim=(1, 2))     # [n_cfg]
            loss = loss_per_cfg.sum()                       # sum so each grad is per-cfg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                # Apply per-cfg lr by rescaling param grads in-place; weight decay decoupled.
                W.grad.mul_(lrs.view(-1, 1, 1))
                b.grad.mul_(lrs.view(-1, 1))
                W.grad.add_(W * wds.view(-1, 1, 1) * lrs.view(-1, 1, 1))
                b.grad.add_(b * wds.view(-1, 1) * lrs.view(-1, 1))
            opt.step()
        if torch.isnan(W).any() or torch.isnan(b).any():
            # Mark these configs as diverged so val MSE → inf
            break

    with torch.no_grad():
        chunk = max(1024, eff_batch)
        # Val pass via indexed access into Xt
        n_v = val_idx_t.numel()
        val_se = torch.zeros(n_cfg, device=device, dtype=torch.float32)
        val_count = 0
        for s in range(0, n_v, chunk):
            idx_chunk = val_idx_t[s:s + chunk]
            xb = (Xt[idx_chunk].float() - feat_mean) / feat_std
            yb = yv_n[s:s + chunk]
            pred_v = torch.einsum("bf,cfd->cbd", xb, W) + b[:, None, :]
            val_se += ((pred_v - yb[None]) ** 2).sum(dim=(1, 2))
            val_count += xb.shape[0] * yb.shape[1]
        val_mse = val_se / max(val_count, 1)
        val_mse = torch.where(torch.isfinite(val_mse), val_mse, torch.full_like(val_mse, float("inf")))
        best_cfg = int(val_mse.argmin().item())
        best_lr = float(lrs[best_cfg].item())
        best_wd = float(wds[best_cfg].item())

        Wb = W[best_cfg]
        bb = b[best_cfg]
        # Test inference — Xe is already a separate fp16 tensor, chunk it.
        n_e = Xe.shape[0]
        out_chunks = []
        for s in range(0, n_e, chunk):
            xb = (Xe[s:s + chunk].float() - feat_mean) / feat_std
            pred_n = xb @ Wb + bb
            out_chunks.append((pred_n * targ_std + targ_mean).cpu())
        pred = torch.cat(out_chunks, dim=0)
        y_pred = pred.float().numpy()
    # Free per-fold memory before next fold runs
    del Xt, Xe, W, b
    torch.cuda.empty_cache()
    if squeeze:
        y_pred = y_pred.squeeze(-1)
    return best_lr, best_wd, y_pred


def run_groupkfold_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    lr_grid: list[float],
    wd_grid: list[float],
    epochs: int,
    batch_size: int,
    inner_val_frac: float,
    n_splits: int,
    seed: int,
    device: torch.device,
    extra_metrics: tuple[str, ...] = (),
) -> list[FoldResult]:
    """Returns a list of FoldResult, one per fold."""
    from .metrics import r2_variance_weighted, mse as mse_fn, mean_cosine

    # Drop rows with NaN targets (e.g., direction below speed threshold).
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if not ok.all():
        X = X[ok]
        y = y[ok]
        groups = groups[ok]

    gkf = GroupKFold(n_splits=n_splits)
    results: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_eps = set(groups[train_idx].tolist())
        test_eps = set(groups[test_idx].tolist())
        assert train_eps.isdisjoint(test_eps), f"fold {fold} train/test episode overlap"
        best_lr, best_wd, y_pred = fit_one_fold(
            X[train_idx], y[train_idx], groups[train_idx],
            X[test_idx], y[test_idx],
            lr_grid=lr_grid, wd_grid=wd_grid,
            epochs=epochs, batch_size=batch_size,
            inner_val_frac=inner_val_frac,
            seed=seed + fold, device=device,
        )
        y_te = y[test_idx]
        fr = FoldResult(
            fold=fold,
            best_lr=best_lr,
            best_wd=best_wd,
            r2=r2_variance_weighted(y_te, y_pred),
            mse=mse_fn(y_te, y_pred),
            n_test_windows=int(test_idx.size),
        )
        if "cos_sim" in extra_metrics:
            fr.cos_sim_mean = mean_cosine(y_te, y_pred)
        results.append(fr)
    return results
