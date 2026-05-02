"""Statistical utility helpers for probing analyses.

This module provides bootstrap confidence intervals, permutation tests,
fold-consistency summaries, and rough memory pre-compute checks.

The implementation is intentionally lightweight:
- imports only ``numpy`` and ``torch``
- handles NaN/inf inputs explicitly
- can be executed via ``/isaac-sim/python.sh`` for dry-run sanity checks
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_1d_finite_array(x: np.ndarray | list[float], name: str) -> np.ndarray:
    """Convert input to 1D float array and keep only finite values.

    Args:
        x: Array-like numeric input.
        name: Human-readable name for validation errors.

    Returns:
        A 1D ``float64`` NumPy array containing only finite values.

    Raises:
        ValueError: If the input becomes empty after filtering finite values.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} has no finite values after NaN/inf filtering.")
    return arr


def _validate_ci(ci_lo: float, ci_hi: float, name: str) -> None:
    """Validate that a confidence interval is finite and ordered."""
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        raise ValueError(f"{name} confidence interval is not finite: ({ci_lo}, {ci_hi}).")
    if not ci_lo < ci_hi:
        raise ValueError(f"{name} confidence interval is not ordered: ({ci_lo}, {ci_hi}).")


def bootstrap_r2_ci(
    per_episode_r2: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap confidence interval over per-episode R².

    NaN/inf entries are removed before bootstrapping. Resampling is performed
    at the episode level.

    Args:
        per_episode_r2: 1D array of per-episode R² values.
        n_boot: Number of bootstrap resamples.
        alpha: Confidence level tail mass. ``alpha=0.05`` gives a 95% CI.
        seed: Optional RNG seed for deterministic resampling.

    Returns:
        Dict with keys ``mean``, ``ci_lo``, ``ci_hi``, ``n_boot``, ``method``.

    Raises:
        ValueError: If inputs are invalid or no finite samples remain.
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    arr = _as_1d_finite_array(per_episode_r2, "per_episode_r2")
    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(arr[idx]))

    mean = float(np.mean(arr))
    ci_lo = float(np.quantile(boot_means, alpha / 2.0))
    ci_hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    _validate_ci(ci_lo, ci_hi, "bootstrap_r2_ci")

    return {
        "mean": mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_boot": int(n_boot),
        "method": "percentile",
    }


def bootstrap_diff_ci(
    a_per_ep: np.ndarray,
    b_per_ep: np.ndarray,
    paired: bool,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap a confidence interval for the mean difference between groups.

    For paired inputs, episodes are assumed aligned and are resampled by shared
    indices after dropping any pair with a non-finite element. For unpaired
    inputs, each group is resampled independently after per-group finite
    filtering.

    The one-tailed p-value is reported for the directional hypothesis
    ``mean(a) - mean(b) > 0``.

    Args:
        a_per_ep: Group A per-episode values.
        b_per_ep: Group B per-episode values.
        paired: Whether to use paired resampling.
        n_boot: Number of bootstrap resamples.
        alpha: Confidence level tail mass.
        seed: Optional RNG seed for deterministic resampling.

    Returns:
        Dict with keys ``diff_mean``, ``ci_lo``, ``ci_hi``,
        ``p_value_one_tailed``.
    """
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    rng = np.random.default_rng(seed)

    if paired:
        a = np.asarray(a_per_ep, dtype=np.float64).reshape(-1)
        b = np.asarray(b_per_ep, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"Paired bootstrap requires matching shapes, got {a.shape} vs {b.shape}.")
        keep = np.isfinite(a) & np.isfinite(b)
        a = a[keep]
        b = b[keep]
        if a.size == 0:
            raise ValueError("Paired bootstrap has no finite aligned pairs.")
        n = a.size
        boot_diffs = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_diffs[i] = float(np.mean(a[idx] - b[idx]))
        diff_mean = float(np.mean(a - b))
    else:
        a = _as_1d_finite_array(a_per_ep, "a_per_ep")
        b = _as_1d_finite_array(b_per_ep, "b_per_ep")
        n_a = a.size
        n_b = b.size
        boot_diffs = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx_a = rng.integers(0, n_a, size=n_a)
            idx_b = rng.integers(0, n_b, size=n_b)
            boot_diffs[i] = float(np.mean(a[idx_a]) - np.mean(b[idx_b]))
        diff_mean = float(np.mean(a) - np.mean(b))

    ci_lo = float(np.quantile(boot_diffs, alpha / 2.0))
    ci_hi = float(np.quantile(boot_diffs, 1.0 - alpha / 2.0))
    _validate_ci(ci_lo, ci_hi, "bootstrap_diff_ci")

    p_value_one_tailed = float((np.count_nonzero(boot_diffs <= 0.0) + 1) / (n_boot + 1))
    return {
        "diff_mean": diff_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value_one_tailed": p_value_one_tailed,
    }


def permutation_null_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 1000,
    two_tailed: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permutation test for a difference in group means.

    Args:
        a: Group A samples.
        b: Group B samples.
        n_perm: Number of label permutations.
        two_tailed: Whether to use a two-tailed p-value.
        seed: Optional RNG seed for deterministic permutations.

    Returns:
        Dict with keys ``diff_observed``, ``p_value``, ``n_perm``.
    """
    if n_perm <= 0:
        raise ValueError(f"n_perm must be positive, got {n_perm}.")

    a_arr = _as_1d_finite_array(a, "a")
    b_arr = _as_1d_finite_array(b, "b")

    n_a = a_arr.size
    n_b = b_arr.size
    pooled = np.concatenate([a_arr, b_arr], axis=0)
    rng = np.random.default_rng(seed)

    diff_observed = float(np.mean(a_arr) - np.mean(b_arr))
    null_diffs = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(pooled.size)
        a_perm = pooled[perm[:n_a]]
        b_perm = pooled[perm[n_a : n_a + n_b]]
        null_diffs[i] = float(np.mean(a_perm) - np.mean(b_perm))

    if two_tailed:
        p_value = float((np.count_nonzero(np.abs(null_diffs) >= abs(diff_observed)) + 1) / (n_perm + 1))
    else:
        p_value = float((np.count_nonzero(null_diffs >= diff_observed) + 1) / (n_perm + 1))

    return {
        "diff_observed": diff_observed,
        "p_value": p_value,
        "n_perm": int(n_perm),
    }


def fold_consistency(per_fold_r2: list[float] | np.ndarray) -> dict[str, Any]:
    """Summarize fold-wise consistency for R² values.

    NaN/inf entries are removed before summarization.

    Args:
        per_fold_r2: Per-fold R² values, typically length 5.

    Returns:
        Dict with mean/std/min/max/spread/negative-flag/fold-count summary.
    """
    arr = _as_1d_finite_array(per_fold_r2, "per_fold_r2")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    spread = float(max_v - min_v)
    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "spread": spread,
        "any_negative": bool(np.any(arr < 0.0)),
        "n_folds": int(arr.size),
    }


def memory_precompute(
    shape: tuple[int, ...],
    dtype: Any = np.float32,
    device_id: int | None = None,
) -> dict[str, Any]:
    """Estimate tensor memory footprint and compare against current CUDA free mem.

    Args:
        shape: Tensor shape tuple.
        dtype: NumPy dtype-like specifier.
        device_id: CUDA device index. ``None`` means current/default device.

    Returns:
        Dict with ``bytes``, ``gb``, ``device_free_gb``, ``fits``, ``headroom_gb``.

    Raises:
        ValueError: If shape is malformed.
    """
    if not isinstance(shape, tuple) or len(shape) == 0:
        raise ValueError(f"shape must be a non-empty tuple, got {shape!r}.")
    if any((not isinstance(dim, (int, np.integer)) or int(dim) <= 0) for dim in shape):
        raise ValueError(f"shape must contain positive integers, got {shape!r}.")

    np_dtype = np.dtype(dtype)
    n_elem = int(np.prod(shape, dtype=np.int64))
    bytes_needed = int(n_elem * np_dtype.itemsize)
    gb_needed = float(bytes_needed / (1024.0**3))

    if torch.cuda.is_available():
        try:
            device = None if device_id is None else int(device_id)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            device_free_gb = float(free_bytes / (1024.0**3))
            headroom_gb = float(device_free_gb - gb_needed)
            fits = bool(headroom_gb >= 0.0)
        except RuntimeError:
            device_free_gb = float("nan")
            headroom_gb = float("nan")
            fits = False
    else:
        device_free_gb = float("nan")
        headroom_gb = float("nan")
        fits = False

    return {
        "bytes": bytes_needed,
        "gb": gb_needed,
        "device_free_gb": device_free_gb,
        "fits": fits,
        "headroom_gb": headroom_gb,
    }


if __name__ == "__main__":
    print("=== stats.py dry-run ===")

    rng = np.random.default_rng(0)

    # 1. bootstrap_r2_ci
    per_ep = np.array([0.1, 0.2, 0.3, np.nan, 0.4, np.inf, -0.1], dtype=np.float64)
    out1 = bootstrap_r2_ci(per_ep, n_boot=200, alpha=0.05, seed=0)
    print("bootstrap_r2_ci:", out1)

    # 2. bootstrap_diff_ci (paired and unpaired)
    a = np.array([0.4, 0.5, np.nan, 0.6, 0.7, 0.8], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6], dtype=np.float64)
    out2_paired = bootstrap_diff_ci(a, b, paired=True, n_boot=200, alpha=0.05, seed=0)
    out2_unpaired = bootstrap_diff_ci(a, b, paired=False, n_boot=200, alpha=0.05, seed=0)
    print("bootstrap_diff_ci paired:", out2_paired)
    print("bootstrap_diff_ci unpaired:", out2_unpaired)

    # 3. permutation_null_diff
    g1 = rng.normal(loc=0.5, scale=0.2, size=20)
    g2 = rng.normal(loc=0.2, scale=0.2, size=18)
    out3 = permutation_null_diff(g1, g2, n_perm=500, two_tailed=True, seed=0)
    print("permutation_null_diff:", out3)

    # 4. fold_consistency
    folds = np.array([0.12, 0.18, -0.03, 0.15, 0.10], dtype=np.float64)
    out4 = fold_consistency(folds)
    print("fold_consistency:", out4)

    # 5. memory_precompute
    out5 = memory_precompute((1024, 8192), dtype=np.float32, device_id=0)
    print("memory_precompute:", out5)
