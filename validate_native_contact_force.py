#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from tqdm import tqdm

from probe_physprobe import DATA_BASE


CONTACT_COLS = {
    "contact_force": "physics_gt.contact_force",
    "contact_finger_l_object_force": "physics_gt.contact_finger_l_object_force",
    "contact_object_surface_force": "physics_gt.contact_object_surface_force",
    "contact_flag": "physics_gt.contact_flag",
    "object_acceleration": "physics_gt.object_acceleration",
}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def find_strike_parquets() -> list[tuple[int, Path]]:
    root = Path(DATA_BASE) / "strike" / "data"
    out: list[tuple[int, Path]] = []
    for path in sorted(root.glob("chunk-*/episode_*.parquet")):
        ep = int(path.stem.split("_")[-1])
        out.append((ep, path))
    if not out:
        raise FileNotFoundError(f"No strike parquet files found under {root}")
    return out


def frame_mag_stats(parquets: list[tuple[int, Path]]) -> dict:
    episodes = []
    nonzero_episode_counts = {
        "contact_force": 0,
        "contact_finger_l_object_force": 0,
        "contact_object_surface_force": 0,
        "contact_flag": 0,
    }
    max_overall = {k: 0.0 for k in nonzero_episode_counts}
    accel_max_overall = 0.0

    cols = list(CONTACT_COLS.values())
    for ep, path in tqdm(parquets, desc="Audit native strike contact columns"):
        df = pd.read_parquet(path, columns=cols)
        cf = np.linalg.norm(np.stack(df[CONTACT_COLS["contact_force"]].values), axis=1)
        fl = np.linalg.norm(np.stack(df[CONTACT_COLS["contact_finger_l_object_force"]].values), axis=1)
        sf = np.linalg.norm(np.stack(df[CONTACT_COLS["contact_object_surface_force"]].values), axis=1)
        flag = np.stack(df[CONTACT_COLS["contact_flag"]].values).reshape(-1)
        acc = np.linalg.norm(np.stack(df[CONTACT_COLS["object_acceleration"]].values), axis=1)

        episode_row = {
            "episode": ep,
            "contact_force_nonzero_frames": int((cf > 0).sum()),
            "contact_force_max": float(cf.max()),
            "contact_finger_l_object_force_nonzero_frames": int((fl > 0).sum()),
            "contact_finger_l_object_force_max": float(fl.max()),
            "contact_object_surface_force_nonzero_frames": int((sf > 0).sum()),
            "contact_object_surface_force_max": float(sf.max()),
            "contact_flag_nonzero_frames": int((flag > 0).sum()),
            "contact_flag_max": float(flag.max()),
            "object_acceleration_max": float(acc.max()),
            "object_acceleration_mean": float(acc.mean()),
        }
        episodes.append(episode_row)

        for k, arr in [
            ("contact_force", cf),
            ("contact_finger_l_object_force", fl),
            ("contact_object_surface_force", sf),
            ("contact_flag", flag),
        ]:
            if np.any(arr > 0):
                nonzero_episode_counts[k] += 1
            max_overall[k] = max(max_overall[k], float(arr.max()))
        accel_max_overall = max(accel_max_overall, float(acc.max()))

    return {
        "num_episodes": len(episodes),
        "nonzero_episode_counts": nonzero_episode_counts,
        "max_overall": max_overall,
        "object_acceleration_max_overall": accel_max_overall,
        "episodes": episodes,
    }


def window_level_validation(
    parquets: list[tuple[int, Path]],
    feature_root: Path,
    episode_limit: int,
) -> dict:
    rows = []
    ep_map = dict(parquets)
    for ep, path in tqdm(parquets[:episode_limit], desc="Window-level surrogate/native validation"):
        feat_path = feature_root / f"{ep:06d}.safetensors"
        if not feat_path.exists():
            continue
        with safe_open(str(feat_path), framework="numpy") as f:
            window_starts = f.get_tensor("window_starts").astype(np.int64)

        df = pd.read_parquet(
            path,
            columns=[
                CONTACT_COLS["object_acceleration"],
                CONTACT_COLS["contact_force"],
                CONTACT_COLS["contact_flag"],
            ],
        )
        acc = np.linalg.norm(np.stack(df[CONTACT_COLS["object_acceleration"]].values), axis=1)
        native = np.linalg.norm(np.stack(df[CONTACT_COLS["contact_force"]].values), axis=1)
        flag = np.stack(df[CONTACT_COLS["contact_flag"]].values).reshape(-1)
        T = len(acc)
        for w, start in enumerate(window_starts):
            s = int(start)
            e = min(s + 16, T)
            rows.append(
                {
                    "episode": ep,
                    "window": int(w),
                    "surrogate_force_proxy": float(acc[s:e].max()),
                    "native_contact_force": float(native[s:e].max()),
                    "native_contact_flag": float(flag[s:e].max()),
                }
            )
    df = pd.DataFrame(rows)
    surrogate = df["surrogate_force_proxy"].to_numpy(dtype=np.float64)
    native = df["native_contact_force"].to_numpy(dtype=np.float64)
    rank_consistency = None
    if surrogate.size and not np.allclose(native, native[0]):
        top_sur = np.argsort(surrogate)[-100:][::-1]
        top_nat = np.argsort(native)[-100:][::-1]
        rank_consistency = float(len(set(top_sur) & set(top_nat)) / 100.0)
    return {
        "num_windows": int(len(df)),
        "native_nonzero_windows": int((df["native_contact_force"] > 0).sum()),
        "native_flag_nonzero_windows": int((df["native_contact_flag"] > 0).sum()),
        "surrogate_max": float(df["surrogate_force_proxy"].max()) if len(df) else 0.0,
        "native_max": float(df["native_contact_force"].max()) if len(df) else 0.0,
        "pearson_surrogate_vs_native": pearson_corr(surrogate, native),
        "spearman_surrogate_vs_native": spearman_corr(surrogate, native),
        "top100_overlap_fraction": rank_consistency,
    }


def render_markdown(summary: dict, out_path: Path) -> None:
    frame = summary["frame_audit"]
    win = summary["window_validation"]
    lines = [
        "# Surrogate Validation Verdict",
        "",
        "## Result",
        "",
        "Native Strike `contact_force` validation is **not possible** with the public Step 0 export.",
        "",
        "The reason is empirical, not inferential: the exported native `contact_force`,",
        "`contact_finger_l_object_force`, `contact_object_surface_force`, and `contact_flag`",
        "channels are zero-filled across the audited Strike data, while object-acceleration",
        "spikes remain large and frequent.",
        "",
        "## Full Strike Audit",
        "",
        f"- audited episodes: `{frame['num_episodes']}`",
        f"- native `contact_force` nonzero episodes: `{frame['nonzero_episode_counts']['contact_force']}` / `{frame['num_episodes']}`",
        f"- native `contact_finger_l_object_force` nonzero episodes: `{frame['nonzero_episode_counts']['contact_finger_l_object_force']}` / `{frame['num_episodes']}`",
        f"- native `contact_object_surface_force` nonzero episodes: `{frame['nonzero_episode_counts']['contact_object_surface_force']}` / `{frame['num_episodes']}`",
        f"- native `contact_flag` nonzero episodes: `{frame['nonzero_episode_counts']['contact_flag']}` / `{frame['num_episodes']}`",
        f"- max exported native `contact_force` magnitude over all audited frames: `{frame['max_overall']['contact_force']:.6f}`",
        f"- max exported native `contact_flag` over all audited frames: `{frame['max_overall']['contact_flag']:.6f}`",
        f"- max object-acceleration magnitude over the same audit: `{frame['object_acceleration_max_overall']:.6f}`",
        "",
        "## Matched Window-Level Check",
        "",
        f"- audited windows: `{win['num_windows']}`",
        f"- native `contact_force` nonzero windows: `{win['native_nonzero_windows']}`",
        f"- native `contact_flag` nonzero windows: `{win['native_flag_nonzero_windows']}`",
        f"- surrogate force-proxy max over those windows: `{win['surrogate_max']:.6f}`",
        f"- native `contact_force` max over those windows: `{win['native_max']:.6f}`",
        f"- Pearson surrogate/native correlation: `{win['pearson_surrogate_vs_native']}`",
        f"- Spearman surrogate/native correlation: `{win['spearman_surrogate_vs_native']}`",
        f"- top-100 window overlap fraction: `{win['top100_overlap_fraction']}`",
        "",
        "Because the native target has zero variance, correlation and rank-consistency are",
        "undefined rather than merely weak.",
        "",
        "## Consequence for the Paper",
        "",
        "- The current surrogate-contact analysis remains necessary.",
        "- We cannot run a meaningful native `contact_force` probe ranking with the public Step 0 Strike export.",
        "- The scientifically correct update is therefore a stronger data audit: native contact channels were rechecked and remain zero-filled, so the Tier-B claim still rests on a surrogate force proxy rather than simulator-native force supervision.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("/mnt/md1/solee/features/physprobe_vitl/strike"))
    parser.add_argument("--episode-limit", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, default=Path("/home/solee/physrepa_tasks/artifacts/results/surrogate_validation_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("/home/solee/physrepa_tasks/artifacts/results/surrogate_validation_verdict.md"))
    args = parser.parse_args()

    parquets = find_strike_parquets()
    frame_audit = frame_mag_stats(parquets)
    window_validation = window_level_validation(parquets, args.feature_root, args.episode_limit)
    summary = {
        "frame_audit": frame_audit,
        "window_validation": window_validation,
        "feature_root": str(args.feature_root),
        "episode_limit": args.episode_limit,
    }
    args.output_json.write_text(json.dumps(summary, indent=2))
    render_markdown(summary, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
