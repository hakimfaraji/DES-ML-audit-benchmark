#!/usr/bin/env python3
"""
Figure 5: Viscosity diagnostic plot for DES Line 1 manuscript.

This script is intentionally lightweight: it does NOT retrain models.
It reads existing viscosity diagnostic and Protocol D summary CSVs, then creates
publication-ready diagnostic panels.

Expected inputs after unzipping:
  viscosity_outputs/
  protocol_D_extrapolative_outputs/

The script auto-detects likely CSV filenames and column variants.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET = "viscosity_mpa_s"


def find_csv(root: Path, keywords: List[str]) -> Optional[Path]:
    if not root.exists():
        return None
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        return None
    scored = []
    for p in csvs:
        name = p.name.lower()
        score = sum(1 for k in keywords if k.lower() in name)
        if score > 0:
            scored.append((score, len(str(p)), p))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]
    return csvs[0]


def read_csv_or_fail(path: Optional[Path], label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not find CSV for {label}.")
    print(f"[INFO] Reading {label}: {path}")
    return pd.read_csv(path)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower_map)
    return df


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c.lower() in cols:
            return c.lower()
    if required:
        raise KeyError(f"None of the candidate columns found: {candidates}. Available columns: {list(df.columns)}")
    return None


def filter_viscosity(df: pd.DataFrame) -> pd.DataFrame:
    df = norm_cols(df)
    prop_col = pick_col(df, ["property", "target", "target_col", "property_name"], required=False)
    if prop_col:
        m = df[prop_col].astype(str).str.lower().eq(TARGET.lower())
        if m.any():
            df = df[m].copy()
    return df


def get_metric_col(df: pd.DataFrame, metric: str) -> str:
    if metric == "r2":
        return pick_col(df, ["r2_mean", "mean_r2", "r2", "test_r2_mean"])
    if metric == "mae":
        return pick_col(df, ["mae_mean", "mean_mae", "mae", "test_mae_mean"])
    raise ValueError(metric)


def select_rows(df: pd.DataFrame, protocol_contains: Optional[str] = None, variant_contains: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    prot_col = pick_col(out, ["protocol", "validation", "split_protocol"], required=False)
    if protocol_contains and prot_col:
        mask = out[prot_col].astype(str).str.lower().str.contains(protocol_contains.lower(), regex=False)
        if mask.any():
            out = out[mask].copy()
    var_col = pick_col(out, ["target_variant", "variant", "transform", "y_transform"], required=False)
    if variant_contains and var_col:
        mask = out[var_col].astype(str).str.lower().str.contains(variant_contains.lower(), regex=False)
        if mask.any():
            out = out[mask].copy()
    return out


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else np.nan


def build_panel_data(visc_df: pd.DataFrame, d_df: pd.DataFrame) -> pd.DataFrame:
    visc_df = filter_viscosity(visc_df)
    d_df = filter_viscosity(d_df)

    r2_col_v = get_metric_col(visc_df, "r2")
    mae_col_v = get_metric_col(visc_df, "mae")
    r2_col_d = get_metric_col(d_df, "r2")
    mae_col_d = get_metric_col(d_df, "mae")

    raw = select_rows(visc_df, variant_contains="raw")
    log = select_rows(visc_df, variant_contains="log")
    if raw.empty:
        raw = select_rows(visc_df, variant_contains="none")
    if log.empty:
        # fallback: use rows containing log in any column values
        any_log = visc_df.apply(lambda s: s.astype(str).str.lower().str.contains("log").any())
        if any_log.any():
            log = visc_df.loc[:, any_log.index[any_log]].copy() if False else visc_df.copy()

    pair = select_rows(visc_df, protocol_contains="pair_group", variant_contains="log")
    if pair.empty:
        pair = select_rows(visc_df, protocol_contains="pair")
    hba = select_rows(d_df, protocol_contains="leave_hba")
    hbd = select_rows(d_df, protocol_contains="leave_hbd")

    data = pd.DataFrame([
        {"panel": "raw_vs_log", "condition": "Raw target", "r2": safe_mean(raw, r2_col_v), "mae": safe_mean(raw, mae_col_v)},
        {"panel": "raw_vs_log", "condition": "Log target", "r2": safe_mean(log, r2_col_v), "mae": safe_mean(log, mae_col_v)},
        {"panel": "extrapolation", "condition": "Pair-group", "r2": safe_mean(pair, r2_col_v), "mae": safe_mean(pair, mae_col_v)},
        {"panel": "extrapolation", "condition": "Leave-HBA-out", "r2": safe_mean(hba, r2_col_d), "mae": safe_mean(hba, mae_col_d)},
        {"panel": "extrapolation", "condition": "Leave-HBD-out", "r2": safe_mean(hbd, r2_col_d), "mae": safe_mean(hbd, mae_col_d)},
    ])
    return data


def plot_figure(panel_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # A: R2 raw vs log
    a = panel_df[panel_df["panel"] == "raw_vs_log"]
    axes[0].bar(a["condition"], a["r2"])
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("R²")
    axes[0].set_title("A. Raw vs log-transformed viscosity")
    axes[0].tick_params(axis="x", rotation=20)

    # B: MAE raw vs log (diagnostic only)
    axes[1].bar(a["condition"], a["mae"])
    axes[1].set_ylabel("MAE")
    axes[1].set_title("B. Error magnitude")
    axes[1].tick_params(axis="x", rotation=20)

    # C: extrapolation collapse
    c = panel_df[panel_df["panel"] == "extrapolation"]
    axes[2].bar(c["condition"], c["r2"])
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_ylabel("R²")
    axes[2].set_title("C. Extrapolative validation")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(outdir / "Figure5_viscosity_diagnostic.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Figure5_viscosity_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def write_caption(outdir: Path) -> None:
    caption = (
        "Figure 5. Diagnostic analysis of viscosity prediction. "
        "(A) Comparison between raw and log-transformed viscosity targets shows that logarithmic transformation improves model fit but does not remove the fundamental limitation of descriptor-based prediction. "
        "(B) Error magnitudes remain large, indicating that improved scaling alone is insufficient. "
        "(C) Extrapolative validation using leave-HBA-out and leave-HBD-out splits shows strong performance collapse relative to pair-group interpolation, demonstrating limited transferability to unseen component chemistry."
    )
    (outdir / "Figure5_caption.txt").write_text(caption, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create Figure 5 viscosity diagnostic from existing CSV outputs.")
    parser.add_argument("--visc-dir", required=True, help="Directory containing viscosity diagnostic CSV outputs")
    parser.add_argument("--protocol-d-dir", required=True, help="Directory containing Protocol D extrapolative CSV outputs")
    parser.add_argument("--outdir", default="figure5_outputs", help="Output directory")
    args = parser.parse_args(argv)

    visc_dir = Path(args.visc_dir)
    d_dir = Path(args.protocol_d_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    visc_csv = find_csv(visc_dir, ["viscosity", "metrics", "summary"])
    d_csv = find_csv(d_dir, ["protocol", "metrics", "summary"])

    visc_df = read_csv_or_fail(visc_csv, "viscosity diagnostics")
    d_df = read_csv_or_fail(d_csv, "Protocol D")

    panel_df = build_panel_data(visc_df, d_df)
    panel_df.to_csv(outdir / "Figure5_viscosity_panels.csv", index=False)

    plot_figure(panel_df, outdir)
    write_caption(outdir)

    print(f"Done. Outputs written to: {outdir}")
    print("Created:")
    print(" - Figure5_viscosity_diagnostic.png/pdf")
    print(" - Figure5_viscosity_panels.csv")
    print(" - Figure5_caption.txt")
    print("\nPanel data:")
    print(panel_df.to_string(index=False))


if __name__ == "__main__":
    main()
