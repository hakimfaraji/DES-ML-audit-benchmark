#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4.1 — Figure 2 improvement
Adds explicit interpolation/extrapolation regime labels to the validation-sensitivity figure.

Input options:
  1) Provide a CSV using --table3 with columns similar to:
     Property, Protocol, Model, R2_mean, R2_sd
     or parsed strings like "R² (mean ± SD)".
  2) If --table3 is not provided, the script uses the current Table 3 values from Upgrade_Manuscript.pdf.

Output:
  Figure2_validation_sensitivity_phase4.png/pdf
  Figure2_validation_sensitivity_phase4_plot_data.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROPERTY_ORDER = ["Density", "Refractive Index", "Surface Tension", "Conductivity", "Viscosity"]
PROTO_ORDER = ["Pair+Ratio (B)", "Pair (C)", "Leave-HBA (D)", "Leave-HBD (D)"]
PROTO_LABELS = {
    "Pair+Ratio (B)": "B: pair+ratio\n(interpolation)",
    "Pair (C)": "C: pair\n(strict grouping)",
    "Leave-HBA (D)": "D: leave-HBA\n(extrapolation)",
    "Leave-HBD (D)": "D: leave-HBD\n(extrapolation)",
}

DEFAULT_ROWS = [
    ("Density", "Pair+Ratio (B)", 0.857, 0.035),
    ("Density", "Pair (C)", 0.825, 0.017),
    ("Density", "Leave-HBA (D)", 0.555, 0.292),
    ("Density", "Leave-HBD (D)", 0.600, 0.144),
    ("Refractive Index", "Pair+Ratio (B)", 0.839, 0.024),
    ("Refractive Index", "Pair (C)", 0.847, 0.032),
    ("Refractive Index", "Leave-HBA (D)", 0.837, 0.050),
    ("Refractive Index", "Leave-HBD (D)", 0.647, 0.208),
    ("Surface Tension", "Pair+Ratio (B)", 0.778, 0.064),
    ("Surface Tension", "Pair (C)", 0.614, 0.075),
    ("Surface Tension", "Leave-HBA (D)", -0.176, 1.465),
    ("Surface Tension", "Leave-HBD (D)", 0.397, 0.522),
    ("Conductivity", "Pair+Ratio (B)", 0.535, 0.121),
    ("Conductivity", "Pair (C)", 0.557, 0.120),
    ("Conductivity", "Leave-HBA (D)", -0.226, 0.555),
    ("Conductivity", "Leave-HBD (D)", -0.947, 2.044),
    ("Viscosity", "Pair+Ratio (B)", 0.145, 0.125),
    ("Viscosity", "Pair (C)", 0.168, 0.296),
    ("Viscosity", "Leave-HBA (D)", -3.625, 4.708),
    ("Viscosity", "Leave-HBD (D)", -2.653, 4.624),
]


def _parse_mean_sd(value: str) -> tuple[float, float]:
    """Parse values like '0.857 ± 0.035' or '0.857 +/- 0.035'."""
    s = str(value).strip().replace("−", "-")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) < 2:
        raise ValueError(f"Cannot parse mean±SD from: {value}")
    return float(nums[0]), float(nums[1])


def load_table3(path: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(DEFAULT_ROWS, columns=["Property", "Protocol", "R2_mean", "R2_sd"])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Normalize names
    prop_col = next((c for c in df.columns if c.lower().startswith("property")), None)
    proto_col = next((c for c in df.columns if c.lower().startswith("protocol")), None)
    if prop_col is None or proto_col is None:
        raise ValueError("Input table must contain Property and Protocol columns.")
    if "R2_mean" in df.columns and "R2_sd" in df.columns:
        out = df[[prop_col, proto_col, "R2_mean", "R2_sd"]].copy()
        out.columns = ["Property", "Protocol", "R2_mean", "R2_sd"]
    else:
        r2_col = next((c for c in df.columns if "r" in c.lower() and ("²" in c or "2" in c.lower()) and "mae" not in c.lower()), None)
        if r2_col is None:
            raise ValueError("Could not find R² column. Use columns R2_mean and R2_sd or a mean±SD R² column.")
        parsed = df[r2_col].apply(_parse_mean_sd)
        out = pd.DataFrame({
            "Property": df[prop_col],
            "Protocol": df[proto_col],
            "R2_mean": [x[0] for x in parsed],
            "R2_sd": [x[1] for x in parsed],
        })
    # Map protocol variants if needed
    proto_map = {
        "B_pair_ratio": "Pair+Ratio (B)", "B: pair+ratio": "Pair+Ratio (B)", "Pair+Ratio": "Pair+Ratio (B)",
        "C_pair": "Pair (C)", "C: pair": "Pair (C)", "Pair": "Pair (C)",
        "D_leave_HBA": "Leave-HBA (D)", "D: leave-HBA-out": "Leave-HBA (D)", "Leave-HBA": "Leave-HBA (D)",
        "D_leave_HBD": "Leave-HBD (D)", "D: leave-HBD-out": "Leave-HBD (D)", "Leave-HBD": "Leave-HBD (D)",
    }
    out["Protocol"] = out["Protocol"].astype(str).str.strip().replace(proto_map)
    out["Property"] = out["Property"].astype(str).str.strip().replace({"Refractive index": "Refractive Index", "Surface tension": "Surface Tension"})
    return out


def plot(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = df[df["Property"].isin(PROPERTY_ORDER) & df["Protocol"].isin(PROTO_ORDER)].copy()
    df["Property"] = pd.Categorical(df["Property"], PROPERTY_ORDER, ordered=True)
    df["Protocol"] = pd.Categorical(df["Protocol"], PROTO_ORDER, ordered=True)
    df = df.sort_values(["Property", "Protocol"])
    df.to_csv(outdir / "Figure2_validation_sensitivity_phase4_plot_data.csv", index=False)

    x = np.arange(len(PROPERTY_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5*width, 1.5*width, len(PROTO_ORDER))

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, proto in enumerate(PROTO_ORDER):
        sub = df[df["Protocol"] == proto].set_index("Property").reindex(PROPERTY_ORDER)
        ax.bar(x + offsets[i], sub["R2_mean"], width, yerr=sub["R2_sd"], capsize=3, label=PROTO_LABELS[proto], alpha=0.92)

    ax.axhline(0, linestyle="--", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(PROPERTY_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Cross-validated R² (mean ± SD)")
    ax.set_title("Validation sensitivity across DES property prediction tasks")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(-1.15, 1.18)

    # Regime labels: B/C versus D protocols
    ax.text(0.22, 1.055, "Interpolation-oriented regime\n(B and C protocols)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.9))
    ax.text(0.74, 1.055, "Extrapolation regime\n(leave-component-out D)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.9))

    ax.legend(ncol=2, fontsize=8.5, frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "Figure2_validation_sensitivity_phase4.png", dpi=450)
    fig.savefig(outdir / "Figure2_validation_sensitivity_phase4.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table3", default=None, help="Optional CSV for Table 3 values. If omitted, uses current manuscript Table 3 values.")
    ap.add_argument("--outdir", default="phase4_figures")
    args = ap.parse_args()
    df = load_table3(args.table3)
    plot(df, Path(args.outdir))
    print(f"[DONE] Outputs written to {args.outdir}")

if __name__ == "__main__":
    main()
