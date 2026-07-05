#!/usr/bin/env python3
"""
Build Figure S1 for the Supplementary Information:
Dataset imbalance and coverage across DES systems.

Input:
  Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Outputs:
  FigureS1_dataset_imbalance_and_coverage.png
  FigureS1_dataset_imbalance_and_coverage.pdf
  FigureS1_source_counts_summary.csv

Design choices:
  - Uses property-specific GOLD records: one row per non-null target property.
  - HBA/HBD counts are computed from the expanded property-record table, so the
    frequencies match the overall property-row concept used in the imbalance audit.
  - Ratio is parsed as HBD/HBA from molar_ratio_raw, unless molar_ratio_numeric exists.
  - Temperature is reported in K as measurement_temperature_c + 273.15, unless
    temperature_k exists.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROPERTY_TARGETS = {
    "Density": "density_g_cm3",
    "Viscosity": "viscosity_mpa_s",
    "Conductivity": "conductivity_ms_cm",
    "Surface tension": "surface_tension_mn_m",
    "Refractive index": "refractive_index",
}


def parse_hbd_hba_ratio(value) -> float:
    """Parse HBD/HBA molar ratio from common raw formats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x if math.isfinite(x) else np.nan

    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "na", "n/a"}:
        return np.nan

    # Remove common textual clutter.
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace(" ", "")
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("hba:hbd=", "").replace("hba/hbd=", "")
    s = s.replace("hba:hbd", "").replace("hbd/hba", "")

    # Typical DES notation in this dataset is HBA:HBD, so HBD/HBA = right/left.
    m = re.match(r"^([0-9]*\.?[0-9]+):([0-9]*\.?[0-9]+)$", s)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        return right / left if left != 0 else np.nan

    # Formats like 1/2 are treated similarly to 1:2.
    m = re.match(r"^([0-9]*\.?[0-9]+)/([0-9]*\.?[0-9]+)$", s)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        return right / left if left != 0 else np.nan

    # First standalone number fallback.
    nums = re.findall(r"[0-9]*\.?[0-9]+", s)
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) >= 2:
        left, right = float(nums[0]), float(nums[1])
        return right / left if left != 0 else np.nan
    return np.nan


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the required columns were found: {candidates}")


def build_property_long_table(df: pd.DataFrame) -> pd.DataFrame:
    hba_col = first_existing(df, ["hba_name_resolved", "hba_name_canonical", "hba_name_raw"])
    hbd_col = first_existing(df, ["hbd_name_resolved", "hbd_name_canonical", "hbd_name_raw"])

    if "molar_ratio_numeric" in df.columns:
        ratio = pd.to_numeric(df["molar_ratio_numeric"], errors="coerce")
    else:
        ratio = df["molar_ratio_raw"].apply(parse_hbd_hba_ratio)

    if "temperature_k" in df.columns:
        temp_k = pd.to_numeric(df["temperature_k"], errors="coerce")
    else:
        temp_c_col = first_existing(df, ["measurement_temperature_c", "temperature_c"])
        temp_k = pd.to_numeric(df[temp_c_col], errors="coerce") + 273.15

    records = []
    for prop, target_col in PROPERTY_TARGETS.items():
        if target_col not in df.columns:
            continue
        mask = pd.to_numeric(df[target_col], errors="coerce").notna()
        sub = pd.DataFrame({
            "Property": prop,
            "HBA": df.loc[mask, hba_col].astype(str).str.strip(),
            "HBD": df.loc[mask, hbd_col].astype(str).str.strip(),
            "Ratio_HBD_HBA": ratio.loc[mask],
            "Temperature_K": temp_k.loc[mask],
            "Target": pd.to_numeric(df.loc[mask, target_col], errors="coerce"),
        })
        sub = sub.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        records.append(sub)
    if not records:
        raise ValueError("No property target columns were found or no non-null property records exist.")
    long_df = pd.concat(records, ignore_index=True)
    return long_df


def rank_counts(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).value_counts().sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Unified GOLD descriptor-ready CSV")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    long_df = build_property_long_table(df)

    hba_counts = rank_counts(long_df["HBA"])
    hbd_counts = rank_counts(long_df["HBD"])
    ratio_values = pd.to_numeric(long_df["Ratio_HBD_HBA"], errors="coerce").dropna()
    temp_by_prop = [
        pd.to_numeric(long_df.loc[long_df["Property"] == prop, "Temperature_K"], errors="coerce").dropna().values
        for prop in PROPERTY_TARGETS.keys()
    ]
    prop_labels = list(PROPERTY_TARGETS.keys())

    # Save a compact source summary for reproducibility.
    summary_rows = []
    for prop in ["Overall"] + prop_labels:
        sub = long_df if prop == "Overall" else long_df[long_df["Property"] == prop]
        summary_rows.append({
            "Property": prop,
            "N_property_records": len(sub),
            "Unique_HBA": sub["HBA"].dropna().nunique(),
            "Unique_HBD": sub["HBD"].dropna().nunique(),
            "Unique_HBA_HBD_pairs": sub.dropna(subset=["HBA", "HBD"]).drop_duplicates(["HBA", "HBD"]).shape[0],
            "Median_ratio_HBD_HBA": pd.to_numeric(sub["Ratio_HBD_HBA"], errors="coerce").median(),
            "Median_temperature_K": pd.to_numeric(sub["Temperature_K"], errors="coerce").median(),
        })
    pd.DataFrame(summary_rows).to_csv(outdir / "FigureS1_source_counts_summary.csv", index=False)

    # Figure layout.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel A: HBA rank-frequency.
    ax1.bar(np.arange(1, len(hba_counts) + 1), hba_counts.values, width=0.82)
    ax1.set_yscale("log")
    ax1.set_xlabel("HBA rank")
    ax1.set_ylabel("Record count")
    ax1.set_title("A. HBA rank-frequency distribution", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_xlim(0, len(hba_counts) + 2)

    # Panel B: HBD rank-frequency.
    ax2.bar(np.arange(1, len(hbd_counts) + 1), hbd_counts.values, width=0.82)
    ax2.set_yscale("log")
    ax2.set_xlabel("HBD rank")
    ax2.set_ylabel("Record count")
    ax2.set_title("B. HBD rank-frequency distribution", loc="left", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_xlim(0, len(hbd_counts) + 2)

    # Panel C: Ratio distribution. Use explicit bins to keep the long tail visible.
    finite_ratio = ratio_values[np.isfinite(ratio_values)]
    finite_ratio = finite_ratio[(finite_ratio >= 0) & (finite_ratio <= 20.0)]
    bins = np.concatenate([np.arange(0, 8.5, 0.5), np.array([10, 12.5, 15, 17.5, 20.5])])
    ax3.hist(finite_ratio, bins=bins, edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("HBD/HBA molar ratio")
    ax3.set_ylabel("Record count")
    ax3.set_title("C. Molar-ratio distribution", loc="left", fontweight="bold")
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_xlim(0, 20.5)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel D: Temperature coverage by property.
    bp = ax4.boxplot(temp_by_prop, labels=prop_labels, showfliers=False, patch_artist=False)
    ax4.set_ylabel("Temperature (K)")
    ax4.set_title("D. Temperature coverage by property", loc="left", fontweight="bold")
    ax4.grid(axis="y", alpha=0.25)
    ax4.tick_params(axis="x", rotation=25)

    fig.suptitle("Dataset imbalance and coverage across DES systems", fontsize=15, fontweight="bold")

    png = outdir / "FigureS1_dataset_imbalance_and_coverage.png"
    pdf = outdir / "FigureS1_dataset_imbalance_and_coverage.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("[DONE] Figure S1 written:")
    print(f" - {png}")
    print(f" - {pdf}")
    print(f" - {outdir / 'FigureS1_source_counts_summary.csv'}")


if __name__ == "__main__":
    main()
