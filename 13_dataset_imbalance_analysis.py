# ============================================================
# Dataset imbalance analysis for Line 1 DES ML manuscript upgrade
# Project: AI_DES formation model — Line 1
#
# Input:
#   Unified_DES_dataset_GOLD_descriptor_ready_subset.csv
#
# Output directory:
#   dataset_imbalance_outputs/
#
# Main outputs:
#   HBA_frequency_overall.csv
#   HBD_frequency_overall.csv
#   HBA_frequency_by_property.csv
#   HBD_frequency_by_property.csv
#   ratio_frequency_overall.csv
#   temperature_summary_by_property.csv
#   imbalance_summary.csv
#   imbalance_summary.json
#   FigureS_dataset_imbalance.png
#   FigureS_dataset_imbalance.pdf
#
# Purpose:
#   Quantify chemical/compositional/temperature imbalance in the frozen GOLD dataset.
#   Designed for SI Section S4.
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROPERTY_TARGETS: Dict[str, str] = {
    "Density": "density_g_cm3",
    "Viscosity": "viscosity_mpa_s",
    "Conductivity": "conductivity_ms_cm",
    "Surface tension": "surface_tension_mn_m",
    "Refractive index": "refractive_index",
}

HBA_CANDIDATES = [
    "hba_name_raw",
    "hba_name_resolved",
    "hba_name_canonical",
    "hba_canonical_name",
]
HBD_CANDIDATES = [
    "hbd_name_raw",
    "hbd_name_resolved",
    "hbd_name_canonical",
    "hbd_canonical_name",
]
TEMP_CANDIDATES_C = ["measurement_temperature_c"]
RATIO_CANDIDATES = ["molar_ratio_raw", "molar_ratio", "ratio_hbd_hba", "parsed_ratio"]


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of the expected columns found: {candidates}")
    return None


def clean_name(x) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "UNKNOWN"
    return s


def parse_ratio_hbd_hba(value) -> float:
    """
    Parse molar_ratio_raw into HBD/HBA numeric ratio.

    Handles common forms such as:
      1:2 -> 2.0
      1/2 -> 2.0 if written as HBA/HBD stoichiometric pair
      2 -> 2.0
      HBA:HBD = 1:2 -> 2.0
      [1, 2] / (1,2) -> 2.0

    The manuscript Table 1 defines ratio as HBD/HBA.
    For two-number ratios, first number is interpreted as HBA and second as HBD.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan

    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return np.nan

    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("：", ":").replace(";", ":")
    # Remove common labels but keep numbers and separators.
    s2 = re.sub(r"[a-zA-Z_()\[\]{}=]+", " ", s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s2)

    if len(nums) >= 2:
        a = float(nums[0])
        b = float(nums[1])
        if a == 0:
            return np.nan
        return b / a
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def gini_from_counts(counts: np.ndarray) -> float:
    """Gini coefficient for non-negative frequency counts."""
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if len(x) == 0 or np.sum(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def frequency_table(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    counts = df[col].map(clean_name).value_counts(dropna=False).rename_axis(label).reset_index(name="count")
    total = counts["count"].sum()
    counts["share"] = counts["count"] / total if total else np.nan
    counts["cumulative_share"] = counts["share"].cumsum()
    counts["rank"] = np.arange(1, len(counts) + 1)
    return counts[["rank", label, "count", "share", "cumulative_share"]]


def summarize_component(freq: pd.DataFrame, name_col: str, prefix: str) -> Dict[str, float]:
    counts = freq["count"].to_numpy(dtype=float)
    total = float(np.sum(counts))
    return {
        f"{prefix}_unique": int(len(freq)),
        f"{prefix}_top1_share": float(freq["share"].iloc[0]) if len(freq) else np.nan,
        f"{prefix}_top5_share": float(freq["share"].head(5).sum()) if len(freq) else np.nan,
        f"{prefix}_top10_share": float(freq["share"].head(10).sum()) if len(freq) else np.nan,
        f"{prefix}_singleton_count": int(np.sum(counts == 1)),
        f"{prefix}_singleton_share_of_components": float(np.mean(counts == 1)) if len(counts) else np.nan,
        f"{prefix}_gini": gini_from_counts(counts),
        f"{prefix}_most_common": str(freq[name_col].iloc[0]) if len(freq) else "NA",
        f"{prefix}_most_common_count": int(freq["count"].iloc[0]) if len(freq) else 0,
    }


def build_property_long(df: pd.DataFrame, hba_col: str, hbd_col: str, temp_col: str, ratio_col: str) -> pd.DataFrame:
    rows = []
    for prop, target in PROPERTY_TARGETS.items():
        if target not in df.columns:
            continue
        sub = df[df[target].notna()].copy()
        if sub.empty:
            continue
        sub["property"] = prop
        sub["target_column"] = target
        sub["hba"] = sub[hba_col].map(clean_name)
        sub["hbd"] = sub[hbd_col].map(clean_name)
        sub["temperature_c"] = pd.to_numeric(sub[temp_col], errors="coerce")
        sub["temperature_k"] = sub["temperature_c"] + 273.15
        sub["ratio_hbd_hba"] = sub[ratio_col].apply(parse_ratio_hbd_hba)
        keep = [
            "property", "target_column", "hba", "hbd", "temperature_c", "temperature_k",
            "ratio_hbd_hba", target,
        ]
        if "unified_row_id" in sub.columns:
            keep.insert(0, "unified_row_id")
        rows.append(sub[keep])
    if not rows:
        raise ValueError("No property-specific rows found. Check target columns.")
    return pd.concat(rows, ignore_index=True)


def make_plots(long_df: pd.DataFrame, overall_hba: pd.DataFrame, overall_hbd: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # HBA rank-frequency
    ax = axes[0, 0]
    ax.bar(overall_hba["rank"], overall_hba["count"])
    ax.set_title("HBA rank-frequency distribution")
    ax.set_xlabel("HBA rank")
    ax.set_ylabel("Record count")
    ax.set_yscale("log")

    # HBD rank-frequency
    ax = axes[0, 1]
    ax.bar(overall_hbd["rank"], overall_hbd["count"])
    ax.set_title("HBD rank-frequency distribution")
    ax.set_xlabel("HBD rank")
    ax.set_ylabel("Record count")
    ax.set_yscale("log")

    # Ratio distribution
    ax = axes[1, 0]
    ratio = long_df["ratio_hbd_hba"].replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[(ratio > 0) & (ratio <= ratio.quantile(0.995))]
    ax.hist(ratio, bins=30)
    ax.set_title("Molar-ratio distribution (HBD/HBA)")
    ax.set_xlabel("HBD/HBA molar ratio")
    ax.set_ylabel("Record count")

    # Temperature by property
    ax = axes[1, 1]
    properties = [p for p in PROPERTY_TARGETS if p in long_df["property"].unique()]
    temp_data = [long_df.loc[long_df["property"] == p, "temperature_k"].dropna().to_numpy() for p in properties]
    ax.boxplot(temp_data, labels=properties, showfliers=False)
    ax.set_title("Temperature coverage by property")
    ax.set_xlabel("Property")
    ax.set_ylabel("Temperature (K)")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(outdir / "FigureS_dataset_imbalance.png", bbox_inches="tight")
    fig.savefig(outdir / "FigureS_dataset_imbalance.pdf", bbox_inches="tight")
    plt.close(fig)

    # Optional top-10 component dominance figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(overall_hba.head(10)["HBA"][::-1], overall_hba.head(10)["share"][::-1] * 100)
    axes[0].set_title("Top-10 HBA contribution")
    axes[0].set_xlabel("Share of records (%)")
    axes[1].barh(overall_hbd.head(10)["HBD"][::-1], overall_hbd.head(10)["share"][::-1] * 100)
    axes[1].set_title("Top-10 HBD contribution")
    axes[1].set_xlabel("Share of records (%)")
    fig.tight_layout()
    fig.savefig(outdir / "FigureS_top10_component_dominance.png", bbox_inches="tight")
    fig.savefig(outdir / "FigureS_top10_component_dominance.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify dataset imbalance for frozen GOLD DES dataset.")
    parser.add_argument("--input", required=True, help="Path to Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--outdir", default="dataset_imbalance_outputs", help="Output directory")
    parser.add_argument("--hba-col", default=None, help="Override HBA column")
    parser.add_argument("--hbd-col", default=None, help="Override HBD column")
    parser.add_argument("--ratio-col", default=None, help="Override molar ratio column")
    parser.add_argument("--temp-col", default=None, help="Override temperature column in Celsius")
    args = parser.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inpath)

    hba_col = args.hba_col or pick_col(df, HBA_CANDIDATES)
    hbd_col = args.hbd_col or pick_col(df, HBD_CANDIDATES)
    ratio_col = args.ratio_col or pick_col(df, RATIO_CANDIDATES)
    temp_col = args.temp_col or pick_col(df, TEMP_CANDIDATES_C)

    long_df = build_property_long(df, hba_col, hbd_col, temp_col, ratio_col)
    long_df.to_csv(outdir / "property_long_dataset_for_imbalance.csv", index=False)

    overall_hba = frequency_table(long_df, "hba", "HBA")
    overall_hbd = frequency_table(long_df, "hbd", "HBD")
    overall_hba.to_csv(outdir / "HBA_frequency_overall.csv", index=False)
    overall_hbd.to_csv(outdir / "HBD_frequency_overall.csv", index=False)

    by_prop_hba = []
    by_prop_hbd = []
    summaries = []

    for prop in [p for p in PROPERTY_TARGETS if p in long_df["property"].unique()]:
        sub = long_df[long_df["property"] == prop].copy()
        hba_freq = frequency_table(sub, "hba", "HBA")
        hbd_freq = frequency_table(sub, "hbd", "HBD")
        hba_freq.insert(0, "property", prop)
        hbd_freq.insert(0, "property", prop)
        by_prop_hba.append(hba_freq)
        by_prop_hbd.append(hbd_freq)

        summary = {
            "property": prop,
            "N": int(len(sub)),
            "temperature_k_min": float(sub["temperature_k"].min()),
            "temperature_k_max": float(sub["temperature_k"].max()),
            "temperature_k_median": float(sub["temperature_k"].median()),
            "ratio_min": float(sub["ratio_hbd_hba"].min()),
            "ratio_max": float(sub["ratio_hbd_hba"].max()),
            "ratio_median": float(sub["ratio_hbd_hba"].median()),
            "unique_hba_hbd_pairs": int(sub[["hba", "hbd"]].drop_duplicates().shape[0]),
        }
        summary.update(summarize_component(hba_freq, "HBA", "HBA"))
        summary.update(summarize_component(hbd_freq, "HBD", "HBD"))
        summaries.append(summary)

    hba_by_property = pd.concat(by_prop_hba, ignore_index=True)
    hbd_by_property = pd.concat(by_prop_hbd, ignore_index=True)
    hba_by_property.to_csv(outdir / "HBA_frequency_by_property.csv", index=False)
    hbd_by_property.to_csv(outdir / "HBD_frequency_by_property.csv", index=False)

    ratio_freq = long_df["ratio_hbd_hba"].round(4).value_counts(dropna=False).rename_axis("ratio_hbd_hba").reset_index(name="count")
    ratio_freq["share"] = ratio_freq["count"] / ratio_freq["count"].sum()
    ratio_freq.to_csv(outdir / "ratio_frequency_overall.csv", index=False)

    temp_summary = long_df.groupby("property")["temperature_k"].describe().reset_index()
    temp_summary.to_csv(outdir / "temperature_summary_by_property.csv", index=False)

    # Overall row
    overall_summary = {
        "property": "Overall_property_rows",
        "N": int(len(long_df)),
        "temperature_k_min": float(long_df["temperature_k"].min()),
        "temperature_k_max": float(long_df["temperature_k"].max()),
        "temperature_k_median": float(long_df["temperature_k"].median()),
        "ratio_min": float(long_df["ratio_hbd_hba"].min()),
        "ratio_max": float(long_df["ratio_hbd_hba"].max()),
        "ratio_median": float(long_df["ratio_hbd_hba"].median()),
        "unique_hba_hbd_pairs": int(long_df[["hba", "hbd"]].drop_duplicates().shape[0]),
    }
    overall_summary.update(summarize_component(overall_hba, "HBA", "HBA"))
    overall_summary.update(summarize_component(overall_hbd, "HBD", "HBD"))
    summaries.insert(0, overall_summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "imbalance_summary.csv", index=False)
    with open(outdir / "imbalance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    make_plots(long_df, overall_hba, overall_hbd, outdir)

    print("[DONE] Dataset imbalance analysis complete.")
    print(f"Input: {inpath}")
    print(f"Output directory: {outdir.resolve()}")
    print("Columns used:")
    print(f"  HBA: {hba_col}")
    print(f"  HBD: {hbd_col}")
    print(f"  Ratio: {ratio_col}")
    print(f"  Temperature C: {temp_col}")
    print("Key outputs:")
    for name in [
        "imbalance_summary.csv",
        "imbalance_summary.json",
        "HBA_frequency_overall.csv",
        "HBD_frequency_overall.csv",
        "HBA_frequency_by_property.csv",
        "HBD_frequency_by_property.csv",
        "FigureS_dataset_imbalance.png",
        "FigureS_top10_component_dominance.png",
    ]:
        print(f"  - {outdir / name}")


if __name__ == "__main__":
    main()
