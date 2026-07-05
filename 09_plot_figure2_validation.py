#!/usr/bin/env python3
"""
Figure 2 — Validation-sensitivity / performance-degradation plot for Line 1 DES manuscript.

Purpose
-------
Builds a publication-ready figure showing how model performance changes as validation becomes
more realistic:
    B: leakage-corrected pair+ratio group
    C: strict HBA-HBD pair group
    D: leave-HBA-out
    D: leave-HBD-out

The script combines results from:
    audit_outputs/diagnostic_metrics_summary.csv
    protocol_D_extrapolative_outputs/protocol_D_metrics_summary.csv

Recommended use
---------------
!python plot_figure2_performance_degradation.py \
  --audit-dir audit_outputs \
  --protocol-d-dir protocol_D_extrapolative_outputs \
  --outdir figure2_outputs \
  --model extra_trees \
  --feature-set full \
  --d-feature-set full

Notes
-----
- Protocol A is excluded by default because it is a forensic/leaky diagnostic.
- Use --include-protocol-a only for Supplementary Information, not for the main manuscript.
- The script does not invent values; if a requested model/feature/protocol is missing, it reports it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROPERTY_ORDER = [
    "density",
    "refractive_index",
    "surface_tension",
    "conductivity",
    "viscosity",
]

PROPERTY_LABELS = {
    "density": "Density",
    "refractive_index": "Refractive index",
    "surface_tension": "Surface tension",
    "conductivity": "Conductivity",
    "viscosity": "Viscosity",
}

PROTOCOL_LABELS = {
    "A_old_like_potentially_leaky": "A: old-like\n(leaky)",
    "B_leakage_corrected_pair_ratio_group": "B: clean\npair+ratio",
    "C_strict_pair_group": "C: strict\npair",
    "D_leave_HBA_out": "D: leave\nHBA out",
    "D_leave_HBD_out": "D: leave\nHBD out",
}

MAIN_PROTOCOL_ORDER = [
    "B_leakage_corrected_pair_ratio_group",
    "C_strict_pair_group",
    "D_leave_HBA_out",
    "D_leave_HBD_out",
]

WITH_A_PROTOCOL_ORDER = ["A_old_like_potentially_leaky"] + MAIN_PROTOCOL_ORDER


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Required file is empty: {path}")
    return df


def normalize_model_name(name: str) -> str:
    aliases = {
        "extratrees": "extra_trees",
        "extra_trees": "extra_trees",
        "ExtraTrees": "extra_trees",
        "histgb": "hist_gradient_boosting",
        "HistGB": "hist_gradient_boosting",
        "hist_gradient_boosting": "hist_gradient_boosting",
        "ridge": "ridge",
        "Ridge": "ridge",
    }
    return aliases.get(name, name)


def filter_summary(
    df: pd.DataFrame,
    *,
    model: str,
    feature_set: str,
    protocols: List[str],
    source_name: str,
) -> pd.DataFrame:
    required_cols = {"task", "protocol", "feature_set", "model", "r2_mean", "r2_std", "mae_mean", "mae_std"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

    model = normalize_model_name(model)
    work = df.copy()
    work["model"] = work["model"].map(normalize_model_name)

    out = work[
        (work["model"] == model)
        & (work["feature_set"] == feature_set)
        & (work["protocol"].isin(protocols))
    ].copy()

    return out


def build_combined_table(
    audit_dir: Path,
    protocol_d_dir: Path,
    model: str,
    feature_set: str,
    d_feature_set: Optional[str],
    include_protocol_a: bool,
) -> pd.DataFrame:
    audit_path = audit_dir / "diagnostic_metrics_summary.csv"
    d_path = protocol_d_dir / "protocol_D_metrics_summary.csv"

    audit = read_csv_required(audit_path)
    protocol_d = read_csv_required(d_path)

    protocol_order = WITH_A_PROTOCOL_ORDER if include_protocol_a else MAIN_PROTOCOL_ORDER
    audit_protocols = [p for p in protocol_order if p.startswith("A_") or p.startswith("B_") or p.startswith("C_")]
    d_protocols = [p for p in protocol_order if p.startswith("D_")]

    d_feature_set = d_feature_set or feature_set

    parts = []
    if audit_protocols:
        parts.append(
            filter_summary(
                audit,
                model=model,
                feature_set=feature_set,
                protocols=audit_protocols,
                source_name=str(audit_path),
            )
        )
    if d_protocols:
        parts.append(
            filter_summary(
                protocol_d,
                model=model,
                feature_set=d_feature_set,
                protocols=d_protocols,
                source_name=str(d_path),
            )
        )

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if combined.empty:
        raise ValueError(
            "No matching rows found. Check --model, --feature-set, --d-feature-set, and protocol names."
        )

    combined["property"] = combined["task"].astype(str)
    combined["property_label"] = combined["property"].map(PROPERTY_LABELS).fillna(combined["property"])
    combined["protocol_label"] = combined["protocol"].map(PROTOCOL_LABELS).fillna(combined["protocol"])
    combined["protocol_order"] = combined["protocol"].map({p: i for i, p in enumerate(protocol_order)})
    combined["property_order"] = combined["property"].map({p: i for i, p in enumerate(PROPERTY_ORDER)}).fillna(999)

    combined = combined.sort_values(["property_order", "protocol_order", "model", "feature_set"])

    keep = [
        "property",
        "property_label",
        "protocol",
        "protocol_label",
        "model",
        "feature_set",
        "n_folds",
        "n_test_total",
        "n_features",
        "r2_mean",
        "r2_std",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
    ]
    keep = [c for c in keep if c in combined.columns]
    return combined[keep]


def report_missing(combined: pd.DataFrame, include_protocol_a: bool) -> str:
    protocol_order = WITH_A_PROTOCOL_ORDER if include_protocol_a else MAIN_PROTOCOL_ORDER
    rows = []
    for prop in PROPERTY_ORDER:
        present = set(combined.loc[combined["property"] == prop, "protocol"])
        missing = [p for p in protocol_order if p not in present]
        if missing:
            rows.append(f"- {PROPERTY_LABELS.get(prop, prop)} missing: {', '.join(missing)}")
    return "\n".join(rows) if rows else "No missing property/protocol combinations for predefined properties."


def make_line_plot(combined: pd.DataFrame, outdir: Path, include_protocol_a: bool, cap_y: bool) -> None:
    protocol_order = WITH_A_PROTOCOL_ORDER if include_protocol_a else MAIN_PROTOCOL_ORDER
    x = np.arange(len(protocol_order))

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for prop in PROPERTY_ORDER:
        sub = combined[combined["property"] == prop].copy()
        if sub.empty:
            continue
        sub = sub.set_index("protocol").reindex(protocol_order)
        y = sub["r2_mean"].to_numpy(dtype=float)
        yerr = sub["r2_std"].to_numpy(dtype=float) if "r2_std" in sub.columns else None
        label = PROPERTY_LABELS.get(prop, prop)
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS.get(p, p) for p in protocol_order])
    ax.set_ylabel("R² (mean ± SD)")
    ax.set_xlabel("Validation protocol")
    ax.set_title("Validation sensitivity across DES physicochemical properties")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.25)

    if cap_y:
        # Preserve negative extrapolation information without allowing extreme values to dominate the figure.
        ymin = max(-1.0, np.nanmin(combined["r2_mean"] - combined.get("r2_std", 0)) - 0.05)
        ymax = min(1.05, np.nanmax(combined["r2_mean"] + combined.get("r2_std", 0)) + 0.05)
        if ymin >= ymax:
            ymin, ymax = -0.2, 1.0
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(outdir / "Figure2_validation_degradation_line.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Figure2_validation_degradation_line.pdf", bbox_inches="tight")
    plt.close(fig)


def make_bar_plot(combined: pd.DataFrame, outdir: Path, include_protocol_a: bool, cap_y: bool) -> None:
    """Alternative journal-friendly grouped bar plot."""
    protocol_order = WITH_A_PROTOCOL_ORDER if include_protocol_a else MAIN_PROTOCOL_ORDER
    props = [p for p in PROPERTY_ORDER if p in set(combined["property"])]
    if not props:
        return

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    width = 0.16 if len(protocol_order) <= 4 else 0.13
    x = np.arange(len(props))

    for i, protocol in enumerate(protocol_order):
        vals = []
        errs = []
        for prop in props:
            row = combined[(combined["property"] == prop) & (combined["protocol"] == protocol)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(row.iloc[0]["r2_mean"]))
                errs.append(float(row.iloc[0].get("r2_std", 0.0)))
        offset = (i - (len(protocol_order) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=PROTOCOL_LABELS.get(protocol, protocol))

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([PROPERTY_LABELS.get(p, p) for p in props], rotation=20, ha="right")
    ax.set_ylabel("R² (mean ± SD)")
    ax.set_title("Performance degradation under increasingly realistic validation")
    ax.legend(frameon=False, ncols=2)
    ax.grid(True, axis="y", alpha=0.25)

    if cap_y:
        ymin = max(-1.0, np.nanmin(combined["r2_mean"] - combined.get("r2_std", 0)) - 0.05)
        ymax = min(1.05, np.nanmax(combined["r2_mean"] + combined.get("r2_std", 0)) + 0.05)
        if ymin >= ymax:
            ymin, ymax = -0.2, 1.0
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(outdir / "Figure2_validation_degradation_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Figure2_validation_degradation_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def write_caption(outdir: Path, include_protocol_a: bool, feature_set: str, d_feature_set: str, model: str) -> None:
    protocol_text = (
        "Protocol A (old-like/leaky diagnostic), Protocol B (leakage-corrected pair+ratio grouping), "
        "Protocol C (strict pair grouping), and Protocol D (leave-HBA-out and leave-HBD-out)"
        if include_protocol_a
        else "Protocol B (leakage-corrected pair+ratio grouping), Protocol C (strict pair grouping), and Protocol D (leave-HBA-out and leave-HBD-out)"
    )
    caption = f"""
Figure 2. Validation sensitivity of DES property prediction across increasingly stringent evaluation protocols. The plot reports cross-validated R² values for the {model} model using `{feature_set}` features for Protocols B/C and `{d_feature_set}` features for Protocol D. {protocol_text} are compared across density, refractive index, surface tension, electrical conductivity, and viscosity. Error bars denote standard deviations across folds. The systematic decrease in R² under stricter and extrapolative validation highlights property-dependent generalizability and the limited transferability of descriptor-based models, particularly for conductivity and viscosity.
""".strip()
    (outdir / "Figure2_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build Figure 2 validation degradation plots for the Line 1 DES manuscript.",
        epilog=textwrap.dedent(
            """
            Example:
              python plot_figure2_performance_degradation.py \
                --audit-dir audit_outputs \
                --protocol-d-dir protocol_D_extrapolative_outputs \
                --outdir figure2_outputs \
                --model extra_trees \
                --feature-set full \
                --d-feature-set full
            """
        ),
    )
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory containing diagnostic_metrics_summary.csv")
    parser.add_argument("--protocol-d-dir", type=Path, required=True, help="Directory containing protocol_D_metrics_summary.csv")
    parser.add_argument("--outdir", type=Path, default=Path("figure2_outputs"))
    parser.add_argument("--model", default="extra_trees", help="Model to plot, e.g. extra_trees or hist_gradient_boosting")
    parser.add_argument("--feature-set", default="full", help="Feature set for audit outputs Protocol B/C, e.g. full")
    parser.add_argument("--d-feature-set", default=None, help="Feature set for Protocol D. Defaults to --feature-set.")
    parser.add_argument("--include-protocol-a", action="store_true", help="Include old-like/leaky Protocol A. Recommended only for SI.")
    parser.add_argument("--no-y-cap", action="store_true", help="Do not cap y-axis to reduce domination by extreme negative R².")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    d_feature_set = args.d_feature_set or args.feature_set

    combined = build_combined_table(
        audit_dir=args.audit_dir,
        protocol_d_dir=args.protocol_d_dir,
        model=args.model,
        feature_set=args.feature_set,
        d_feature_set=d_feature_set,
        include_protocol_a=args.include_protocol_a,
    )

    combined.to_csv(args.outdir / "Figure2_validation_degradation_data.csv", index=False)

    missing_report = report_missing(combined, args.include_protocol_a)
    (args.outdir / "Figure2_missing_report.txt").write_text(missing_report + "\n", encoding="utf-8")

    make_line_plot(combined, args.outdir, args.include_protocol_a, cap_y=not args.no_y_cap)
    make_bar_plot(combined, args.outdir, args.include_protocol_a, cap_y=not args.no_y_cap)
    write_caption(args.outdir, args.include_protocol_a, args.feature_set, d_feature_set, normalize_model_name(args.model))

    print(f"Done. Outputs written to: {args.outdir}")
    print("Created:")
    print(" - Figure2_validation_degradation_line.png/pdf")
    print(" - Figure2_validation_degradation_bars.png/pdf")
    print(" - Figure2_validation_degradation_data.csv")
    print(" - Figure2_caption.txt")
    print(" - Figure2_missing_report.txt")
    print("\nMissing report:")
    print(missing_report)


if __name__ == "__main__":
    main()
