#!/usr/bin/env python3
"""
Build a publication-ready Figure 1 workflow diagram for the leakage-aware DES ML audit.

Outputs:
  Figure1_leakage_aware_audit_workflow.png
  Figure1_leakage_aware_audit_workflow.pdf

Usage:
  python build_main_figure1_audit_workflow.py --outdir main_final_figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, w, h, text, fc="#F7F9FC", ec="#2F3A4A", lw=1.2, fontsize=9, weight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color="#1F2933")
    return box


def arrow(ax, start, end, color="#5B677A", lw=1.25, rad=0.0):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=5, shrinkB=5
    )
    ax.add_patch(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="main_final_figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Palette
    blue = "#EAF2FF"
    green = "#EAF7EF"
    amber = "#FFF6E5"
    purple = "#F1EDFF"
    rose = "#FFF0F0"
    gray = "#F7F9FC"

    # Title strip
    add_box(ax, (0.18, 0.90), 0.64, 0.075,
            "Leakage-aware audit workflow for DES property prediction",
            fc="#EEF4FF", ec="#375A9E", lw=1.4, fontsize=12, weight="bold")

    # Top row
    add_box(ax, (0.06, 0.74), 0.20, 0.10, "Curated DES dataset\nGOLD subset", fc=blue, ec="#2B5C9A", fontsize=10, weight="bold")
    add_box(ax, (0.40, 0.74), 0.20, 0.10, "Static descriptor\nrepresentation", fc=green, ec="#2E7D4F", fontsize=10, weight="bold")
    add_box(ax, (0.74, 0.74), 0.20, 0.10, "Property-specific\nML models", fc=purple, ec="#5B47A5", fontsize=10, weight="bold")
    arrow(ax, (0.26, 0.79), (0.40, 0.79))
    arrow(ax, (0.60, 0.79), (0.74, 0.79))

    # Validation center
    add_box(ax, (0.30, 0.55), 0.40, 0.105,
            "Validation hierarchy\nA diagnostic  →  B leakage-corrected  →  C group-aware  →  D extrapolative",
            fc=amber, ec="#B7791F", fontsize=9.5, weight="bold")
    arrow(ax, (0.84, 0.74), (0.61, 0.655), rad=0.05)
    arrow(ax, (0.50, 0.74), (0.50, 0.655))

    # Audit modules
    y = 0.35
    boxes = [
        (0.05, y, 0.18, 0.105, "Leakage audit\nfeature checks", rose, "#B94A48"),
        (0.29, y, 0.18, 0.105, "Baseline comparison\ntemperature / ratio", gray, "#4B5563"),
        (0.53, y, 0.18, 0.105, "Feature ablation\ndescriptors / T / ratio", gray, "#4B5563"),
        (0.77, y, 0.18, 0.105, "Interpretability\npermutation + SHAP", gray, "#4B5563"),
    ]
    for x, yy, w, h, text, fc, ec in boxes:
        add_box(ax, (x, yy), w, h, text, fc=fc, ec=ec, fontsize=9.2, weight="bold")
        arrow(ax, (0.50, 0.55), (x+w/2, yy+h), rad=0.0)

    # New evidence modules
    add_box(ax, (0.19, 0.14), 0.25, 0.105, "Dataset imbalance\ncomponent and ratio coverage", fc="#E9FBF7", ec="#16836A", fontsize=9.2, weight="bold")
    add_box(ax, (0.56, 0.14), 0.25, 0.105, "Applicability domain\ndistance + nearest neighbor", fc="#F5F0FF", ec="#7353BA", fontsize=9.2, weight="bold")
    arrow(ax, (0.14, y), (0.29, 0.245), rad=-0.05)
    arrow(ax, (0.38, y), (0.32, 0.245), rad=0.0)
    arrow(ax, (0.62, y), (0.68, 0.245), rad=0.0)
    arrow(ax, (0.86, y), (0.69, 0.245), rad=0.05)

    # Final conclusion box
    add_box(ax, (0.30, 0.02), 0.40, 0.075,
            "Property-dependent learnability and transferability limits",
            fc="#F8FAFC", ec="#111827", fontsize=10, weight="bold")
    arrow(ax, (0.315, 0.14), (0.45, 0.095), rad=0.0)
    arrow(ax, (0.685, 0.14), (0.55, 0.095), rad=0.0)

    # Small explanatory note
    ax.text(0.50, 0.505,
            "Audit modules are evaluated under identical leakage-safe splits to separate interpolation-driven accuracy from extrapolative behavior.",
            ha="center", va="center", fontsize=8.6, color="#4B5563")

    fig.savefig(outdir / "Figure1_leakage_aware_audit_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Figure1_leakage_aware_audit_workflow.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
