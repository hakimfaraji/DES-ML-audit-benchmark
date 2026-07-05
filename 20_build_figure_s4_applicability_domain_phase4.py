#!/usr/bin/env python3
"""
Build Figure S4 (Phase 4): applicability-domain error vs distance.

Input:
  distance_error_quartiles.csv
Required columns supported:
  property_label, protocol_label, distance_quartile, mean_knn_distance, mean_absolute_error

Output:
  FigureS4_applicability_domain_phase4.png/pdf
  FigureS4_applicability_domain_phase4_plot_data.csv
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quartile_order(x: str) -> int:
    m = re.search(r"Q(\d)", str(x))
    return int(m.group(1)) if m else 99


def protocol_sort_key(label: str) -> int:
    s = str(label)
    if s.startswith("B"):
        return 0
    if s.startswith("C"):
        return 1
    if "HBA" in s:
        return 2
    if "HBD" in s:
        return 3
    return 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quartiles", required=True, help="distance_error_quartiles.csv")
    ap.add_argument("--outdir", default="phase4_figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.quartiles)
    required = ["property_label", "protocol_label", "distance_quartile", "mean_knn_distance", "mean_absolute_error"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.quartiles}: {missing}. Available: {list(df.columns)}")

    df = df.copy()
    df["q_order"] = df["distance_quartile"].map(quartile_order)
    df = df.sort_values(["protocol_label", "property_label", "q_order"])
    df.to_csv(outdir / "FigureS4_applicability_domain_phase4_plot_data.csv", index=False)

    protocols = sorted(df["protocol_label"].dropna().unique(), key=protocol_sort_key)
    n_panels = len(protocols)
    if n_panels == 0:
        raise ValueError("No protocol labels found after reading input file.")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=False, sharey=False)
    axes = axes.ravel()

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    properties = list(df["property_label"].dropna().unique())

    for ax, prot in zip(axes, protocols):
        sub = df[df["protocol_label"] == prot]
        plotted = False
        for i, prop in enumerate(properties):
            sp = sub[sub["property_label"] == prop].sort_values("q_order")
            if sp.empty:
                continue
            x = sp["mean_knn_distance"].astype(float).to_numpy()
            y = sp["mean_absolute_error"].astype(float).to_numpy()
            if len(x) == 0:
                continue
            ax.plot(x, y, marker=marker_cycle[i % len(marker_cycle)], linewidth=1.8, markersize=5.5, label=prop)
            # Add a simple regression/trend line when at least 3 points are available.
            if len(x) >= 3 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.nanstd(x) > 0:
                try:
                    coef = np.polyfit(x, y, deg=1)
                    xx = np.linspace(np.nanmin(x), np.nanmax(x), 60)
                    yy = np.polyval(coef, xx)
                    ax.plot(xx, yy, linestyle="--", linewidth=1.1, alpha=0.65)
                except Exception:
                    pass
            plotted = True
        ax.set_title(prot, fontsize=12)
        ax.set_xlabel("Mean kNN distance to training set")
        ax.set_ylabel("Mean absolute error")
        ax.grid(True, axis="both", alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Hide unused panels if any.
    for ax in axes[n_panels:]:
        ax.axis("off")

    # Single legend.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Applicability-domain analysis: error increases with distance to the training domain", fontsize=15, y=1.025)
    fig.text(0.5, 0.01, "Dashed lines show simple linear trend fits within each protocol/property.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])

    fig.savefig(outdir / "FigureS4_applicability_domain_phase4.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "FigureS4_applicability_domain_phase4.pdf", bbox_inches="tight")
    print(f"[DONE] Wrote {outdir / 'FigureS4_applicability_domain_phase4.png'}")
    print(f"[DONE] Wrote {outdir / 'FigureS4_applicability_domain_phase4.pdf'}")


if __name__ == "__main__":
    main()
