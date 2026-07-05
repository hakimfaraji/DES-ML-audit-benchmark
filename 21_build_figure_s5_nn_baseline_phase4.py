#!/usr/bin/env python3
"""
Build Figure S5 (Phase 4): nearest-neighbor baseline under extrapolative validation.
Robust to the actual knn_baseline_summary.csv format used in the DES project.

Input:
  knn_baseline_summary.csv
Required columns:
  property, protocol, baseline, k, r2_mean, r2_sd

Output:
  FigureS5_NN_baseline_phase4.png/pdf
  FigureS5_NN_baseline_phase4_plot_data.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def property_order_key(p):
    order = {"Density": 0, "Refractive index": 1, "Surface tension": 2, "Conductivity": 3, "Viscosity": 4}
    return order.get(str(p), 99)


def make_method(row):
    b = str(row.get("baseline", ""))
    k = row.get("k", np.nan)
    if b.lower().startswith("dummy"):
        return "Dummy mean"
    if b.upper() == "KNN" or b.lower().startswith("knn"):
        try:
            kval = int(k)
        except Exception:
            kval = None
        if kval == 1:
            return "1-NN"
        if kval == 5:
            return "5-NN"
        return f"{kval}-NN" if kval else "k-NN"
    return b


def add_panel_break_marks(ax_top, ax_bottom):
    d = 0.008
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="knn_baseline_summary.csv")
    ap.add_argument("--outdir", default="phase4_figures")
    ap.add_argument("--protocol", default="D_leave_HBD", help="Protocol to plot, e.g., D_leave_HBD or D_leave_HBA")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = ["property", "protocol", "baseline", "k", "r2_mean", "r2_sd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {missing}. Available: {list(df.columns)}")

    sub = df[df["protocol"].astype(str) == args.protocol].copy()
    if sub.empty:
        available = sorted(df["protocol"].dropna().astype(str).unique())
        raise ValueError(f"No rows found for protocol={args.protocol}. Available protocols: {available}")

    sub["method"] = sub.apply(make_method, axis=1)
    keep_methods = ["Dummy mean", "1-NN", "5-NN"]
    sub = sub[sub["method"].isin(keep_methods)].copy()
    if sub.empty:
        raise ValueError("No Dummy mean / 1-NN / 5-NN rows found after method mapping.")

    # If duplicate rows exist, keep mean over duplicates.
    sub = sub.groupby(["property", "method"], as_index=False).agg(r2_mean=("r2_mean", "mean"), r2_sd=("r2_sd", "mean"))
    sub["property_order"] = sub["property"].map(property_order_key)
    sub["method_order"] = sub["method"].map({m: i for i, m in enumerate(keep_methods)})
    sub = sub.sort_values(["property_order", "method_order"])
    sub.to_csv(outdir / "FigureS5_NN_baseline_phase4_plot_data.csv", index=False)

    props = sorted(sub["property"].unique(), key=property_order_key)
    x = np.arange(len(props))
    width = 0.24
    offsets = {"Dummy mean": -width, "1-NN": 0.0, "5-NN": width}

    # Broken axis ranges. Top emphasizes moderate failures; bottom shows severe negative means.
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08})

    for method in keep_methods:
        ys, es = [], []
        for prop in props:
            row = sub[(sub["property"] == prop) & (sub["method"] == method)]
            if row.empty:
                ys.append(np.nan); es.append(0)
            else:
                ys.append(float(row["r2_mean"].iloc[0])); es.append(float(row["r2_sd"].iloc[0]))
        pos = x + offsets[method]
        for ax in (ax_top, ax_bot):
            ax.bar(pos, ys, width=width, label=method if ax is ax_top else None, alpha=0.9)
            # Draw SD error bars but let axes clip extreme values for readability.
            ax.errorbar(pos, ys, yerr=es, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0, clip_on=True)

    ax_top.axhline(0, linestyle="--", linewidth=1.1)
    ax_bot.axhline(0, linestyle="--", linewidth=1.1)
    ax_top.grid(True, axis="y", alpha=0.25)
    ax_bot.grid(True, axis="y", alpha=0.25)

    # Determine bottom lower limit based on means, not SD, to avoid unreadable axis from huge SD.
    min_mean = np.nanmin(sub["r2_mean"].values)
    bottom_low = min(-16000, np.floor(min_mean / 1000) * 1000 - 1000) if min_mean < -1000 else -2000
    ax_top.set_ylim(-25, 1.0)
    ax_bot.set_ylim(bottom_low, -150)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()
    add_panel_break_marks(ax_top, ax_bot)

    ax_top.set_title("Nearest-neighbor baseline comparison under extrapolative validation", fontsize=15)
    fig.text(0.025, 0.5, "R² (mean ± SD)", va="center", rotation="vertical", fontsize=12)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(props, rotation=22, ha="right")
    ax_top.legend(loc="lower left", frameon=True)
    ax_bot.text(0.01, 0.08, "Broken y-axis preserves severe negative R² values.", transform=ax_bot.transAxes, fontsize=9)
    ax_top.annotate("Failure under extrapolation", xy=(x[3], -20), xytext=(x[3]-0.75, -9),
                    arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10, ha="center")

    fig.tight_layout(rect=[0.04, 0.02, 1, 0.98])
    fig.savefig(outdir / "FigureS5_NN_baseline_phase4.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "FigureS5_NN_baseline_phase4.pdf", bbox_inches="tight")
    print(f"[DONE] Wrote {outdir / 'FigureS5_NN_baseline_phase4.png'}")
    print(f"[DONE] Wrote {outdir / 'FigureS5_NN_baseline_phase4.pdf'}")


if __name__ == "__main__":
    main()
