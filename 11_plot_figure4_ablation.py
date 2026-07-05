#!/usr/bin/env python3
"""
Figure 4 builder for feature ablation.
- No pandas
- No seaborn
- No model training
- Reads ablation_metrics_summary.csv using Python csv only
- Should run in seconds on Colab
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROPERTY_ORDER = ["density", "refractive_index", "surface_tension", "conductivity", "viscosity"]
PROPERTY_LABEL = {
    "density": "Density",
    "refractive_index": "Refractive index",
    "surface_tension": "Surface tension",
    "conductivity": "Conductivity",
    "viscosity": "Viscosity",
}
FEATURE_ORDER = [
    "descriptors_only",
    "descriptors_plus_ratio",
    "descriptors_plus_ratio_temp",
    "descriptors_plus_ratio_temp_interactions",
    "full_leakage_safe",
]
FEATURE_LABEL = {
    "descriptors_only": "Descriptors",
    "descriptors_plus_ratio": "+Ratio",
    "descriptors_plus_ratio_temp": "+Temp",
    "descriptors_plus_ratio_temp_interactions": "+Interactions",
    "full_leakage_safe": "Full",
}


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_rows(path: Path, protocol: str, model: str, viscosity_variant: str):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            prop = r.get("property", "")
            if r.get("protocol") != protocol:
                continue
            if r.get("model") != model:
                continue
            if r.get("feature_set") not in FEATURE_ORDER:
                continue
            # Use log-viscosity by default because diagnostics showed raw viscosity is skewed.
            if prop == "viscosity" and r.get("target_variant") != viscosity_variant:
                continue
            if prop != "viscosity" and r.get("target_variant", "raw") not in ("raw", ""):
                continue
            r2 = safe_float(r.get("r2"))
            mae = safe_float(r.get("mae"))
            if r2 is None:
                continue
            rows.append({
                "property": prop,
                "property_label": PROPERTY_LABEL.get(prop, prop),
                "feature_set": r.get("feature_set"),
                "feature_label": FEATURE_LABEL.get(r.get("feature_set"), r.get("feature_set")),
                "r2": r2,
                "mae": mae,
                "target_variant": r.get("target_variant", ""),
                "protocol": r.get("protocol", ""),
                "model": r.get("model", ""),
            })
    rows.sort(key=lambda r: (PROPERTY_ORDER.index(r["property"]) if r["property"] in PROPERTY_ORDER else 999,
                             FEATURE_ORDER.index(r["feature_set"])))
    return rows


def write_csv(rows, path: Path):
    fields = ["property", "property_label", "target_variant", "protocol", "model", "feature_set", "feature_label", "r2", "mae"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_lines(rows, outdir: Path):
    by_prop = defaultdict(list)
    for r in rows:
        by_prop[r["property"]].append(r)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = list(range(len(FEATURE_ORDER)))
    for prop in PROPERTY_ORDER:
        if prop not in by_prop:
            continue
        vals = {r["feature_set"]: r["r2"] for r in by_prop[prop]}
        y = [vals.get(fs, float("nan")) for fs in FEATURE_ORDER]
        ax.plot(x, y, marker="o", linewidth=2, label=PROPERTY_LABEL[prop])
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABEL[f] for f in FEATURE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Cross-validated R²")
    ax.set_xlabel("Feature set")
    ax.set_title("Feature ablation across DES property prediction tasks")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "Figure4_feature_ablation_r2.png", dpi=300)
    fig.savefig(outdir / "Figure4_feature_ablation_r2.pdf")
    plt.close(fig)


def compute_delta(rows):
    out = []
    by_prop = defaultdict(dict)
    for r in rows:
        by_prop[r["property"]][r["feature_set"]] = r["r2"]
    for prop in PROPERTY_ORDER:
        d = by_prop.get(prop, {})
        if "descriptors_only" in d and "full_leakage_safe" in d:
            out.append({
                "property": prop,
                "property_label": PROPERTY_LABEL[prop],
                "r2_descriptors_only": d["descriptors_only"],
                "r2_full": d["full_leakage_safe"],
                "delta_full_minus_descriptors": d["full_leakage_safe"] - d["descriptors_only"],
            })
    return out


def write_delta(delta, path: Path):
    fields = ["property", "property_label", "r2_descriptors_only", "r2_full", "delta_full_minus_descriptors"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in delta:
            w.writerow(r)


def plot_delta(delta, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = list(range(len(delta)))
    y = [r["delta_full_minus_descriptors"] for r in delta]
    labels = [r["property_label"] for r in delta]
    ax.bar(x, y)
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ΔR² (Full − descriptors only)")
    ax.set_title("Added value of full feature representation")
    fig.tight_layout()
    fig.savefig(outdir / "Figure4_delta_full_vs_descriptors.png", dpi=300)
    fig.savefig(outdir / "Figure4_delta_full_vs_descriptors.pdf")
    plt.close(fig)


def write_caption(outdir: Path, protocol: str, model: str, viscosity_variant: str):
    text = (
        "Figure 4. Feature ablation analysis for DES property prediction. "
        f"Cross-validated R² values are shown for the {model} model under the {protocol} validation protocol. "
        "Feature sets increase in complexity from component descriptors only to ratio-aware, temperature-aware, "
        "interaction-augmented, and full leakage-safe representations. Viscosity is shown using the "
        f"{viscosity_variant} target variant. The analysis quantifies whether chemistry-informed descriptors "
        "provide predictive value beyond basic composition and temperature information."
    )
    (outdir / "Figure4_caption.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-summary", required=True)
    parser.add_argument("--protocol", default="pair_ratio_group")
    parser.add_argument("--model", default="extra_trees")
    parser.add_argument("--viscosity-variant", default="log10", choices=["raw", "log10"])
    parser.add_argument("--outdir", default="figure4_outputs_emergency")
    args = parser.parse_args()

    inpath = Path(args.ablation_summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not inpath.exists():
        raise SystemExit(f"ERROR: missing input file: {inpath}")

    rows = read_rows(inpath, args.protocol, args.model, args.viscosity_variant)
    if not rows:
        raise SystemExit("ERROR: no rows matched. Check --protocol, --model, and --viscosity-variant.")

    write_csv(rows, outdir / "Figure4_feature_ablation_data.csv")
    plot_lines(rows, outdir)
    delta = compute_delta(rows)
    write_delta(delta, outdir / "Figure4_delta_vs_descriptors.csv")
    plot_delta(delta, outdir)
    write_caption(outdir, args.protocol, args.model, args.viscosity_variant)

    print(f"Done. Outputs written to: {outdir}")
    print(f"Rows used: {len(rows)}")
    print("Created:")
    for name in [
        "Figure4_feature_ablation_r2.png/pdf",
        "Figure4_delta_full_vs_descriptors.png/pdf",
        "Figure4_feature_ablation_data.csv",
        "Figure4_delta_vs_descriptors.csv",
        "Figure4_caption.txt",
    ]:
        print(" -", name)

if __name__ == "__main__":
    main()
