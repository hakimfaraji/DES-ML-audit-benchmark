#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a cleaned SI Figure S7 SHAP summary.

This script accepts the SHAP global summary produced by script 07 in the v6 pipeline:
  shap_global_summary.csv

Recommended workflow
--------------------
1) First generate SHAP outputs, if you do not already have them:
   unzip -q DES_ML_Audit_GitHub_Ready_v6_finalclean.zip
   python DES_ML_Audit_GitHub_Ready/scripts/07_shap_interpretability.py \
     --dataset Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
     --outdir shap_outputs_clean \
     --protocols pair_ratio_group \
     --models ExtraTrees \
     --feature-set descriptors_ratio_temp \
     --max-folds 2 \
     --max-background 80 \
     --max-explain 120

2) Then build the cleaned SI figure:
   python build_clean_shap_summary.py \
     --shap-summary shap_outputs_clean/shap_global_summary.csv \
     --outdir si_cleanup_outputs

Outputs
-------
  FigureS7_SHAP_clean_summary.png/pdf
  Table_S3_SHAP_top_features_clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_LABELS = {
    "density_g_cm3": "Density",
    "viscosity_mpa_s": "Viscosity",
    "conductivity_ms_cm": "Conductivity",
    "surface_tension_mn_m": "Surface tension",
    "refractive_index": "Refractive index",
}
PROPERTY_ORDER = ["Density", "Refractive index", "Surface tension", "Conductivity", "Viscosity"]


def pretty_feature(name: str) -> str:
    s = str(name)
    s = s.replace("measurement_temperature_c", "Temperature")
    s = s.replace("temperature_k", "Temperature")
    s = s.replace("molar_ratio_numeric", "Molar ratio")
    s = s.replace("hba_descriptor_", "HBA ")
    s = s.replace("hbd_descriptor_", "HBD ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        "mol wt": "molecular weight",
        "exact mw": "exact mass",
        "tpsa": "TPSA",
        "logp": "logP",
        "qed": "QED",
        "hbond donors": "H-bond donors",
        "hbond acceptors": "H-bond acceptors",
        "fraction csp3": "fraction Csp3",
    }
    low = s.lower()
    for k, v in replacements.items():
        low = low.replace(k, v)
    # Restore HBA/HBD capitalization after lower-case replacement.
    low = low.replace("hba ", "HBA ").replace("hbd ", "HBD ")
    return low[:1].upper() + low[1:]


def normalize_property(x: str) -> str:
    x = str(x)
    return TARGET_LABELS.get(x, x.replace("_", " ").title())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap-summary", required=True, help="shap_global_summary.csv from script 07")
    ap.add_argument("--outdir", default="si_cleanup_outputs")
    ap.add_argument("--protocol", default="pair_ratio_group", help="Protocol to show in cleaned figure")
    ap.add_argument("--model", default="ExtraTrees")
    ap.add_argument("--feature-set", default="descriptors_ratio_temp")
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.shap_summary)
    required = {"target", "protocol", "model", "feature_set", "feature", "mean_abs_shap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SHAP summary file missing columns: {sorted(missing)}")

    sub = df[(df["protocol"] == args.protocol) & (df["model"] == args.model) & (df["feature_set"] == args.feature_set)].copy()
    if sub.empty:
        available = df[["protocol", "model", "feature_set"]].drop_duplicates().to_string(index=False)
        raise ValueError("No rows for requested protocol/model/feature-set. Available combinations:\n" + available)

    sub["property_label"] = sub["target"].map(normalize_property)
    sub["feature_pretty"] = sub["feature"].map(pretty_feature)

    properties = [p for p in PROPERTY_ORDER if p in set(sub["property_label"])]
    if not properties:
        properties = list(sub["property_label"].dropna().unique())

    ncols = 2
    nrows = int(np.ceil(len(properties) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4.2, 3.8 * nrows)))
    axes = np.atleast_1d(axes).ravel()

    top_table_rows = []
    for ax, prop in zip(axes, properties):
        g = sub[sub["property_label"] == prop].sort_values("mean_abs_shap", ascending=False).head(args.top_n)
        for rank, (_, row) in enumerate(g.iterrows(), start=1):
            top_table_rows.append({
                "Property": prop,
                "Rank": rank,
                "Feature": row["feature"],
                "Feature label": row["feature_pretty"],
                "Mean |SHAP|": row["mean_abs_shap"],
            })
        plot = g.iloc[::-1]
        ax.barh(plot["feature_pretty"], plot["mean_abs_shap"])
        ax.set_title(prop)
        ax.set_xlabel("Mean |SHAP value|")
        ax.grid(axis="x", alpha=0.25)

    for ax in axes[len(properties):]:
        ax.axis("off")

    fig.suptitle("Figure S7. Clean SHAP feature-importance summary under leakage-corrected pair+ratio validation", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "FigureS7_SHAP_clean_summary.png", dpi=300)
    fig.savefig(outdir / "FigureS7_SHAP_clean_summary.pdf")
    plt.close(fig)

    pd.DataFrame(top_table_rows).to_csv(outdir / "Table_S3_SHAP_top_features_clean.csv", index=False)

    print("[DONE] Clean SHAP summary generated.")
    print(f"[OUT] {outdir / 'FigureS7_SHAP_clean_summary.png'}")
    print(f"[OUT] {outdir / 'Table_S3_SHAP_top_features_clean.csv'}")


if __name__ == "__main__":
    main()
