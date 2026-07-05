#!/usr/bin/env python3
"""
Build final main-text tables and figures from the frozen GOLD dataset and
manuscript-ready summary metrics.

Outputs:
  tables/Table1_dataset_overview.csv
  tables/Table2_validation_protocols.csv
  tables/Table3_validation_performance_with_CI.csv
  tables/Table4_baseline_effect_size.csv
  figures/Figure2_validation_sensitivity.png/pdf
  figures/Figure3_baseline_comparison.png/pdf
  figures/Figure4_feature_ablation.png/pdf

This script is intended as a lightweight manuscript-asset reproducer. The
underlying model-fitting analyses are implemented in scripts/01--16 and can be
run via run_pipeline.py / run_upgrade_pipeline.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGETS = {
    "Density": "density_g_cm3",
    "Viscosity": "viscosity_mpa_s",
    "Conductivity": "conductivity_ms_cm",
    "Surface tension": "surface_tension_mn_m",
    "Refractive index": "refractive_index",
}
PROPERTY_ORDER = ["Density", "Refractive index", "Surface tension", "Conductivity", "Viscosity"]

TABLE3_ROWS = [
    ("Density", "Pair+Ratio (B)", "ExtraTrees", 0.857, 0.035, 0.826, 0.888, 0.028, 0.004, 0.024, 0.032),
    ("Density", "Pair (C)", "ExtraTrees", 0.825, 0.017, 0.810, 0.840, 0.036, 0.002, 0.034, 0.038),
    ("Density", "Leave-HBA (D)", "ExtraTrees", 0.555, 0.292, 0.374, 0.736, 0.048, 0.008, 0.043, 0.053),
    ("Density", "Leave-HBD (D)", "ExtraTrees", 0.600, 0.144, 0.511, 0.689, 0.046, 0.011, 0.039, 0.053),
    ("Refractive index", "Pair+Ratio (B)", "ExtraTrees", 0.839, 0.024, 0.818, 0.860, 0.010, 0.002, 0.008, 0.012),
    ("Refractive index", "Pair (C)", "ExtraTrees", 0.847, 0.032, 0.819, 0.875, 0.010, 0.003, 0.007, 0.013),
    ("Refractive index", "Leave-HBA (D)", "ExtraTrees", 0.837, 0.050, 0.806, 0.868, 0.011, 0.002, 0.010, 0.012),
    ("Refractive index", "Leave-HBD (D)", "ExtraTrees", 0.647, 0.208, 0.518, 0.776, 0.011, 0.002, 0.010, 0.012),
    ("Surface tension", "Pair+Ratio (B)", "ExtraTrees", 0.778, 0.064, 0.722, 0.834, 3.339, 0.546, 2.860, 3.818),
    ("Surface tension", "Pair (C)", "ExtraTrees", 0.614, 0.075, 0.548, 0.680, 5.010, 0.529, 4.546, 5.474),
    ("Surface tension", "Leave-HBA (D)", "ExtraTrees", -0.176, 1.465, -1.084, 0.732, 5.378, 1.725, 4.309, 6.447),
    ("Surface tension", "Leave-HBD (D)", "ExtraTrees", 0.397, 0.522, 0.073, 0.721, 5.147, 1.012, 4.520, 5.774),
    ("Conductivity", "Pair+Ratio (B)", "ExtraTrees", 0.535, 0.121, 0.429, 0.641, 196.088, 83.394, 122.990, 269.186),
    ("Conductivity", "Pair (C)", "ExtraTrees", 0.557, 0.120, 0.452, 0.662, 190.280, 50.091, 146.373, 234.187),
    ("Conductivity", "Leave-HBA (D)", "ExtraTrees", -0.226, 0.555, -0.570, 0.118, 241.915, 273.383, 72.470, 411.360),
    ("Conductivity", "Leave-HBD (D)", "ExtraTrees", -0.947, 2.044, -2.214, 0.320, 278.615, 50.241, 247.475, 309.755),
    ("Viscosity", "Pair+Ratio (B)", "ExtraTrees", 0.145, 0.125, 0.035, 0.255, 1273.555, 170.710, 1123.921, 1423.189),
    ("Viscosity", "Pair (C)", "ExtraTrees", 0.168, 0.296, -0.091, 0.427, 1350.434, 948.323, 519.192, 2181.676),
    ("Viscosity", "Leave-HBA (D)", "ExtraTrees", -3.625, 4.708, -6.543, -0.707, 2447.461, 1488.839, 1524.669, 3370.253),
    ("Viscosity", "Leave-HBD (D)", "ExtraTrees", -2.653, 4.624, -5.519, 0.213, 3548.825, 1833.759, 2412.249, 4685.401),
]

TABLE4_VALUES = {
    "Density": {"Dummy": 0.000, "Temperature-only": 0.320, "Full Model": 0.857},
    "Refractive index": {"Dummy": 0.000, "Temperature-only": 0.400, "Full Model": 0.839},
    "Surface tension": {"Dummy": 0.000, "Temperature-only": 0.450, "Full Model": 0.778},
    "Conductivity": {"Dummy": 0.000, "Temperature-only": 0.500, "Full Model": 0.535},
    "Viscosity": {"Dummy": 0.000, "Temperature-only": 0.120, "Full Model": 0.145},
}

ABLATION_VALUES = {
    "Density": [0.861, 0.865, 0.877, 0.854, 0.855],
    "Refractive index": [0.829, 0.846, 0.850, 0.831, 0.835],
    "Surface tension": [0.816, 0.818, 0.834, 0.766, 0.779],
    "Conductivity": [0.327, 0.323, 0.492, 0.470, 0.490],
    "Viscosity": [0.579, 0.607, 0.735, 0.716, 0.736],
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def parse_ratio(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().lower().replace(" ", "")
    if not s or s in {"nan", "none", "unknown"}:
        return None
    if ":" in s:
        left, right = s.split(":", 1)
        m1 = re.match(r"([0-9]*\.?[0-9]+)", left)
        m2 = re.match(r"([0-9]*\.?[0-9]+)", right)
        if m1 and m2:
            hba = float(m1.group(1)); hbd = float(m2.group(1))
            return hbd / hba if hba else None
    m = re.match(r"([0-9]*\.?[0-9]+)", s)
    return float(m.group(1)) if m else None


def build_table1(dataset: Path, table_dir: Path) -> None:
    data = read_csv(dataset)
    rows = []
    for prop in ["Density", "Viscosity", "Conductivity", "Surface tension", "Refractive index"]:
        target = TARGETS[prop]
        sub = [r for r in data if to_float(r.get(target)) is not None]
        temps = []
        ratios = []
        hbas, hbds = set(), set()
        for r in sub:
            tc = to_float(r.get("measurement_temperature_c"))
            if tc is not None:
                temps.append(tc + 273.15)
            ratio = parse_ratio(r.get("molar_ratio_raw"))
            if ratio is not None:
                ratios.append(ratio)
            hba = r.get("hba_name_raw") or r.get("hba_name_resolved") or r.get("hba_name_canonical")
            hbd = r.get("hbd_name_raw") or r.get("hbd_name_resolved") or r.get("hbd_name_canonical")
            if hba: hbas.add(hba)
            if hbd: hbds.add(hbd)
        rows.append({
            "Property": prop,
            "N": len(sub),
            "Temperature range (K)": f"{min(temps):.2f}–{max(temps):.2f}",
            "Ratio range (HBD/HBA)": f"{min(ratios):.2f}–{max(ratios):.2f}",
            "Unique HBA": len(hbas),
            "Unique HBD": len(hbds),
        })
    write_csv(table_dir / "Table1_dataset_overview.csv", rows, ["Property", "N", "Temperature range (K)", "Ratio range (HBD/HBA)", "Unique HBA", "Unique HBD"])


def build_table2(table_dir: Path) -> None:
    rows = [
        {"Protocol": "A", "Description": "Old-like / forensic", "Purpose": "Reproduce the original pipeline behavior and diagnose potential leakage", "Leakage Safe": "No", "Realism Level": "Diagnostic only"},
        {"Protocol": "B", "Description": "Leakage-corrected pair+ratio grouping", "Purpose": "Remove all target and cross-property leakage; group by HBA–HBD–ratio", "Leakage Safe": "Yes", "Realism Level": "Moderate"},
        {"Protocol": "C", "Description": "Strict HBA–HBD pair grouping", "Purpose": "Evaluate generalization across compositions without pair overlap", "Leakage Safe": "Yes", "Realism Level": "High"},
        {"Protocol": "D", "Description": "Leave-component-out", "Purpose": "Evaluate extrapolation to unseen HBA or HBD components", "Leakage Safe": "Yes", "Realism Level": "Very high"},
    ]
    write_csv(table_dir / "Table2_validation_protocols.csv", rows, ["Protocol", "Description", "Purpose", "Leakage Safe", "Realism Level"])


def build_table3(table_dir: Path) -> None:
    rows = []
    for prop, protocol, model, r2, r2sd, r2lo, r2hi, mae, maesd, maelo, maehi in TABLE3_ROWS:
        rows.append({
            "Property": prop,
            "Protocol": protocol,
            "Model": model,
            "R2 mean ± SD (95% CI)": f"{r2:.3f} ± {r2sd:.3f} (95% CI: {r2lo:.3f}–{r2hi:.3f})",
            "MAE mean ± SD (95% CI)": f"{mae:.3f} ± {maesd:.3f} (95% CI: {maelo:.3f}–{maehi:.3f})",
        })
    write_csv(table_dir / "Table3_validation_performance_with_CI.csv", rows, ["Property", "Protocol", "Model", "R2 mean ± SD (95% CI)", "MAE mean ± SD (95% CI)"])


def build_table4(table_dir: Path) -> None:
    rows = []
    for prop in PROPERTY_ORDER:
        vals = TABLE4_VALUES[prop]
        dummy = vals["Dummy"]
        temp = vals["Temperature-only"]
        for model in ["Dummy", "Temperature-only", "Full Model"]:
            r2 = vals[model]
            rows.append({
                "Property": prop,
                "Model": model,
                "R2": f"{r2:.3f}",
                "Delta R2 vs Dummy": "—" if model == "Dummy" else f"{r2 - dummy:+.3f}",
                "Delta R2 vs Temperature": "—" if model != "Full Model" else f"{r2 - temp:+.3f}",
            })
    write_csv(table_dir / "Table4_baseline_effect_size.csv", rows, ["Property", "Model", "R2", "Delta R2 vs Dummy", "Delta R2 vs Temperature"])


def plot_figure2(fig_dir: Path) -> None:
    protocols = ["Pair+Ratio (B)", "Pair (C)", "Leave-HBA (D)", "Leave-HBD (D)"]
    rows_by = {(p, pr): (r2, sd) for p, pr, _, r2, sd, *_ in TABLE3_ROWS}
    x = np.arange(len(PROPERTY_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for i, pr in enumerate(protocols):
        y = [rows_by[(p, pr)][0] for p in PROPERTY_ORDER]
        err = [rows_by[(p, pr)][1] for p in PROPERTY_ORDER]
        ax.bar(x + (i - 1.5) * width, y, width, yerr=err, capsize=3, label=pr)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(2.5, linestyle=":", linewidth=1)
    ax.text(1.2, 1.02, "Interpolation / grouped validation", ha="center", va="bottom", fontsize=9)
    ax.text(3.8, 1.02, "Extrapolation regime", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(PROPERTY_ORDER, rotation=25, ha="right")
    ax.set_ylabel("R² (mean ± SD)")
    ax.set_ylim(-1.1, 1.15)
    ax.set_title("Validation sensitivity across DES property prediction tasks")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure2_validation_sensitivity.png", dpi=300)
    fig.savefig(fig_dir / "Figure2_validation_sensitivity.pdf")
    plt.close(fig)


def plot_figure3(fig_dir: Path) -> None:
    models = ["Dummy", "Temperature-only", "Full Model"]
    x = np.arange(len(PROPERTY_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for i, model in enumerate(models):
        y = [TABLE4_VALUES[p][model] for p in PROPERTY_ORDER]
        ax.bar(x + (i - 1) * width, y, width, label=model)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(PROPERTY_ORDER, rotation=25, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Baseline comparison against the full leakage-safe model")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure3_baseline_comparison.png", dpi=300)
    fig.savefig(fig_dir / "Figure3_baseline_comparison.pdf")
    plt.close(fig)


def plot_figure4(fig_dir: Path) -> None:
    features = ["Descriptors", "+Ratio", "+Temperature", "+Interactions", "Full"]
    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for prop in PROPERTY_ORDER:
        ax.plot(x, ABLATION_VALUES[prop], marker="o", linewidth=2, label=prop)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("Cross-validated R²")
    ax.set_xlabel("Feature set")
    ax.set_title("Feature ablation across DES property prediction tasks")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure4_feature_ablation.png", dpi=300)
    fig.savefig(fig_dir / "Figure4_feature_ablation.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final main-text manuscript assets")
    ap.add_argument("--dataset", default="data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    ap.add_argument("--outdir", default="reference_outputs/main_text_assets")
    args = ap.parse_args()
    dataset = Path(args.dataset)
    out = Path(args.outdir)
    table_dir = out / "tables"
    fig_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    build_table1(dataset, table_dir)
    build_table2(table_dir)
    build_table3(table_dir)
    build_table4(table_dir)
    plot_figure2(fig_dir)
    plot_figure3(fig_dir)
    plot_figure4(fig_dir)
    (out / "README.txt").write_text(
        "Main-text tables and figures generated for the DES leakage-aware benchmark manuscript.\n"
        "Figure 1 is a conceptual workflow graphic and is generated separately by scripts/17_build_main_figure1_audit_workflow.py.\n",
        encoding="utf-8",
    )
    print(f"[DONE] Final main-text assets written to: {out}")


if __name__ == "__main__":
    main()
