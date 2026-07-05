#!/usr/bin/env python3
"""
Integrated upgrade runner for the DES ML audit benchmark workflow (v7-upgrade).

This runner combines the original v6 reproducible pipeline with the manuscript-upgrade
analyses added after reviewer-style assessment:
  - Dataset imbalance analysis / Figure S1 / Table S4
  - Applicability-domain distance analysis / Figure S4 / Table S6
  - Nearest-neighbor baseline / Figure S5 / Table S7
  - Leakage-audit table / Table S5
  - Updated workflow Figure 1 and Phase-4 Figure 2
  - Optional viscosity diagnostics Figure S6
  - Optional cleaned SHAP Figure S7 if SHAP summary is available

The script is intentionally modular: each step can be run independently by calling the
corresponding script in scripts/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"
DEFAULT_DATASET = REPO / "data" / "Unified_DES_dataset_GOLD_descriptor_ready_subset.csv"


def run(cmd: list[str], label: str, cwd: Path = REPO) -> None:
    print("\n" + "=" * 88)
    print(f"[RUN] {label}")
    print(" ".join(str(x) for x in cmd))
    print("=" * 88)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_dataset(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    return dataset


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run integrated DES ML audit upgrade pipeline")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to GOLD descriptor-ready CSV")
    ap.add_argument("--outdir", default="v7_upgrade_outputs", help="Output directory")
    ap.add_argument("--skip-original", action="store_true", help="Skip original v6 core pipeline")
    ap.add_argument("--fast-original", action="store_true", help="Run original v6 pipeline in --fast mode")
    ap.add_argument("--skip-shap-clean", action="store_true", help="Skip Figure S7 clean SHAP generation")
    ap.add_argument("--shap-summary", default="", help="Optional path to shap_global_summary.csv")
    args = ap.parse_args()

    py = sys.executable
    dataset = ensure_dataset(Path(args.dataset))
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 0) Original v6 workflow remains available for reproducing main tables/Figures 2-4.
    if not args.skip_original:
        cmd = [py, str(REPO / "run_pipeline.py"), "--dataset", str(dataset)]
        if args.fast_original:
            cmd.append("--fast")
        cmd.append("--include-si")
        run(cmd, "Original v6 leakage-aware audit workflow")

    # 1) Upgrade analyses.
    imbalance_dir = out / "dataset_imbalance_outputs"
    ad_dir = out / "applicability_domain_outputs"
    knn_dir = out / "knn_baseline_outputs"
    tables_dir = out / "si_tables"
    main_fig_dir = out / "main_figures"
    si_fig_dir = out / "si_figures"

    run([py, str(SCRIPTS / "13_dataset_imbalance_analysis.py"), "--input", str(dataset), "--outdir", str(imbalance_dir)],
        "Dataset imbalance analysis")

    run([py, str(SCRIPTS / "14_applicability_domain_analysis.py"), "--input", str(dataset), "--outdir", str(ad_dir)],
        "Applicability-domain distance analysis")

    run([py, str(SCRIPTS / "15_nearest_neighbor_baseline.py"), "--input", str(dataset), "--outdir", str(knn_dir)],
        "Nearest-neighbor baseline")

    run([py, str(SCRIPTS / "16_leakage_audit_table.py"), "--input", str(dataset), "--outdir", str(tables_dir)],
        "Leakage-audit table")

    # 2) Final figure assets.
    run([py, str(SCRIPTS / "17_build_main_figure1_audit_workflow.py"), "--outdir", str(main_fig_dir)],
        "Main Figure 1 workflow")

    run([py, str(SCRIPTS / "18_build_figure_s1_dataset_imbalance.py"), "--input", str(dataset), "--outdir", str(si_fig_dir)],
        "Figure S1 dataset imbalance")

    run([py, str(SCRIPTS / "19_build_figure2_validation_sensitivity_phase4.py"), "--outdir", str(main_fig_dir)],
        "Main Figure 2 with interpolation/extrapolation labels")

    run([py, str(SCRIPTS / "20_build_figure_s4_applicability_domain_phase4.py"),
         "--quartiles", str(ad_dir / "distance_error_quartiles.csv"), "--outdir", str(si_fig_dir)],
        "Figure S4 applicability-domain phase-4 plot")

    # Two Figure S5 variants are included. The phase4 variant adds annotation; the publication
    # variant uses a broken y-axis. The manuscript-ready choice can be selected later.
    run([py, str(SCRIPTS / "21b_build_figure_s5_nn_baseline_publication.py"),
         "--input", str(knn_dir / "knn_baseline_summary.csv"), "--outdir", str(si_fig_dir), "--protocol", "D_leave_HBD"],
        "Figure S5 nearest-neighbor baseline publication plot")

    run([py, str(SCRIPTS / "22_build_figure_s6_viscosity_diagnostics.py"), "--input", str(dataset), "--outdir", str(si_fig_dir)],
        "Figure S6 viscosity diagnostics")

    # 3) Copy table source CSVs into a single table directory.
    copy_if_exists(imbalance_dir / "imbalance_summary.csv", tables_dir / "Table_S4_dataset_imbalance_summary.csv")
    copy_if_exists(ad_dir / "distance_error_quartiles.csv", tables_dir / "Table_S6_applicability_domain_quartiles.csv")
    copy_if_exists(knn_dir / "knn_baseline_summary.csv", tables_dir / "Table_S7_nearest_neighbor_baseline_summary.csv")

    # 4) Optional cleaned SHAP summary if user supplies shap_global_summary.csv.
    shap_summary = Path(args.shap_summary).expanduser().resolve() if args.shap_summary else Path("")
    if not args.skip_shap_clean and args.shap_summary and shap_summary.exists():
        run([py, str(SCRIPTS / "23_build_clean_shap_summary.py"), "--shap-summary", str(shap_summary), "--outdir", str(si_fig_dir)],
            "Figure S7 cleaned SHAP summary")
    else:
        print("\n[SKIP] Clean SHAP Figure S7. Provide --shap-summary path to generate it.")

    print("\n[DONE] Integrated v7-upgrade workflow complete.")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
