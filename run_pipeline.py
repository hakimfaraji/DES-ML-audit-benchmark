#!/usr/bin/env python3
"""
Master runner for the DES ML audit workflow.

Default/--fast mode reproduces data-derived main-manuscript outputs:
  - Tables 1-4
  - Figures 2-4

Figure 1 is a conceptual workflow graphic and is maintained as a static
manuscript asset rather than regenerated from data. Supporting SI diagnostics
can be added with --include-si and/or --with-shap.
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
DEFAULT_DATASET_NAME = "Unified_DES_dataset_GOLD_descriptor_ready_subset.csv"


def run(cmd: list[str], label: str, cwd: Path = REPO) -> None:
    print("\n" + "=" * 80)
    print(f"[RUN] {label}")
    print(" ".join(str(x) for x in cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_dataset(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    root_copy = REPO / DEFAULT_DATASET_NAME
    if root_copy.exists():
        return root_copy
    try:
        os.symlink(dataset, root_copy)
    except Exception:
        shutil.copy2(dataset, root_copy)
    return root_copy


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DES ML audit workflow")
    ap.add_argument("--dataset", required=True, help="Path to Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    ap.add_argument("--fast", action="store_true", help="Reviewer smoke-test mode for main manuscript outputs")
    ap.add_argument("--include-si", action="store_true", help="Also generate optional SI diagnostics such as viscosity Figure S4")
    ap.add_argument("--with-shap", action="store_true", help="Run SHAP add-on analysis for SI")
    ap.add_argument("--skip-figures", action="store_true", help="Skip figure/table generation")
    args = ap.parse_args()

    dataset_root = ensure_dataset(Path(args.dataset))
    py = sys.executable
    n_splits = "3" if args.fast else "5"

    # Core analyses needed for main manuscript tables and Figures 2-4.
    cmd = [py, str(SCRIPTS / "01_diagnostic_audit.py"), "--input", str(dataset_root), "--outdir", "audit_outputs", "--n_splits", n_splits]
    if args.fast:
        cmd.append("--fast")
    run(cmd, "Diagnostic audit: Protocols A-C")

    run([py, str(SCRIPTS / "02_protocol_D_extrapolative_validation.py")], "Protocol D extrapolative validation")

    run([py, str(SCRIPTS / "03_baseline_analysis.py"), "--input_csv", str(dataset_root), "--out_dir", "baseline_outputs", "--n_splits", n_splits], "Trivial baseline analysis")

    ablation_cmd = [
        py, str(SCRIPTS / "04_feature_ablation.py"),
        "--input_csv", str(dataset_root),
        "--out_dir", "ablation_outputs",
        "--n_splits", n_splits,
    ]
    if args.fast:
        ablation_cmd += [
            "--protocols", "pair_ratio_group",
            "--models", "extra_trees",
            "--feature_sets",
            "descriptors_only",
            "descriptors_plus_ratio",
            "descriptors_plus_ratio_temp",
            "descriptors_plus_ratio_temp_interactions",
            "full_leakage_safe",
        ]
    run(ablation_cmd, "Feature ablation")

    # Optional SI analyses.
    if args.include_si:
        run([py, str(SCRIPTS / "05_viscosity_diagnostic.py"), "--input", str(dataset_root), "--outdir", "viscosity_outputs", "--n_splits", n_splits], "Viscosity diagnostic for SI")
    else:
        print("\n[SKIP] Viscosity diagnostic in main mode. Use --include-si for Figure S4.")

    if not args.fast and args.include_si:
        run([py, str(SCRIPTS / "06_permutation_interpretability.py"), "--data", str(dataset_root), "--outdir", "interpretability_outputs", "--n-splits", n_splits], "Permutation interpretability for SI")
    elif args.fast:
        print("\n[SKIP] Permutation interpretability in --fast mode. Use --include-si without --fast for SI outputs.")

    if args.with_shap:
        shap_cmd = [
            py, str(SCRIPTS / "07_shap_interpretability.py"),
            "--dataset", str(dataset_root),
            "--outdir", "shap_outputs_full",
            "--models", "ExtraTrees",
            "--protocols", "pair_group", "pair_ratio_group",
            "--feature-set", "descriptors_ratio_temp",
            "--n-splits", n_splits,
            "--max-folds", "2" if args.fast else "3",
            "--max-explain", "120" if args.fast else "200",
            "--max-background", "80" if args.fast else "120",
            "--make-plots",
        ]
        run(shap_cmd, "SHAP interpretability add-on for SI")

    if not args.skip_figures:
        # Main figures: Table 3 and Table 4 are built from Figure 2 data to guarantee consistency.
        run([
            py, str(SCRIPTS / "09_plot_figure2_validation.py"),
            "--audit-dir", "audit_outputs",
            "--protocol-d-dir", "protocol_D_extrapolative_outputs",
            "--outdir", "figure2_outputs",
            "--model", "extra_trees",
            "--feature-set", "full",
            "--d-feature-set", "full",
        ], "Figure 2 validation degradation")

        run([
            py, str(SCRIPTS / "10_plot_figure3_baseline.py"),
            "--baseline-metrics", "baseline_outputs/baseline_metrics_summary.csv",
            "--full-model-csv", "figure2_outputs/Figure2_validation_degradation_data.csv",
            "--protocol", "pair_ratio_group",
            "--outdir", "figure3_outputs",
        ], "Figure 3 baseline comparison")

        run([
            py, str(SCRIPTS / "11_plot_figure4_ablation.py"),
            "--ablation-summary", "ablation_outputs/ablation_metrics_summary.csv",
            "--protocol", "pair_ratio_group",
            "--model", "extra_trees",
            "--viscosity-variant", "log10",
            "--outdir", "figure4_outputs",
        ], "Figure 4 feature ablation")

        if args.include_si:
            run([
                py, str(SCRIPTS / "12_plot_figure5_viscosity.py"),
                "--visc-dir", "viscosity_outputs",
                "--protocol-d-dir", "protocol_D_extrapolative_outputs",
                "--outdir", "figureS4_viscosity_outputs",
            ], "Figure S4 viscosity diagnostic")

        run([
            py, str(SCRIPTS / "08_build_main_tables.py"),
            "--dataset", str(dataset_root),
            "--figure2-csv", "figure2_outputs/Figure2_validation_degradation_data.csv",
            "--baseline-csv", "baseline_outputs/baseline_metrics_summary.csv",
            "--ablation-dir", "ablation_outputs",
            "--shap-dir", "shap_outputs_full",
            "--outdir", "manuscript_tables",
        ], "Build manuscript tables")

    print("\n[DONE] Workflow complete.")


if __name__ == "__main__":
    main()
