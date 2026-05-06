# DES Leakage-Aware Benchmark Pipeline

Reproducible code and data package for the manuscript:

**A Leakage-Aware Benchmark Study of Machine Learning Models for Deep Eutectic Solvent Property Prediction**

This package contains the frozen GOLD dataset, the leakage-aware modeling workflow, post-review upgrade analyses, and manuscript-ready table/figure generators. It is intended to support reviewer and reader verification of the reported benchmark results.

## Contents

```text
DES_ML_Audit_Benchmark_Reproducible_Pipeline_v1/
  data/
    Unified_DES_dataset_GOLD_descriptor_ready_subset.csv
    README.md
  scripts/
    01_diagnostic_audit.py
    02_protocol_D_extrapolative_validation.py
    03_baseline_analysis.py
    04_feature_ablation.py
    05_viscosity_diagnostic.py
    06_permutation_interpretability.py
    07_shap_interpretability.py
    08_build_main_tables.py
    09_plot_figure2_validation.py
    10_plot_figure3_baseline.py
    11_plot_figure4_ablation.py
    12_plot_figure5_viscosity.py
    13_dataset_imbalance_analysis.py
    14_applicability_domain_analysis.py
    15_nearest_neighbor_baseline.py
    16_leakage_audit_table.py
    17_build_main_figure1_audit_workflow.py
    18_build_figure_s1_dataset_imbalance.py
    19_build_figure2_validation_sensitivity_phase4.py
    20_build_figure_s4_applicability_domain_phase4.py
    21_build_figure_s5_nn_baseline_phase4.py
    21b_build_figure_s5_nn_baseline_publication.py
    22_build_figure_s6_viscosity_diagnostics.py
    23_build_clean_shap_summary.py
    24_build_final_main_assets.py
  reference_outputs/
    main_text_assets/
      tables/
      figures/
  docs/
    MANUSCRIPT_OUTPUT_MAP.md
    REPRODUCIBILITY_CHECKLIST.md
    SCRIPT_INVENTORY.md
    FILE_MANIFEST.md
  run_pipeline.py
  run_upgrade_pipeline.py
  run_manuscript_assets.py
  requirements.txt
  environment.yml
  LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

Alternatively, create a conda environment:

```bash
conda env create -f environment.yml
conda activate des-ml-audit
```

## Fast verification: regenerate main manuscript assets

This lightweight command regenerates the main-text Tables 1–4 and Figures 2–4 from the frozen dataset and the manuscript-ready summary metrics:

```bash
python run_manuscript_assets.py \
  --dataset data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir reproduced_outputs
```

Expected outputs:

```text
reproduced_outputs/main_text_assets/tables/Table1_dataset_overview.csv
reproduced_outputs/main_text_assets/tables/Table2_validation_protocols.csv
reproduced_outputs/main_text_assets/tables/Table3_validation_performance_with_CI.csv
reproduced_outputs/main_text_assets/tables/Table4_baseline_effect_size.csv
reproduced_outputs/main_text_assets/figures/Figure2_validation_sensitivity.png/pdf
reproduced_outputs/main_text_assets/figures/Figure3_baseline_comparison.png/pdf
reproduced_outputs/main_text_assets/figures/Figure4_feature_ablation.png/pdf
```

Figure 1 is a conceptual workflow diagram and can be regenerated separately:

```bash
python scripts/17_build_main_figure1_audit_workflow.py --outdir reproduced_outputs/main_text_assets/figures
```

## Full analysis workflow

To run the integrated leakage-aware workflow and upgrade analyses:

```bash
python run_upgrade_pipeline.py \
  --dataset data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir full_reproduction_outputs \
  --fast-original
```

For a shorter upgrade-only run that skips the original core workflow:

```bash
python run_upgrade_pipeline.py \
  --dataset data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir upgrade_only_outputs \
  --skip-original
```

## Output map

The exact mapping between manuscript tables/figures and scripts is provided in:

```text
docs/MANUSCRIPT_OUTPUT_MAP.md
```

## Notes

- The GOLD dataset is treated as frozen.
- Protocol A is diagnostic only and should not be interpreted as a valid performance estimate.
- Protocols B–D are leakage-safe and form the basis of the manuscript conclusions.
- SHAP analysis can be computationally more expensive than the other analyses. Clean Figure S7 generation requires a precomputed `shap_global_summary.csv` or a full SHAP run.
