# Script inventory

## Core modeling and validation

- `01_diagnostic_audit.py` — diagnostic Protocol A and leakage-corrected/grouped Protocols B–C.
- `02_protocol_D_extrapolative_validation.py` — leave-HBA-out and leave-HBD-out extrapolative validation.
- `03_baseline_analysis.py` — mean, temperature-only, ratio-only, and related baseline analyses.
- `04_feature_ablation.py` — feature-block ablation analysis.
- `05_viscosity_diagnostic.py` — viscosity-specific diagnostic analysis.
- `06_permutation_interpretability.py` — permutation feature importance.
- `07_shap_interpretability.py` — SHAP analysis.

## Main manuscript assets

- `08_build_main_tables.py` — table builder used in the core workflow.
- `09_plot_figure2_validation.py` — validation-sensitivity plot from core validation outputs.
- `10_plot_figure3_baseline.py` — baseline-comparison plot from baseline outputs.
- `11_plot_figure4_ablation.py` — feature-ablation plot from ablation outputs.
- `17_build_main_figure1_audit_workflow.py` — conceptual audit workflow diagram.
- `19_build_figure2_validation_sensitivity_phase4.py` — revised Figure 2 with interpolation/extrapolation labeling.
- `24_build_final_main_assets.py` — lightweight final generator for main-text Tables 1–4 and Figures 2–4.

## Supplementary analyses and assets

- `12_plot_figure5_viscosity.py` — legacy viscosity diagnostic plotter retained for traceability.
- `13_dataset_imbalance_analysis.py` — dataset imbalance statistics and source data for Table S4.
- `14_applicability_domain_analysis.py` — distance-to-training analysis and source data for Table S6 / Figure S4.
- `15_nearest_neighbor_baseline.py` — nearest-neighbor baseline and source data for Table S7 / Figure S5.
- `16_leakage_audit_table.py` — leakage-audit summary for Table S5.
- `18_build_figure_s1_dataset_imbalance.py` — Figure S1.
- `20_build_figure_s4_applicability_domain_phase4.py` — Figure S4.
- `21_build_figure_s5_nn_baseline_phase4.py` — annotated Figure S5 variant.
- `21b_build_figure_s5_nn_baseline_publication.py` — publication-style Figure S5 with broken y-axis.
- `22_build_figure_s6_viscosity_diagnostics.py` — Figure S6.
- `23_build_clean_shap_summary.py` — Figure S7 and cleaned SHAP top-feature table.
