# Manuscript output map

This file maps each manuscript item to the script and expected output file used to reproduce it.

## Main-text items

| Manuscript item | Script | Primary output |
|---|---|---|
| Table 1. Dataset overview | `scripts/24_build_final_main_assets.py` | `main_text_assets/tables/Table1_dataset_overview.csv` |
| Table 2. Validation protocols | `scripts/24_build_final_main_assets.py` | `main_text_assets/tables/Table2_validation_protocols.csv` |
| Table 3. Validation performance with 95% CI | `scripts/24_build_final_main_assets.py` | `main_text_assets/tables/Table3_validation_performance_with_CI.csv` |
| Table 4. Baseline/effect-size comparison | `scripts/24_build_final_main_assets.py` | `main_text_assets/tables/Table4_baseline_effect_size.csv` |
| Figure 1. Audit workflow | `scripts/17_build_main_figure1_audit_workflow.py` | `Figure1_leakage_aware_audit_workflow.png/pdf` |
| Figure 2. Validation sensitivity | `scripts/24_build_final_main_assets.py` | `main_text_assets/figures/Figure2_validation_sensitivity.png/pdf` |
| Figure 3. Baseline comparison | `scripts/24_build_final_main_assets.py` | `main_text_assets/figures/Figure3_baseline_comparison.png/pdf` |
| Figure 4. Feature ablation | `scripts/24_build_final_main_assets.py` | `main_text_assets/figures/Figure4_feature_ablation.png/pdf` |

## Supplementary items

| Supplementary item | Script | Primary output |
|---|---|---|
| Figure S1. Dataset imbalance and coverage | `scripts/18_build_figure_s1_dataset_imbalance.py` | `FigureS1_dataset_imbalance_and_coverage.png/pdf` |
| Figure S4. Applicability-domain analysis | `scripts/20_build_figure_s4_applicability_domain_phase4.py` | `FigureS4_applicability_domain_phase4.png/pdf` |
| Figure S5. Nearest-neighbor baseline | `scripts/21b_build_figure_s5_nn_baseline_publication.py` | `FigureS5_NN_baseline_publication.png/pdf` |
| Figure S6. Viscosity diagnostics | `scripts/22_build_figure_s6_viscosity_diagnostics.py` | `FigureS6_extended_viscosity_diagnostics.png/pdf` |
| Figure S7. Clean SHAP summary | `scripts/23_build_clean_shap_summary.py` | `FigureS7_SHAP_clean_summary.png/pdf` |
| Table S4. Dataset imbalance summary | `scripts/13_dataset_imbalance_analysis.py` | `imbalance_summary.csv` |
| Table S5. Leakage audit summary | `scripts/16_leakage_audit_table.py` | `Table_S5_leakage_audit_summary.csv` |
| Table S6. Applicability-domain quartiles | `scripts/14_applicability_domain_analysis.py` | `distance_error_quartiles.csv` |
| Table S7. Nearest-neighbor metrics | `scripts/15_nearest_neighbor_baseline.py` | `knn_baseline_summary.csv` |

## Recommended verification order

1. `python run_manuscript_assets.py --outdir reproduced_outputs`
2. Check `reproduced_outputs/main_text_assets/tables/` and `reproduced_outputs/main_text_assets/figures/`.
3. Run `python run_upgrade_pipeline.py --outdir full_reproduction_outputs --fast-original` for upgrade analyses and supplementary outputs.
