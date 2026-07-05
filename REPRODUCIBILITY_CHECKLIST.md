# Reproducibility checklist

## Required input

- `data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv`

## Minimal verification

```bash
python run_manuscript_assets.py \
  --dataset data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir reproduced_outputs
```

Confirm that the following files exist:

- `reproduced_outputs/main_text_assets/tables/Table1_dataset_overview.csv`
- `reproduced_outputs/main_text_assets/tables/Table2_validation_protocols.csv`
- `reproduced_outputs/main_text_assets/tables/Table3_validation_performance_with_CI.csv`
- `reproduced_outputs/main_text_assets/tables/Table4_baseline_effect_size.csv`
- `reproduced_outputs/main_text_assets/figures/Figure2_validation_sensitivity.png`
- `reproduced_outputs/main_text_assets/figures/Figure3_baseline_comparison.png`
- `reproduced_outputs/main_text_assets/figures/Figure4_feature_ablation.png`

## Extended verification

```bash
python run_upgrade_pipeline.py \
  --dataset data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir full_reproduction_outputs \
  --fast-original
```

Confirm that the following directories are generated:

- `full_reproduction_outputs/dataset_imbalance_outputs/`
- `full_reproduction_outputs/applicability_domain_outputs/`
- `full_reproduction_outputs/knn_baseline_outputs/`
- `full_reproduction_outputs/si_tables/`
- `full_reproduction_outputs/main_figures/`
- `full_reproduction_outputs/si_figures/`

## Notes

- Full model training and SHAP runs may take longer than manuscript asset generation.
- Some stochastic model-training outputs may show small numerical differences across Python/scikit-learn versions. The manuscript-ready reference outputs are included in `reference_outputs/`.
