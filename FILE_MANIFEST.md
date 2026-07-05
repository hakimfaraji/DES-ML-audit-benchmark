# File manifest

## Data

- `data/Unified_DES_dataset_GOLD_descriptor_ready_subset.csv` — frozen GOLD descriptor-ready dataset used for all reported analyses.

## Top-level runners

- `run_manuscript_assets.py` — lightweight reproduction of main manuscript Tables 1–4 and Figures 2–4.
- `run_pipeline.py` — core leakage-aware audit workflow.
- `run_upgrade_pipeline.py` — integrated workflow including dataset imbalance, applicability-domain, nearest-neighbor, leakage-audit, and final supplementary assets.

## Reference outputs

- `reference_outputs/main_text_assets/` — manuscript-ready tables and figures generated from `run_manuscript_assets.py`.

## Documentation

- `README.md` — installation and execution instructions.
- `docs/MANUSCRIPT_OUTPUT_MAP.md` — mapping between manuscript items and scripts.
- `docs/REPRODUCIBILITY_CHECKLIST.md` — minimal and extended reproducibility checks.
- `docs/SCRIPT_INVENTORY.md` — short description of each script.
