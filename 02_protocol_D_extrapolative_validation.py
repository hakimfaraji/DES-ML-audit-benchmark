#!/usr/bin/env python3
"""
Line 1 Protocol D: all-property extrapolative validation.

Inputs:
  Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Outputs:
  protocol_D_extrapolative_outputs/
    protocol_D_metrics_summary.csv
    protocol_D_metrics_long.csv
    protocol_D_predictions.csv
    protocol_D_run_manifest.json

Purpose:
  Extend the mid-diagnostic validation audit beyond Protocols A-C by evaluating
  leave-HBA-out and leave-HBD-out generalization for all five properties.

Notes:
  - This script is intentionally conservative: all target/property columns are removed from X.
  - It reports target_in_X and property_cols_in_X safeguards.
  - It uses GroupKFold by component identity as a computationally stable approximation
    of leave-component-out when there are many groups. Set USE_TRUE_LOGO=True for exact
    LeaveOneGroupOut, but this can be slow for many unique components.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

RANDOM_STATE = 42
USE_TRUE_LOGO = False
MAX_GROUP_FOLDS = 5

DATASET = "Unified_DES_dataset_GOLD_descriptor_ready_subset.csv"
OUTDIR = Path("protocol_D_extrapolative_outputs")
OUTDIR.mkdir(exist_ok=True)

TARGETS = {
    "density": "density_g_cm3",
    "viscosity": "viscosity_mpa_s",
    "conductivity": "conductivity_ms_cm",
    "surface_tension": "surface_tension_mn_m",
    "refractive_index": "refractive_index",
}

PROPERTY_COLS = list(TARGETS.values())

META_DROP_EXACT = {
    "unified_row_id", "source_corpus", "source_period", "source_origin",
    "source_filename", "source_schema_variant", "article_title", "journal",
    "year", "doi", "entry_id_local", "composition_label",
    "component_1_name_raw", "component_2_name_raw", "hba_name_raw",
    "hbd_name_raw", "molar_ratio_raw", "ratio_basis", "special_ratio_note",
    "stability_flag", "measurement_temperature_mode", "measurement_condition_note",
    "reference_ids_raw", "reference_titles_raw", "reference_dois_raw",
    "traceability_note", "gold_primary_reason", "relaxed_primary_reason",
    "hba_name_canonical", "hba_slug_canonical", "hba_component_registry_id",
    "hbd_name_canonical", "hbd_slug_canonical", "hbd_component_registry_id",
    "hba_name_resolved", "hbd_name_resolved", "smiles_hba", "smiles_hbd",
    "hba_smiles", "hbd_smiles", "hba_canonical_name", "hbd_canonical_name",
    "hba_canonical_name_resolved", "hbd_canonical_name_resolved",
    "hba_canonical_slug", "hbd_canonical_slug", "hba_observed_roles",
    "hbd_observed_roles", "hba_preferred_role", "hbd_preferred_role",
}

def infer_ratio_features(df):
    feats = []
    for c in df.columns:
        lc = c.lower()
        if ("ratio" in lc or "composition" in lc) and pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats

def feature_sets(df):
    temp = [c for c in ["measurement_temperature_c"] if c in df.columns]
    descriptors = [c for c in df.columns if ("descriptor_" in c and pd.api.types.is_numeric_dtype(df[c]))]
    ratio = infer_ratio_features(df)
    allowed_binary_flags = [
        c for c in df.columns
        if (c.startswith("is_") or c.startswith("has_") or c.startswith("contains_") or c.endswith("_ready"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    full = sorted(set(temp + descriptors + ratio + allowed_binary_flags))
    return {
        "temperature_only": temp,
        "descriptors_only": descriptors,
        "descriptors_ratio_temp": sorted(set(descriptors + ratio + temp)),
        "full": full,
    }

def clean_feature_list(cols, target_col):
    return [c for c in cols if c != target_col and c not in PROPERTY_COLS and c not in META_DROP_EXACT]

def make_model(name):
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
    if name == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                min_samples_leaf=2,
                n_jobs=-1
            ))
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                random_state=RANDOM_STATE,
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01
            ))
        ])
    raise ValueError(name)

def evaluate_cv(X, y, groups, model_name, task, target_col, protocol, group_definition, feature_set_name):
    unique_groups = pd.Series(groups).dropna().unique()
    if len(unique_groups) < 2:
        return [], []
    if USE_TRUE_LOGO:
        splitter = LeaveOneGroupOut()
        splits = splitter.split(X, y, groups)
    else:
        n_splits = min(MAX_GROUP_FOLDS, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups)

    rows = []
    preds = []
    for fold, (tr, te) in enumerate(splits, start=1):
        model = make_model(model_name)
        model.fit(X.iloc[tr], y.iloc[tr])
        yhat = model.predict(X.iloc[te])
        mae = mean_absolute_error(y.iloc[te], yhat)
        rmse = mean_squared_error(y.iloc[te], yhat) ** 0.5
        r2 = r2_score(y.iloc[te], yhat) if len(te) > 1 else np.nan
        rows.append({
            "task": task, "target_col": target_col, "protocol": protocol,
            "group_definition": group_definition, "feature_set": feature_set_name,
            "model": model_name, "fold": fold, "n_train": len(tr), "n_test": len(te),
            "n_features": X.shape[1], "mae": mae, "rmse": rmse, "r2": r2,
        })
        pred_df = pd.DataFrame({
            "task": task, "protocol": protocol, "group_definition": group_definition,
            "feature_set": feature_set_name, "model": model_name, "fold": fold,
            "y_true": y.iloc[te].to_numpy(), "y_pred": yhat, "group": pd.Series(groups).iloc[te].to_numpy()
        })
        preds.append(pred_df)
    return rows, preds

def summarize(long_df):
    agg = long_df.groupby(["task","target_col","protocol","group_definition","feature_set","model"], as_index=False).agg(
        n_folds=("fold","nunique"),
        n_test_total=("n_test","sum"),
        n_features=("n_features","mean"),
        mae_mean=("mae","mean"),
        mae_std=("mae","std"),
        rmse_mean=("rmse","mean"),
        rmse_std=("rmse","std"),
        r2_mean=("r2","mean"),
        r2_std=("r2","std"),
    )
    return agg

def main():
    df = pd.read_csv(DATASET)
    fs_all = feature_sets(df)
    all_rows = []
    all_preds = []
    audit_rows = []

    for task, target_col in TARGETS.items():
        dft = df[df[target_col].notna()].copy()
        for protocol, group_col in [
            ("D_leave_HBA_out", "hba_name_resolved"),
            ("D_leave_HBD_out", "hbd_name_resolved"),
        ]:
            if group_col not in dft.columns:
                continue
            dfg = dft[dft[group_col].notna()].copy()
            groups = dfg[group_col].astype(str)
            for fs_name, cols0 in fs_all.items():
                cols = clean_feature_list(cols0, target_col)
                # keep only numeric
                cols = [c for c in cols if c in dfg.columns and pd.api.types.is_numeric_dtype(dfg[c])]
                if not cols:
                    continue
                X = dfg[cols].copy()
                y = dfg[target_col].astype(float)
                audit_rows.append({
                    "task": task, "target_col": target_col, "protocol": protocol,
                    "feature_set": fs_name, "n_rows": len(dfg), "n_features": len(cols),
                    "target_in_X": target_col in cols,
                    "property_cols_in_X": ";".join([c for c in PROPERTY_COLS if c in cols]),
                    "group_col": group_col,
                    "n_groups": groups.nunique(),
                })
                for model_name in ["dummy_mean", "ridge", "extra_trees", "hist_gradient_boosting"]:
                    rows, preds = evaluate_cv(
                        X, y, groups, model_name, task, target_col,
                        protocol, group_col, fs_name
                    )
                    all_rows.extend(rows)
                    all_preds.extend(preds)

    long_df = pd.DataFrame(all_rows)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    summary = summarize(long_df) if not long_df.empty else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)

    long_df.to_csv(OUTDIR / "protocol_D_metrics_long.csv", index=False)
    summary.to_csv(OUTDIR / "protocol_D_metrics_summary.csv", index=False)
    pred_df.to_csv(OUTDIR / "protocol_D_predictions.csv", index=False)
    audit.to_csv(OUTDIR / "protocol_D_leakage_audit.csv", index=False)

    manifest = {
        "dataset": DATASET,
        "random_state": RANDOM_STATE,
        "use_true_leave_one_group_out": USE_TRUE_LOGO,
        "max_group_folds_if_groupkfold": MAX_GROUP_FOLDS,
        "targets": TARGETS,
        "protocols": ["D_leave_HBA_out", "D_leave_HBD_out"],
    }
    with open(OUTDIR / "protocol_D_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
