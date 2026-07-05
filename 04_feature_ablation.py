#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line 1 DES manuscript — protocol-aligned all-property feature ablation analysis

Purpose
-------
Quantify whether chemistry-informed descriptor blocks add predictive value beyond
ratio and temperature information under the same validation logic used in the
Line 1 diagnostic/baseline stages.

Expected input
--------------
Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Main outputs
------------
ablation_outputs/
  ablation_metrics_summary.csv
  ablation_metrics_long.csv
  ablation_predictions.csv
  ablation_leakage_audit.csv
  ablation_delta_vs_descriptors_only.csv
  ablation_incremental_gains.csv
  ablation_best_by_property_protocol.csv
  ablation_run_manifest.json
  README_ablation_outputs.md

Validation protocols
--------------------
- random_kfold
- pair_group
- pair_ratio_group
- leave_hba_out
- leave_hbd_out

Feature blocks
--------------
- descriptors_only
- descriptors_plus_ratio
- descriptors_plus_ratio_temp
- descriptors_plus_ratio_temp_interactions
- full_leakage_safe

Models
------
- Ridge
- ExtraTreesRegressor
- HistGradientBoostingRegressor

Leakage policy
--------------
Target/property columns are always removed from X. The script audits:
- whether the current target appears in X;
- whether any property/target column appears in X;
- which columns were dropped.

Viscosity is evaluated as both raw viscosity and log10(viscosity).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS: Dict[str, str] = {
    "density": "density_g_cm3",
    "viscosity": "viscosity_mpa_s",
    "conductivity": "conductivity_ms_cm",
    "surface_tension": "surface_tension_mn_m",
    "refractive_index": "refractive_index",
}

ALL_TARGET_COLS = list(TARGETS.values()) + ["tm_c"]

META_DROP_EXACT = {
    "unified_row_id", "source_corpus", "source_period", "source_origin", "source_filename",
    "source_schema_variant", "article_title", "journal", "year", "doi", "entry_id_local",
    "composition_label", "component_1_name_raw", "component_2_name_raw", "hba_name_raw",
    "hbd_name_raw", "ratio_basis", "special_ratio_note", "stability_flag",
    "measurement_temperature_mode", "measurement_condition_note", "reference_ids_raw",
    "reference_titles_raw", "reference_dois_raw", "traceability_note", "gold_inclusion_status",
    "gold_primary_reason", "relaxed_inclusion_status", "relaxed_primary_reason",
    "hba_name_canonical", "hba_slug_canonical", "hba_component_registry_id",
    "hbd_name_canonical", "hbd_slug_canonical", "hbd_component_registry_id",
    "hba_name_resolved", "hbd_name_resolved", "smiles_hba", "smiles_hbd",
    "hba_smiles_mapped", "hbd_smiles_mapped", "hba_smiles", "hbd_smiles",
    "hba_canonical_name", "hba_canonical_name_resolved", "hba_canonical_slug",
    "hbd_canonical_name", "hbd_canonical_name_resolved", "hbd_canonical_slug",
    "pair_id", "pair_ratio_id", "hba_group", "hbd_group",
}

META_DROP_PATTERNS = [
    r".*_raw$", r".*_note$", r".*_title.*", r".*doi.*", r".*reference.*",
    r".*source.*", r".*schema.*", r".*slug.*", r".*registry.*", r".*smiles.*",
    r".*canonical_name.*", r".*observed_roles.*", r".*preferred_role.*",
]

RATIO_FEATURES = ["parsed_hba_ratio", "parsed_hbd_ratio", "parsed_hba_fraction", "parsed_hbd_fraction"]
TEMP_FEATURES = ["measurement_temperature_c"]


def parse_ratio(raw) -> Tuple[float, float, float, float]:
    if pd.isna(raw):
        return (np.nan, np.nan, np.nan, np.nan)
    s = str(raw).strip().lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"\b(mol|molar|ratio|hba|hbd)\b", " ", s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        a, b = 1.0, float(nums[0])
    else:
        return (np.nan, np.nan, np.nan, np.nan)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    total = a + b
    return (a, b, a / total, b / total)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out.get("molar_ratio_raw", pd.Series([np.nan] * len(out), index=out.index)).apply(parse_ratio)
    ratio_df = pd.DataFrame(parsed.tolist(), columns=RATIO_FEATURES, index=out.index)
    # Assign rather than concat so existing parsed-ratio columns are overwritten,
    # avoiding duplicate column names if the input dataset already contains them.
    for col in RATIO_FEATURES:
        out[col] = pd.to_numeric(ratio_df[col], errors="coerce").astype(float)

    hba_name = out.get("hba_name_canonical", out.get("hba_name_raw", pd.Series(["NA"] * len(out), index=out.index))).astype(str).fillna("NA")
    hbd_name = out.get("hbd_name_canonical", out.get("hbd_name_raw", pd.Series(["NA"] * len(out), index=out.index))).astype(str).fillna("NA")
    out["pair_id"] = hba_name + " || " + hbd_name
    out["pair_ratio_id"] = out["pair_id"] + " || " + out.get("molar_ratio_raw", pd.Series(["NA"] * len(out))).astype(str).fillna("NA")
    out["hba_group"] = hba_name
    out["hbd_group"] = hbd_name

    # Pairwise descriptor interaction features: for shared HBA/HBD descriptor suffixes,
    # add sum, absolute difference, and product. These are chemistry-informed but not targets.
    hba_desc = [c for c in out.columns if c.startswith("hba_descriptor_")]
    for hba_col in hba_desc:
        suffix = hba_col.replace("hba_descriptor_", "")
        hbd_col = "hbd_descriptor_" + suffix
        if hbd_col not in out.columns:
            continue
        # Some RDKit/descriptive columns may be boolean. Pandas keeps boolean dtype
        # after pd.to_numeric(), and boolean subtraction is not supported in recent
        # pandas/numpy versions. Cast explicitly to float before arithmetic.
        a = pd.to_numeric(out[hba_col], errors="coerce").astype(float)
        b = pd.to_numeric(out[hbd_col], errors="coerce").astype(float)
        out[f"pair_sum_{suffix}"] = a + b
        out[f"pair_absdiff_{suffix}"] = (a - b).abs()
        out[f"pair_product_{suffix}"] = a * b
    return out


def is_meta_column(col: str) -> bool:
    if col in META_DROP_EXACT:
        return True
    return any(re.fullmatch(pat, col) for pat in META_DROP_PATTERNS)


def numeric_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    existing = [c for c in cols if c in df.columns]
    out = []
    for c in existing:
        # Bool columns are acceptable numeric features after coercion by pandas/sklearn.
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            out.append(c)
        else:
            # keep parsed/generated features if they can be coerced numerically
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > 0:
                df[c] = coerced
                out.append(c)
    return out


def get_descriptor_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("hba_descriptor_") or c.startswith("hbd_descriptor_")]
    return numeric_existing(df, cols)


def get_interaction_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("pair_sum_") or c.startswith("pair_absdiff_") or c.startswith("pair_product_")]
    return numeric_existing(df, cols)


def get_full_safe_cols(df: pd.DataFrame) -> List[str]:
    # Full leakage-safe numeric feature space: all numeric columns except targets and metadata.
    cols = []
    for c in df.columns:
        if c in ALL_TARGET_COLS or is_meta_column(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def get_feature_cols(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    descriptors = get_descriptor_cols(df)
    interactions = get_interaction_cols(df)
    ratio = numeric_existing(df, RATIO_FEATURES)
    temp = numeric_existing(df, TEMP_FEATURES)

    if feature_set == "descriptors_only":
        cols = descriptors
    elif feature_set == "descriptors_plus_ratio":
        cols = descriptors + ratio
    elif feature_set == "descriptors_plus_ratio_temp":
        cols = descriptors + ratio + temp
    elif feature_set == "descriptors_plus_ratio_temp_interactions":
        cols = descriptors + ratio + temp + interactions
    elif feature_set == "full_leakage_safe":
        cols = get_full_safe_cols(df)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Final hard leakage guard.
    dropped = [c for c in cols if c in ALL_TARGET_COLS]
    cols = [c for c in cols if c not in ALL_TARGET_COLS]
    # De-duplicate while preserving order.
    cols = list(dict.fromkeys(cols))
    return cols, dropped


def make_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            )
        ],
        remainder="drop",
    )


def build_model(model_name: str, numeric_cols: List[str], seed: int):
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
        return Pipeline([("prep", make_preprocessor(numeric_cols)), ("model", model)])
    if model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        return Pipeline([("prep", make_preprocessor(numeric_cols)), ("model", model)])
    if model_name == "hist_gradient_boosting":
        # HGBR handles nonlinearities but still benefits from imputation/scaling consistency.
        model = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
        )
        return Pipeline([("prep", make_preprocessor(numeric_cols)), ("model", model)])
    raise ValueError(f"Unknown model_name: {model_name}")


def get_cv(protocol: str, df_task: pd.DataFrame, n_splits: int, seed: int):
    if protocol == "random_kfold":
        cv = KFold(n_splits=min(n_splits, len(df_task)), shuffle=True, random_state=seed)
        return cv.split(df_task), None, "random row-wise KFold"
    if protocol == "pair_group":
        groups = df_task["pair_id"].astype(str).values
        n_groups = len(pd.unique(groups))
        if n_groups < 2:
            return None, groups, "insufficient pair groups"
        cv = GroupKFold(n_splits=min(n_splits, n_groups))
        return cv.split(df_task, groups=groups), groups, "GroupKFold by HBA-HBD pair"
    if protocol == "pair_ratio_group":
        groups = df_task["pair_ratio_id"].astype(str).values
        n_groups = len(pd.unique(groups))
        if n_groups < 2:
            return None, groups, "insufficient pair+ratio groups"
        cv = GroupKFold(n_splits=min(n_splits, n_groups))
        return cv.split(df_task, groups=groups), groups, "GroupKFold by HBA-HBD pair plus molar ratio"
    if protocol == "leave_hba_out":
        groups = df_task["hba_group"].astype(str).values
        if len(pd.unique(groups)) < 2:
            return None, groups, "insufficient HBA groups"
        cv = LeaveOneGroupOut()
        return cv.split(df_task, groups=groups), groups, "LeaveOneGroupOut by HBA"
    if protocol == "leave_hbd_out":
        groups = df_task["hbd_group"].astype(str).values
        if len(pd.unique(groups)) < 2:
            return None, groups, "insufficient HBD groups"
        cv = LeaveOneGroupOut()
        return cv.split(df_task, groups=groups), groups, "LeaveOneGroupOut by HBD"
    raise ValueError(f"Unknown protocol: {protocol}")


def safe_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n_eval": 0}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    return {"r2": float(r2), "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(rmse), "n_eval": int(len(y_true))}


def run_one(df: pd.DataFrame, property_name: str, target_col: str, target_variant: str,
            protocol: str, feature_set: str, model_name: str, n_splits: int, seed: int,
            min_test_size: int = 2):
    df_task = df.loc[df[target_col].notna()].copy()
    if target_variant == "log10" and property_name == "viscosity":
        df_task = df_task.loc[df_task[target_col] > 0].copy()
        y_all = np.log10(df_task[target_col].astype(float).values)
    elif target_variant == "raw":
        y_all = df_task[target_col].astype(float).values
    else:
        return None, [], None, None

    if len(df_task) < 10:
        return None, [], None, None

    feature_cols, forcibly_dropped = get_feature_cols(df_task, feature_set)
    if len(feature_cols) == 0:
        return None, [], {
            "property": property_name, "target_col": target_col, "target_variant": target_variant,
            "protocol": protocol, "feature_set": feature_set, "model": model_name,
            "status": "skipped_no_features", "selected_features": "",
            "target_in_X": False, "any_property_col_in_X": False,
            "property_cols_in_X": "", "forced_dropped_property_cols": "|".join(forcibly_dropped),
            "n_rows_with_target": int(len(df_task)),
        }, None

    leakage = {
        "property": property_name,
        "target_col": target_col,
        "target_variant": target_variant,
        "protocol": protocol,
        "feature_set": feature_set,
        "model": model_name,
        "n_features": int(len(feature_cols)),
        "selected_features": "|".join(feature_cols),
        "target_in_X": target_col in feature_cols,
        "any_property_col_in_X": any(c in ALL_TARGET_COLS for c in feature_cols),
        "property_cols_in_X": "|".join([c for c in feature_cols if c in ALL_TARGET_COLS]),
        "forced_dropped_property_cols": "|".join(forcibly_dropped),
        "n_rows_with_target": int(len(df_task)),
    }

    split_iter, groups, group_desc = get_cv(protocol, df_task, n_splits, seed)
    leakage["group_definition"] = group_desc
    if split_iter is None:
        leakage["status"] = "skipped"
        return None, [], leakage, None

    estimator = build_model(model_name, feature_cols, seed)
    X_all = df_task[feature_cols].copy()
    preds = np.full(len(df_task), np.nan, dtype=float)
    fold_rows = []
    n_folds_run = 0

    for fold_id, (tr, te) in enumerate(split_iter):
        if len(te) < min_test_size or len(tr) < 5:
            continue
        est = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_all.iloc[tr], y_all[tr])
            preds[te] = est.predict(X_all.iloc[te])
        m = safe_metrics(y_all[te], preds[te])
        fold_rows.append({
            "property": property_name, "target_col": target_col, "target_variant": target_variant,
            "protocol": protocol, "feature_set": feature_set, "model": model_name,
            "fold_id": fold_id, "n_train": int(len(tr)), "n_test": int(len(te)),
            "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"],
            "group_definition": group_desc,
        })
        n_folds_run += 1

    metrics = safe_metrics(y_all, preds)
    metrics_summary = {
        "property": property_name,
        "target_col": target_col,
        "target_variant": target_variant,
        "protocol": protocol,
        "feature_set": feature_set,
        "model": model_name,
        "n_total": int(len(df_task)),
        "n_eval": metrics["n_eval"],
        "n_folds_run": int(n_folds_run),
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_features": int(len(feature_cols)),
        "features_used": "|".join(feature_cols),
        "group_definition": group_desc,
    }

    id_cols = [c for c in ["unified_row_id", "hba_name_canonical", "hbd_name_canonical", "molar_ratio_raw", "measurement_temperature_c", target_col] if c in df_task.columns]
    pred_df = df_task[id_cols].copy()
    pred_df.insert(0, "property", property_name)
    pred_df.insert(1, "target_variant", target_variant)
    pred_df.insert(2, "protocol", protocol)
    pred_df.insert(3, "feature_set", feature_set)
    pred_df.insert(4, "model", model_name)
    pred_df["y_true_model_space"] = y_all
    pred_df["y_pred_model_space"] = preds
    pred_df["residual_model_space"] = pred_df["y_true_model_space"] - pred_df["y_pred_model_space"]
    if target_variant == "log10" and property_name == "viscosity":
        pred_df["y_pred_raw_backtransformed"] = np.power(10.0, pred_df["y_pred_model_space"])
    else:
        pred_df["y_pred_raw_backtransformed"] = pred_df["y_pred_model_space"]

    leakage["status"] = "ok"
    return metrics_summary, fold_rows, leakage, pred_df


def write_incremental_tables(metrics: pd.DataFrame, out_dir: Path):
    if metrics.empty:
        return
    base_keys = ["property", "target_variant", "protocol", "model"]

    desc = metrics.loc[metrics["feature_set"] == "descriptors_only", base_keys + ["r2", "mae", "rmse"]].rename(
        columns={"r2": "descriptors_only_r2", "mae": "descriptors_only_mae", "rmse": "descriptors_only_rmse"}
    )
    delta = metrics.merge(desc, on=base_keys, how="left")
    delta["delta_r2_vs_descriptors_only"] = delta["r2"] - delta["descriptors_only_r2"]
    delta["delta_mae_vs_descriptors_only"] = delta["descriptors_only_mae"] - delta["mae"]
    delta["delta_rmse_vs_descriptors_only"] = delta["descriptors_only_rmse"] - delta["rmse"]
    delta.to_csv(out_dir / "ablation_delta_vs_descriptors_only.csv", index=False)

    # Incremental step gains for the canonical ordered ablation path.
    order = [
        "descriptors_only",
        "descriptors_plus_ratio",
        "descriptors_plus_ratio_temp",
        "descriptors_plus_ratio_temp_interactions",
        "full_leakage_safe",
    ]
    rows = []
    for key, g in metrics.groupby(base_keys, dropna=False):
        g2 = g.set_index("feature_set")
        for prev, curr in zip(order[:-1], order[1:]):
            if prev not in g2.index or curr not in g2.index:
                continue
            prev_row = g2.loc[prev]
            curr_row = g2.loc[curr]
            if isinstance(prev_row, pd.DataFrame):
                prev_row = prev_row.iloc[0]
            if isinstance(curr_row, pd.DataFrame):
                curr_row = curr_row.iloc[0]
            rows.append({
                "property": key[0], "target_variant": key[1], "protocol": key[2], "model": key[3],
                "from_feature_set": prev, "to_feature_set": curr,
                "delta_r2": curr_row["r2"] - prev_row["r2"],
                "delta_mae_improvement": prev_row["mae"] - curr_row["mae"],
                "delta_rmse_improvement": prev_row["rmse"] - curr_row["rmse"],
                "r2_from": prev_row["r2"], "r2_to": curr_row["r2"],
                "mae_from": prev_row["mae"], "mae_to": curr_row["mae"],
                "rmse_from": prev_row["rmse"], "rmse_to": curr_row["rmse"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "ablation_incremental_gains.csv", index=False)

    best = (
        metrics.sort_values(["property", "target_variant", "protocol", "model", "r2"], ascending=[True, True, True, True, False])
        .groupby(["property", "target_variant", "protocol", "model"], as_index=False)
        .head(1)
    )
    best.to_csv(out_dir / "ablation_best_by_property_protocol.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--out_dir", default="ablation_outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_predictions", action="store_true", help="Write full out-of-fold predictions. Can be large.")
    parser.add_argument("--protocols", nargs="+", default=["random_kfold", "pair_group", "pair_ratio_group", "leave_hba_out", "leave_hbd_out"])
    parser.add_argument("--feature_sets", nargs="+", default=[
        "descriptors_only",
        "descriptors_plus_ratio",
        "descriptors_plus_ratio_temp",
        "descriptors_plus_ratio_temp_interactions",
        "full_leakage_safe",
    ])
    parser.add_argument("--models", nargs="+", default=["ridge", "extra_trees", "hist_gradient_boosting"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_engineered_features(df)

    metrics_rows = []
    fold_rows_all = []
    leakage_rows = []
    pred_parts = []

    for prop, target_col in TARGETS.items():
        target_variants = ["raw"] + (["log10"] if prop == "viscosity" else [])
        for target_variant in target_variants:
            for protocol in args.protocols:
                for feature_set in args.feature_sets:
                    for model_name in args.models:
                        result = run_one(
                            df=df, property_name=prop, target_col=target_col,
                            target_variant=target_variant, protocol=protocol,
                            feature_set=feature_set, model_name=model_name,
                            n_splits=args.n_splits, seed=args.seed,
                        )
                        m, folds, leak, pred = result
                        if m is not None:
                            metrics_rows.append(m)
                            fold_rows_all.extend(folds)
                            if args.write_predictions and pred is not None:
                                pred_parts.append(pred)
                        if leak is not None:
                            leakage_rows.append(leak)

    metrics = pd.DataFrame(metrics_rows)
    folds = pd.DataFrame(fold_rows_all)
    leakage = pd.DataFrame(leakage_rows)

    metrics.to_csv(out_dir / "ablation_metrics_summary.csv", index=False)
    folds.to_csv(out_dir / "ablation_metrics_long.csv", index=False)
    leakage.to_csv(out_dir / "ablation_leakage_audit.csv", index=False)

    if args.write_predictions:
        preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
        preds.to_csv(out_dir / "ablation_predictions.csv", index=False)

    write_incremental_tables(metrics, out_dir)

    manifest = {
        "script": "run_all_property_feature_ablation.py",
        "input_csv": str(args.input_csv),
        "out_dir": str(out_dir),
        "n_rows_input": int(len(df)),
        "n_columns_input_after_engineering": int(df.shape[1]),
        "targets": TARGETS,
        "protocols": args.protocols,
        "feature_sets": args.feature_sets,
        "models": args.models,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "viscosity_variants": ["raw", "log10"],
        "leakage_policy": "hard removal of all target/property columns from every feature set",
        "interaction_features": "sum, absolute difference, and product for matched HBA/HBD descriptor suffixes",
        "prediction_output": "written only if --write_predictions is supplied",
    }
    with open(out_dir / "ablation_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme = f"""# Feature ablation outputs

Generated by `run_all_property_feature_ablation.py`.

## Targets
{json.dumps(TARGETS, indent=2)}

## Protocols
{args.protocols}

## Feature sets
1. `descriptors_only`: HBA/HBD molecular descriptors only.
2. `descriptors_plus_ratio`: descriptors + parsed molar-ratio features.
3. `descriptors_plus_ratio_temp`: descriptors + ratio + measurement temperature.
4. `descriptors_plus_ratio_temp_interactions`: previous block + pairwise descriptor sum/absolute-difference/product terms.
5. `full_leakage_safe`: all numeric leakage-safe non-target/non-metadata columns.

## Models
{args.models}

## Main interpretation files
- `ablation_metrics_summary.csv`: pooled out-of-fold metrics.
- `ablation_metrics_long.csv`: fold-level metrics.
- `ablation_leakage_audit.csv`: confirms target/property columns are absent from X.
- `ablation_delta_vs_descriptors_only.csv`: how much each feature set improves over descriptors alone.
- `ablation_incremental_gains.csv`: stepwise gains along the ablation path.
- `ablation_best_by_property_protocol.csv`: best feature set for each property/protocol/model.

## Validity check
For a valid leakage-safe run, `ablation_leakage_audit.csv` should show:
- `target_in_X = False`
- `any_property_col_in_X = False`

## Recommended manuscript use
Use this stage to support or weaken the phrase "chemistry-informed features". If ratio/temperature dominate and descriptor additions do not improve extrapolative protocols, the manuscript should state that explicitly rather than overclaiming chemical generalization.
"""
    with open(out_dir / "README_ablation_outputs.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Done. Wrote ablation outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
