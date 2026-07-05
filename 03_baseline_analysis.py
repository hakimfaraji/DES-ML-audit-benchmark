#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line 1 DES manuscript — all-property trivial baseline analysis

Purpose
-------
Run leakage-safe, protocol-aligned trivial/simple baselines for all target
properties in the frozen GOLD descriptor-ready dataset.

This script is intentionally conservative:
- it never uses target/property columns as predictors;
- it keeps baseline features minimal and interpretable;
- it uses the same validation logic used in the diagnostic stage:
  random CV, pair group, pair+ratio group, leave-HBA-out, leave-HBD-out;
- it treats viscosity both as raw viscosity and log10(viscosity) because the
  viscosity diagnostic stage showed strong skewness.

Expected input
--------------
Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Main outputs
------------
baseline_outputs/
  baseline_metrics_summary.csv
  baseline_metrics_long.csv
  baseline_predictions.csv
  baseline_leakage_audit.csv
  baseline_run_manifest.json
  baseline_best_by_property_protocol.csv
  baseline_delta_vs_dummy.csv
  baseline_delta_vs_temperature.csv
  README_baseline_outputs.md

Notes
-----
This is not intended to replace the full ML benchmark. It quantifies whether
full models learn more than trivial mean, temperature, ratio, and identity
baselines under identical validation settings.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
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

TEXT_ID_COLS = [
    "unified_row_id",
    "source_corpus",
    "source_period",
    "source_origin",
    "source_filename",
    "source_schema_variant",
    "article_title",
    "journal",
    "doi",
    "entry_id_local",
    "composition_label",
    "component_1_name_raw",
    "component_2_name_raw",
    "hba_name_raw",
    "hbd_name_raw",
    "molar_ratio_raw",
    "ratio_basis",
    "special_ratio_note",
    "stability_flag",
    "measurement_temperature_mode",
    "measurement_condition_note",
    "reference_ids_raw",
    "reference_titles_raw",
    "reference_dois_raw",
    "traceability_note",
    "gold_inclusion_status",
    "gold_primary_reason",
    "relaxed_inclusion_status",
    "relaxed_primary_reason",
    "hba_name_canonical",
    "hba_slug_canonical",
    "hba_component_registry_id",
    "hbd_name_canonical",
    "hbd_slug_canonical",
    "hbd_component_registry_id",
    "hba_smiles",
    "hbd_smiles",
]

BOOL_FLAG_COLS = [
    "contains_footnote",
    "contains_structural_water",
    "contains_water",
    "is_binary_mixture_with_water",
    "is_derived_property",
    "is_hba_hbd_ambiguous",
    "is_multi_composition_data",
    "is_multi_pressure_data",
    "is_multi_temperature_data",
    "is_special_ratio_format",
    "is_ternary_system",
    "needs_manual_review",
    "is_unstable",
    "has_core_identity",
    "has_any_target_property",
    "hba_name_needs_manual_review",
    "hbd_name_needs_manual_review",
]


def parse_ratio(raw) -> Tuple[float, float, float, float]:
    """Parse common DES ratio strings such as '1:2', '1 : 2', '1/2', '2'.
    Returns hba_ratio, hbd_ratio, hba_fraction, hbd_fraction.
    Unknown formats return NaNs.
    """
    if pd.isna(raw):
        return (np.nan, np.nan, np.nan, np.nan)
    s = str(raw).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    # remove common labels
    s = re.sub(r"\b(mol|molar|ratio|hba|hbd)\b", " ", s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        # Interpret a single number as HBA:HBD = 1:number
        a, b = 1.0, float(nums[0])
    else:
        return (np.nan, np.nan, np.nan, np.nan)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    total = a + b
    return (a, b, a / total, b / total)


def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out.get("molar_ratio_raw", pd.Series([np.nan] * len(out))).apply(parse_ratio)
    ratio_df = pd.DataFrame(
        parsed.tolist(),
        columns=["parsed_hba_ratio", "parsed_hbd_ratio", "parsed_hba_fraction", "parsed_hbd_fraction"],
        index=out.index,
    )
    out = pd.concat([out, ratio_df], axis=1)
    out["pair_id"] = (
        out.get("hba_name_canonical", out.get("hba_name_raw", "")).astype(str).fillna("NA")
        + " || "
        + out.get("hbd_name_canonical", out.get("hbd_name_raw", "")).astype(str).fillna("NA")
    )
    out["pair_ratio_id"] = out["pair_id"] + " || " + out.get("molar_ratio_raw", "").astype(str).fillna("NA")
    out["hba_group"] = out.get("hba_name_canonical", out.get("hba_name_raw", "")).astype(str).fillna("NA")
    out["hbd_group"] = out.get("hbd_name_canonical", out.get("hbd_name_raw", "")).astype(str).fillna("NA")
    return out


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(feature_set: str):
    if feature_set == "dummy_mean":
        return DummyRegressor(strategy="mean"), [], []
    if feature_set == "temperature_only":
        return Ridge(alpha=1.0), ["measurement_temperature_c"], []
    if feature_set == "ratio_only":
        return Ridge(alpha=1.0), [
            "parsed_hba_ratio",
            "parsed_hbd_ratio",
            "parsed_hba_fraction",
            "parsed_hbd_fraction",
        ], []
    if feature_set == "temperature_plus_ratio":
        return Ridge(alpha=1.0), [
            "measurement_temperature_c",
            "parsed_hba_ratio",
            "parsed_hbd_ratio",
            "parsed_hba_fraction",
            "parsed_hbd_fraction",
        ], []
    if feature_set == "component_identity_only":
        return Ridge(alpha=1.0), [], ["hba_group", "hbd_group"]
    if feature_set == "temperature_ratio_identity":
        return Ridge(alpha=1.0), [
            "measurement_temperature_c",
            "parsed_hba_ratio",
            "parsed_hbd_ratio",
            "parsed_hba_fraction",
            "parsed_hbd_fraction",
        ], ["hba_group", "hbd_group"]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def build_pipeline(feature_set: str):
    model, num_cols, cat_cols = make_model(feature_set)
    if feature_set == "dummy_mean":
        return model, num_cols, cat_cols
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", model)]), num_cols, cat_cols


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
    return {
        "r2": float(r2),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "n_eval": int(len(y_true)),
    }


def run_one(df: pd.DataFrame, property_name: str, target_col: str, target_variant: str,
            protocol: str, feature_set: str, n_splits: int, seed: int,
            min_test_size: int = 2):
    df_task = df.loc[df[target_col].notna()].copy()
    if target_variant == "log10" and property_name == "viscosity":
        df_task = df_task.loc[df_task[target_col] > 0].copy()
        y_all = np.log10(df_task[target_col].astype(float).values)
    elif target_variant == "raw":
        y_all = df_task[target_col].astype(float).values
    else:
        return None, [], None

    if len(df_task) < 10:
        return None, [], None

    estimator, num_cols, cat_cols = build_pipeline(feature_set)
    selected_features = num_cols + cat_cols

    leakage = {
        "property": property_name,
        "target_col": target_col,
        "target_variant": target_variant,
        "protocol": protocol,
        "feature_set": feature_set,
        "selected_features": "|".join(selected_features),
        "target_in_X": target_col in selected_features,
        "any_property_col_in_X": any(c in ALL_TARGET_COLS for c in selected_features),
        "property_cols_in_X": "|".join([c for c in selected_features if c in ALL_TARGET_COLS]),
        "n_rows_with_target": int(len(df_task)),
    }

    split_iter, groups, group_desc = get_cv(protocol, df_task, n_splits, seed)
    if split_iter is None:
        return None, [], {**leakage, "group_definition": group_desc, "status": "skipped"}

    preds = np.full(len(df_task), np.nan, dtype=float)
    fold_rows = []
    n_folds_run = 0

    X_all = df_task[selected_features].copy() if selected_features else pd.DataFrame(index=df_task.index)

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
            "property": property_name,
            "target_col": target_col,
            "target_variant": target_variant,
            "protocol": protocol,
            "feature_set": feature_set,
            "fold_id": fold_id,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "r2": m["r2"],
            "mae": m["mae"],
            "rmse": m["rmse"],
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
        "n_total": int(len(df_task)),
        "n_eval": metrics["n_eval"],
        "n_folds_run": int(n_folds_run),
        "r2": metrics["r2"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "features_used": "|".join(selected_features) if selected_features else "none",
        "group_definition": group_desc,
    }

    pred_df = df_task[[
        "unified_row_id", "hba_name_canonical", "hbd_name_canonical", "molar_ratio_raw",
        "measurement_temperature_c", target_col
    ]].copy()
    pred_df.insert(0, "property", property_name)
    pred_df.insert(1, "target_variant", target_variant)
    pred_df.insert(2, "protocol", protocol)
    pred_df.insert(3, "feature_set", feature_set)
    pred_df["y_true_model_space"] = y_all
    pred_df["y_pred_model_space"] = preds
    pred_df["residual_model_space"] = pred_df["y_true_model_space"] - pred_df["y_pred_model_space"]

    # for log-viscosity, also provide back-transformed predictions
    if target_variant == "log10" and property_name == "viscosity":
        pred_df["y_pred_raw_backtransformed"] = np.power(10.0, pred_df["y_pred_model_space"])
    else:
        pred_df["y_pred_raw_backtransformed"] = pred_df["y_pred_model_space"]

    leakage["group_definition"] = group_desc
    leakage["status"] = "ok"
    return metrics_summary, fold_rows, leakage, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--out_dir", default="baseline_outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["random_kfold", "pair_group", "pair_ratio_group", "leave_hba_out", "leave_hbd_out"],
    )
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=[
            "dummy_mean",
            "temperature_only",
            "ratio_only",
            "temperature_plus_ratio",
            "component_identity_only",
            "temperature_ratio_identity",
        ],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_baseline_features(df)

    metrics_rows = []
    fold_rows_all = []
    leakage_rows = []
    pred_parts = []

    for prop, target_col in TARGETS.items():
        target_variants = ["raw"]
        if prop == "viscosity":
            target_variants.append("log10")
        for target_variant in target_variants:
            for protocol in args.protocols:
                for feature_set in args.feature_sets:
                    result = run_one(
                        df=df,
                        property_name=prop,
                        target_col=target_col,
                        target_variant=target_variant,
                        protocol=protocol,
                        feature_set=feature_set,
                        n_splits=args.n_splits,
                        seed=args.seed,
                    )
                    if result[0] is None:
                        # result may include leakage skip row
                        if len(result) >= 3 and result[2] is not None:
                            leakage_rows.append(result[2])
                        continue
                    m, folds, leak, pred = result
                    metrics_rows.append(m)
                    fold_rows_all.extend(folds)
                    leakage_rows.append(leak)
                    pred_parts.append(pred)

    metrics = pd.DataFrame(metrics_rows)
    folds = pd.DataFrame(fold_rows_all)
    leakage = pd.DataFrame(leakage_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    metrics.to_csv(out_dir / "baseline_metrics_summary.csv", index=False)
    folds.to_csv(out_dir / "baseline_metrics_long.csv", index=False)
    leakage.to_csv(out_dir / "baseline_leakage_audit.csv", index=False)
    preds.to_csv(out_dir / "baseline_predictions.csv", index=False)

    if not metrics.empty:
        # Best by property/protocol/target_variant using R2, but keep failures explicit.
        best = (
            metrics.sort_values(["property", "target_variant", "protocol", "r2"], ascending=[True, True, True, False])
            .groupby(["property", "target_variant", "protocol"], as_index=False)
            .head(1)
        )
        best.to_csv(out_dir / "baseline_best_by_property_protocol.csv", index=False)

        base_cols = ["property", "target_variant", "protocol"]
        dummy = metrics.loc[metrics["feature_set"] == "dummy_mean", base_cols + ["r2", "mae", "rmse"]].rename(
            columns={"r2": "dummy_r2", "mae": "dummy_mae", "rmse": "dummy_rmse"}
        )
        temp = metrics.loc[metrics["feature_set"] == "temperature_only", base_cols + ["r2", "mae", "rmse"]].rename(
            columns={"r2": "temperature_r2", "mae": "temperature_mae", "rmse": "temperature_rmse"}
        )

        delta_dummy = metrics.merge(dummy, on=base_cols, how="left")
        delta_dummy["delta_r2_vs_dummy"] = delta_dummy["r2"] - delta_dummy["dummy_r2"]
        delta_dummy["delta_mae_vs_dummy"] = delta_dummy["dummy_mae"] - delta_dummy["mae"]
        delta_dummy["delta_rmse_vs_dummy"] = delta_dummy["dummy_rmse"] - delta_dummy["rmse"]
        delta_dummy.to_csv(out_dir / "baseline_delta_vs_dummy.csv", index=False)

        delta_temp = metrics.merge(temp, on=base_cols, how="left")
        delta_temp["delta_r2_vs_temperature"] = delta_temp["r2"] - delta_temp["temperature_r2"]
        delta_temp["delta_mae_vs_temperature"] = delta_temp["temperature_mae"] - delta_temp["mae"]
        delta_temp["delta_rmse_vs_temperature"] = delta_temp["temperature_rmse"] - delta_temp["rmse"]
        delta_temp.to_csv(out_dir / "baseline_delta_vs_temperature.csv", index=False)

    manifest = {
        "script": "run_all_property_baseline_analysis.py",
        "input_csv": str(args.input_csv),
        "out_dir": str(out_dir),
        "n_rows_input": int(len(df)),
        "n_columns_input": int(df.shape[1]),
        "targets": TARGETS,
        "protocols": args.protocols,
        "feature_sets": args.feature_sets,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "ratio_parsing": "parsed from molar_ratio_raw; unknown formats imputed inside pipelines",
        "viscosity_variants": ["raw", "log10"],
        "leakage_policy": "target and all property columns are never included in baseline feature sets",
    }
    with open(out_dir / "baseline_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme = f"""# Baseline outputs

This folder was generated by `run_all_property_baseline_analysis.py`.

## What was tested

Targets:
{json.dumps(TARGETS, indent=2)}

Protocols:
{args.protocols}

Feature sets:
{args.feature_sets}

Viscosity was evaluated twice:
- raw viscosity
- log10(viscosity), only for positive viscosity values

## Interpretation rules

- `dummy_mean` is the mean-predictor baseline.
- `temperature_only` tests whether property prediction is mostly explained by measurement temperature.
- `ratio_only` tests whether the molar ratio alone is sufficient.
- `temperature_plus_ratio` is the minimal physicochemical baseline.
- `component_identity_only` tests memorization/interpolation based only on HBA/HBD identity.
- `temperature_ratio_identity` is a stronger but still simple baseline; under leave-component-out it cannot memorize unseen HBA/HBD categories.

Use `baseline_delta_vs_dummy.csv` and `baseline_delta_vs_temperature.csv` to decide whether a model/feature set adds value beyond trivial predictors.

## Leakage status

See `baseline_leakage_audit.csv`. A valid run should have:
- `target_in_X = False`
- `any_property_col_in_X = False`

"""
    with open(out_dir / "README_baseline_outputs.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Done. Wrote baseline outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
