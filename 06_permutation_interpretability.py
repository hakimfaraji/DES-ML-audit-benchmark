#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line 1 DES ML — all-property interpretability pipeline
======================================================

Purpose
-------
Run leakage-safe interpretability analyses that are aligned with the previous
validation, baseline, and feature-ablation logic.

Outputs
-------
interpretability_outputs/
  - permutation_importance_long.csv
  - permutation_importance_summary.csv
  - top_features_by_property_protocol_model.csv
  - shap_importance_long.csv                         (only if SHAP is installed)
  - shap_importance_summary.csv                      (only if SHAP is installed)
  - interpretability_run_manifest.json
  - leakage_audit_interpretability.csv
  - figures/top_permutation_<property>_<protocol>_<model>.png

Design choices
--------------
1. No target leakage:
   All target/property columns are removed from X for every task.
2. Same task definitions as earlier audit:
   density, surface tension, refractive index, conductivity, viscosity.
3. Viscosity is modeled on log10(viscosity) by default for interpretability,
   because previous diagnostics showed raw viscosity is strongly skewed.
4. Permutation importance is computed on held-out validation folds.
5. SHAP is optional. If SHAP is not installed, the script does not fail.
6. Protocols:
   - random
   - pair_group
   - pair_ratio_group
   - leave_hba_out
   - leave_hbd_out
7. Models:
   - ExtraTreesRegressor
   - HistGradientBoostingRegressor
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


TARGETS = {
    "density": "density_g_cm3",
    "viscosity": "viscosity_mpa_s",
    "conductivity": "conductivity_ms_cm",
    "surface_tension": "surface_tension_mn_m",
    "refractive_index": "refractive_index",
}
ALL_TARGET_COLS = list(TARGETS.values())

TEMP_COL = "measurement_temperature_c"
HBA_COL = "hba_name_resolved"
HBD_COL = "hbd_name_resolved"
HBA_FALLBACKS = ["hba_name_canonical", "hba_name_raw", "hba_canonical_name_resolved", "hba_canonical_name"]
HBD_FALLBACKS = ["hbd_name_canonical", "hbd_name_raw", "hbd_canonical_name_resolved", "hbd_canonical_name"]

META_DROP_LIKE = [
    "source_", "article", "journal", "doi", "entry_id", "composition_label",
    "component_1", "component_2", "reference_", "traceability",
    "gold_", "relaxed_", "smiles", "canonical_name", "canonical_slug",
    "observed_roles", "preferred_role", "registry_id", "validation",
    "molar_ratio_raw", "ratio_basis", "special_ratio_note",
]


def parse_ratio(raw) -> float:
    """Parse common DES ratio strings into HBA/(HBA+HBD) if possible.

    Examples:
        "1:2" -> 0.3333
        "1 : 4" -> 0.2
        "2/1" -> 0.6667
    Returns NaN if parsing is not safe.
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().lower()
    s = s.replace(" ", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)[:/]([0-9]*\.?[0-9]+)$", s)
    if not m:
        # handle strings such as hba:hbd=1:2 or 1:2 (mol/mol)
        nums = re.findall(r"[0-9]*\.?[0-9]+", s)
        if len(nums) >= 2:
            try:
                a, b = float(nums[0]), float(nums[1])
            except Exception:
                return np.nan
        else:
            return np.nan
    else:
        a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0 or not np.isfinite(a + b):
        return np.nan
    return a / (a + b)


def first_existing(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of {[preferred] + fallbacks}")


def add_engineered_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hba_mole_fraction_from_ratio" not in out.columns:
        out["hba_mole_fraction_from_ratio"] = out.get("molar_ratio_raw", pd.Series(np.nan, index=out.index)).map(parse_ratio)
    out["hbd_mole_fraction_from_ratio"] = 1.0 - out["hba_mole_fraction_from_ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["hba_to_hbd_ratio_numeric"] = out["hba_mole_fraction_from_ratio"] / out["hbd_mole_fraction_from_ratio"]
        out["hbd_to_hba_ratio_numeric"] = out["hbd_mole_fraction_from_ratio"] / out["hba_mole_fraction_from_ratio"]
    return out


def build_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hba = first_existing(out, HBA_COL, HBA_FALLBACKS)
    hbd = first_existing(out, HBD_COL, HBD_FALLBACKS)
    out["_hba_group"] = out[hba].fillna("UNKNOWN_HBA").astype(str).str.strip().str.lower()
    out["_hbd_group"] = out[hbd].fillna("UNKNOWN_HBD").astype(str).str.strip().str.lower()
    out["_pair_group"] = out["_hba_group"] + " || " + out["_hbd_group"]

    if "hba_mole_fraction_from_ratio" in out.columns:
        ratio_bin = out["hba_mole_fraction_from_ratio"].round(3).astype(str)
    else:
        ratio_bin = out.get("molar_ratio_raw", pd.Series("NA", index=out.index)).astype(str)
    out["_pair_ratio_group"] = out["_pair_group"] + " || ratio=" + ratio_bin.fillna("NA").astype(str)
    return out


def infer_feature_columns(df: pd.DataFrame, target_col: str, feature_mode: str) -> List[str]:
    """Return leakage-safe numeric features.

    feature_mode:
      - descriptors_ratio_temp: descriptor columns + ratio-engineered + temperature
      - full_safe: all numeric columns except target/property/meta-like columns
    """
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    banned = set(ALL_TARGET_COLS)
    banned.add(target_col)
    banned.update(["unified_row_id"])

    descriptor_cols = [c for c in numeric_cols if "_descriptor_" in c]
    ratio_cols = [
        c for c in numeric_cols
        if c in {
            "hba_mole_fraction_from_ratio", "hbd_mole_fraction_from_ratio",
            "hba_to_hbd_ratio_numeric", "hbd_to_hba_ratio_numeric",
            "is_special_ratio_format",
        }
    ]
    temp_cols = [TEMP_COL] if TEMP_COL in numeric_cols else []

    if feature_mode == "descriptors_ratio_temp":
        feats = descriptor_cols + ratio_cols + temp_cols
    elif feature_mode == "full_safe":
        feats = []
        for c in numeric_cols:
            if c in banned:
                continue
            if c.startswith("_"):
                continue
            if any(c.startswith(prefix) for prefix in ["source_"]):
                continue
            if any(token in c for token in ["target", "property"]):
                continue
            # Keep numeric quality flags and descriptors, but remove obvious row/source IDs.
            if c in ["year"]:
                continue
            feats.append(c)
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    # Preserve order, de-duplicate, remove banned.
    seen = set()
    clean = []
    for c in feats:
        if c in banned or c in seen:
            continue
        seen.add(c)
        clean.append(c)
    return clean


def get_cv(protocol: str, df_task: pd.DataFrame, n_splits: int, seed: int):
    n = len(df_task)
    if protocol == "random":
        k = min(n_splits, n)
        return KFold(n_splits=k, shuffle=True, random_state=seed), None

    if protocol == "pair_group":
        groups = df_task["_pair_group"].values
    elif protocol == "pair_ratio_group":
        groups = df_task["_pair_ratio_group"].values
    elif protocol == "leave_hba_out":
        groups = df_task["_hba_group"].values
    elif protocol == "leave_hbd_out":
        groups = df_task["_hbd_group"].values
    else:
        raise ValueError(f"Unknown protocol={protocol}")

    unique = pd.Series(groups).nunique()
    if protocol.startswith("leave_"):
        return LeaveOneGroupOut(), groups
    k = min(n_splits, unique)
    if k < 2:
        raise ValueError(f"Not enough groups for {protocol}: {unique}")
    return GroupKFold(n_splits=k), groups


def model_factory(name: str, seed: int):
    if name == "ExtraTrees":
        estimator = ExtraTreesRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features=1.0,
        )
        # Tree ensembles do not need scaling.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
    if name == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=400,
            learning_rate=0.04,
            l2_regularization=0.01,
            max_leaf_nodes=31,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
    raise ValueError(f"Unknown model={name}")


def safe_metrics(y_true, y_pred) -> Dict[str, float]:
    out = {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return out


def run_permutation_for_fold(pipe, X_val, y_val, feature_names, seed, n_repeats):
    result = permutation_importance(
        pipe, X_val, y_val,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    rows = []
    for f, mean, std in zip(feature_names, result.importances_mean, result.importances_std):
        rows.append({
            "feature": f,
            "perm_importance_mean": float(mean),
            "perm_importance_std": float(std),
        })
    return rows


def run_optional_shap(model_name: str, fitted_pipe, X_sample: pd.DataFrame, feature_names: List[str]):
    """Return SHAP mean absolute values if shap is installed and compatible."""
    try:
        import shap  # type: ignore
    except Exception:
        return None, "SHAP_NOT_INSTALLED"

    try:
        imputer = fitted_pipe.named_steps["imputer"]
        model = fitted_pipe.named_steps["model"]
        X_imp = imputer.transform(X_sample)
        X_imp_df = pd.DataFrame(X_imp, columns=feature_names)

        # TreeExplainer usually supports ExtraTrees. HGB support may depend on SHAP version.
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_imp_df)
        if isinstance(values, list):
            values = values[0]
        mean_abs = np.abs(values).mean(axis=0)
        rows = [{"feature": f, "shap_mean_abs": float(v)} for f, v in zip(feature_names, mean_abs)]
        return rows, "OK"
    except Exception as e:
        return None, f"SHAP_FAILED: {type(e).__name__}: {e}"


def save_top_plot(df_top: pd.DataFrame, out_path: Path, title: str, value_col: str):
    if df_top.empty:
        return
    plot_df = df_top.sort_values(value_col, ascending=True)
    plt.figure(figsize=(8, max(4, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--outdir", default="interpretability_outputs")
    parser.add_argument("--feature-mode", default="descriptors_ratio_temp",
                        choices=["descriptors_ratio_temp", "full_safe"])
    parser.add_argument("--protocols", nargs="+",
                        default=["pair_group", "pair_ratio_group", "leave_hba_out", "leave_hbd_out"])
    parser.add_argument("--models", nargs="+",
                        default=["ExtraTrees", "HistGradientBoosting"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-leave-groups", type=int, default=30,
                        help="For leave-one-component protocols, evaluate at most this many largest groups to control runtime.")
    parser.add_argument("--min-task-n", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-sample", type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df = add_engineered_ratio_features(df)
    df = build_pair_columns(df)

    perm_rows = []
    shap_rows = []
    leak_rows = []
    metric_rows = []
    manifest = {
        "data": args.data,
        "n_rows_total": int(len(df)),
        "feature_mode": args.feature_mode,
        "protocols": args.protocols,
        "models": args.models,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "viscosity_target_transform": "log10",
        "notes": [
            "Permutation importance is computed on validation folds only.",
            "All target/property columns are banned from X.",
            "SHAP is optional and executed only with --run-shap.",
        ],
    }

    for prop, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue

        df_task = df.loc[df[target_col].notna()].copy()
        if prop == "viscosity":
            df_task = df_task.loc[df_task[target_col] > 0].copy()
            df_task["_target_y"] = np.log10(df_task[target_col].astype(float))
            target_transform = "log10"
        else:
            df_task["_target_y"] = df_task[target_col].astype(float)
            target_transform = "none"

        if len(df_task) < args.min_task_n:
            continue

        features = infer_feature_columns(df_task, target_col, args.feature_mode)
        features = [f for f in features if f in df_task.columns]
        X_all = df_task[features].copy()
        y_all = df_task["_target_y"].values

        target_in_X = target_col in features
        any_property_in_X = any(c in features for c in ALL_TARGET_COLS)
        leak_rows.append({
            "property": prop,
            "target_col": target_col,
            "n": int(len(df_task)),
            "n_features": int(len(features)),
            "target_in_X": bool(target_in_X),
            "any_property_col_in_X": bool(any_property_in_X),
            "property_cols_in_X": ";".join([c for c in ALL_TARGET_COLS if c in features]),
            "feature_mode": args.feature_mode,
        })

        if target_in_X or any_property_in_X:
            raise RuntimeError(f"Leakage detected for {prop}: target/property column appears in X.")

        for protocol in args.protocols:
            try:
                cv, groups = get_cv(protocol, df_task, args.n_splits, args.seed)
            except Exception as e:
                print(f"[SKIP] {prop} {protocol}: {e}")
                continue

            splits = list(cv.split(X_all, y_all, groups=groups))
            # Control runtime for LeaveOneGroupOut by evaluating largest groups first.
            if protocol.startswith("leave_") and len(splits) > args.max_leave_groups:
                sizes = []
                for i, (_, test_idx) in enumerate(splits):
                    sizes.append((i, len(test_idx)))
                keep = {i for i, _ in sorted(sizes, key=lambda x: x[1], reverse=True)[:args.max_leave_groups]}
                splits = [s for i, s in enumerate(splits) if i in keep]

            for model_name in args.models:
                fold_perm = []
                y_true_all, y_pred_all = [], []

                for fold_id, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[test_idx]
                    y_train, y_val = y_all[train_idx], y_all[test_idx]

                    if len(np.unique(y_train)) < 2 or len(y_val) < 3:
                        continue

                    pipe = model_factory(model_name, args.seed + fold_id)
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_val)
                    y_true_all.extend(y_val.tolist())
                    y_pred_all.extend(pred.tolist())

                    fold_metrics = safe_metrics(y_val, pred)
                    metric_rows.append({
                        "property": prop,
                        "target_col": target_col,
                        "target_transform": target_transform,
                        "protocol": protocol,
                        "model": model_name,
                        "feature_mode": args.feature_mode,
                        "fold": fold_id,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        **fold_metrics,
                    })

                    for row in run_permutation_for_fold(
                        pipe, X_val, y_val, features,
                        seed=args.seed + fold_id,
                        n_repeats=args.n_repeats
                    ):
                        row.update({
                            "property": prop,
                            "target_col": target_col,
                            "target_transform": target_transform,
                            "protocol": protocol,
                            "model": model_name,
                            "feature_mode": args.feature_mode,
                            "fold": fold_id,
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                        })
                        perm_rows.append(row)

                if len(y_true_all) >= 5:
                    overall = safe_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))
                    metric_rows.append({
                        "property": prop,
                        "target_col": target_col,
                        "target_transform": target_transform,
                        "protocol": protocol,
                        "model": model_name,
                        "feature_mode": args.feature_mode,
                        "fold": "pooled",
                        "n_train": np.nan,
                        "n_test": int(len(y_true_all)),
                        **overall,
                    })

                # Optional SHAP on a final model trained on all available rows for the given property.
                if args.run_shap:
                    try:
                        final_pipe = model_factory(model_name, args.seed)
                        final_pipe.fit(X_all, y_all)
                        sample_n = min(args.shap_sample, len(X_all))
                        X_sample = X_all.sample(sample_n, random_state=args.seed)
                        shap_imp, shap_status = run_optional_shap(model_name, final_pipe, X_sample, features)
                        if shap_imp is not None:
                            for row in shap_imp:
                                row.update({
                                    "property": prop,
                                    "target_col": target_col,
                                    "target_transform": target_transform,
                                    "protocol": "trained_all_rows_for_shap",
                                    "model": model_name,
                                    "feature_mode": args.feature_mode,
                                    "n_sample": int(sample_n),
                                    "shap_status": shap_status,
                                })
                                shap_rows.append(row)
                        else:
                            shap_rows.append({
                                "property": prop,
                                "target_col": target_col,
                                "target_transform": target_transform,
                                "protocol": "trained_all_rows_for_shap",
                                "model": model_name,
                                "feature_mode": args.feature_mode,
                                "feature": "__SHAP_STATUS__",
                                "shap_mean_abs": np.nan,
                                "n_sample": int(sample_n),
                                "shap_status": shap_status,
                            })
                    except Exception as e:
                        shap_rows.append({
                            "property": prop,
                            "target_col": target_col,
                            "target_transform": target_transform,
                            "protocol": "trained_all_rows_for_shap",
                            "model": model_name,
                            "feature_mode": args.feature_mode,
                            "feature": "__SHAP_STATUS__",
                            "shap_mean_abs": np.nan,
                            "n_sample": np.nan,
                            "shap_status": f"FINAL_MODEL_FAILED: {type(e).__name__}: {e}",
                        })

    perm_df = pd.DataFrame(perm_rows)
    leak_df = pd.DataFrame(leak_rows)
    metrics_df = pd.DataFrame(metric_rows)
    shap_df = pd.DataFrame(shap_rows)

    leak_df.to_csv(outdir / "leakage_audit_interpretability.csv", index=False)
    metrics_df.to_csv(outdir / "interpretability_fold_metrics.csv", index=False)

    if not perm_df.empty:
        perm_df.to_csv(outdir / "permutation_importance_long.csv", index=False)
        summary = (
            perm_df.groupby(["property", "target_col", "target_transform", "protocol", "model", "feature_mode", "feature"], as_index=False)
            .agg(
                perm_importance_mean=("perm_importance_mean", "mean"),
                perm_importance_std_across_folds=("perm_importance_mean", "std"),
                n_folds=("fold", "nunique"),
                mean_n_test=("n_test", "mean"),
            )
            .sort_values(["property", "protocol", "model", "perm_importance_mean"], ascending=[True, True, True, False])
        )
        summary.to_csv(outdir / "permutation_importance_summary.csv", index=False)

        top = (
            summary.sort_values("perm_importance_mean", ascending=False)
            .groupby(["property", "protocol", "model"], as_index=False)
            .head(args.top_k)
        )
        top.to_csv(outdir / "top_features_by_property_protocol_model.csv", index=False)

        for (prop, protocol, model), g in top.groupby(["property", "protocol", "model"]):
            safe = lambda s: str(s).replace("/", "_").replace(" ", "_")
            path = figdir / f"top_permutation_{safe(prop)}_{safe(protocol)}_{safe(model)}.png"
            save_top_plot(
                g.head(args.top_k),
                path,
                title=f"{prop} | {protocol} | {model}",
                value_col="perm_importance_mean",
            )

    if not shap_df.empty:
        shap_df.to_csv(outdir / "shap_importance_long.csv", index=False)
        valid = shap_df.loc[shap_df["feature"] != "__SHAP_STATUS__"].copy()
        if not valid.empty:
            shap_summary = (
                valid.groupby(["property", "target_col", "target_transform", "model", "feature_mode", "feature"], as_index=False)
                .agg(shap_mean_abs=("shap_mean_abs", "mean"), n=("shap_mean_abs", "size"))
                .sort_values(["property", "model", "shap_mean_abs"], ascending=[True, True, False])
            )
            shap_summary.to_csv(outdir / "shap_importance_summary.csv", index=False)

    manifest["outputs"] = sorted([p.name for p in outdir.glob("*.csv")])
    with open(outdir / "interpretability_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print(" - permutation_importance_summary.csv")
    print(" - top_features_by_property_protocol_model.csv")
    print(" - interpretability_fold_metrics.csv")
    print(" - leakage_audit_interpretability.csv")
    if args.run_shap:
        print(" - shap_importance_summary.csv if SHAP succeeded")


if __name__ == "__main__":
    main()
