
# ============================================================
# Line 1 DES ML Diagnostic/Audit Pipeline
# Project: AI_DES formation model — Line 1 manuscript
#
# Purpose:
#   Forensically compare old-like, leakage-corrected, strict group-aware,
#   and extrapolative validation protocols on the frozen GOLD dataset.
#
# Inputs:
#   --input: frozen pairwise/ratio feature dataset preferred.
#            If only descriptor-ready dataset is provided, pipeline still runs
#            but pairwise/ratio ablations may be reduced.
#
# Main outputs:
#   diagnostic_audit_outputs/
#     audit_feature_leakage.csv
#     diagnostic_metrics_long.csv
#     diagnostic_metrics_summary.csv
#     dataset_bias_summary.csv
#     component_frequency_hba.csv
#     component_frequency_hbd.csv
#     viscosity_diagnostics.csv
#     top_feature_importances.csv
#     run_manifest.json
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42

TARGETS = {
    "density": "density_g_cm3",
    "viscosity": "viscosity_mpa_s",
    "conductivity": "conductivity_ms_cm",
    "surface_tension": "surface_tension_mn_m",
    "refractive_index": "refractive_index",
}
ALL_TARGET_COLS = list(TARGETS.values())
POSITIVE_SKEWED = {"viscosity", "conductivity"}

META_DROP = {
    "unified_row_id", "source_corpus", "source_period", "source_origin", "source_filename",
    "source_schema_variant", "entry_id_local", "article_title", "journal", "year", "doi",
    "reference_ids_raw", "reference_titles_raw", "reference_dois_raw", "traceability_note",
    "component_1_name_raw", "component_2_name_raw", "hba_name_raw", "hbd_name_raw",
    "hba_name_canonical", "hbd_name_canonical", "hba_slug_canonical", "hbd_slug_canonical",
    "hba_component_registry_id", "hbd_component_registry_id", "hba_name_resolved", "hbd_name_resolved",
    "hba_canonical_name", "hbd_canonical_name", "hba_canonical_name_resolved", "hbd_canonical_name_resolved",
    "hba_canonical_slug", "hbd_canonical_slug", "hba_preferred_role", "hbd_preferred_role",
    "hba_observed_roles", "hbd_observed_roles", "smiles_hba", "smiles_hbd", "hba_smiles", "hbd_smiles",
    "hba_smiles_mapped", "hbd_smiles_mapped", "molar_ratio_raw", "composition_label",
    "measurement_temperature_mode", "measurement_condition_note", "gold_primary_reason", "relaxed_primary_reason",
    "gold_inclusion_status", "relaxed_inclusion_status", "ratio_basis", "special_ratio_note",
}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))

def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "n_test": int(len(y_true)),
    }

def make_pair_group(df):
    hba = df.get("hba_name_resolved", pd.Series(["NA"]*len(df), index=df.index)).fillna("NA").astype(str)
    hbd = df.get("hbd_name_resolved", pd.Series(["NA"]*len(df), index=df.index)).fillna("NA").astype(str)
    return hba + " || " + hbd

def make_pair_ratio_group(df):
    base = make_pair_group(df)
    ratio = df.get("molar_ratio_raw", pd.Series(["NA"]*len(df), index=df.index)).fillna("NA").astype(str)
    return base + " || " + ratio

def make_component_group(df, component):
    col = "hba_name_resolved" if component == "hba" else "hbd_name_resolved"
    return df.get(col, pd.Series(["NA"]*len(df), index=df.index)).fillna("NA").astype(str)

def numeric_nonempty(df, cols):
    keep = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            if df[c].notna().any():
                keep.append(c)
    return keep

def infer_feature_columns(df, target_col, protocol, feature_set):
    numeric_cols = numeric_nonempty(df, list(df.columns))

    # old-like notebook behavior: drops all target columns then discards current target from drop list,
    # so current target remains eligible as a numeric feature.
    if protocol == "A_old_like_potentially_leaky":
        drop = set(META_DROP).union(ALL_TARGET_COLS)
        drop.discard(target_col)
    else:
        drop = set(META_DROP).union(ALL_TARGET_COLS)

    eligible = [c for c in numeric_cols if c not in drop]

    temp_cols = [c for c in eligible if c in {"measurement_temperature_c", "temperature_c"} or "temperature" in c.lower()]
    ratio_cols = [c for c in eligible if c.startswith("ratio_") or "ratio_" in c]
    descriptor_cols = [c for c in eligible if ("descriptor_" in c) or c in {"tm_c"}]
    pair_cols = [c for c in eligible if c.startswith("pair_") or c.startswith("ratio_weighted_")]

    # Ensure no leakage in named ablations under corrected protocols.
    if feature_set == "temperature_only":
        cols = temp_cols
    elif feature_set == "ratio_only":
        cols = ratio_cols
    elif feature_set == "descriptors_only":
        cols = descriptor_cols
    elif feature_set == "descriptors_plus_ratio":
        cols = sorted(set(descriptor_cols + ratio_cols + temp_cols))
    elif feature_set == "full":
        cols = eligible
    else:
        raise ValueError(feature_set)

    return [c for c in cols if c in df.columns]

def make_model(model_name, task_name):
    if model_name == "dummy_mean":
        model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("model", DummyRegressor(strategy="mean"))])
    elif model_name == "ridge":
        model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler()),
                          ("model", Ridge(alpha=1.0))])
    elif model_name == "extra_trees":
        model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("model", ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2,
                                                        random_state=SEED, n_jobs=-1))])
    elif model_name == "hist_gradient_boosting":
        model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                                                  l2_regularization=0.01,
                                                                  random_state=SEED))])
    else:
        raise ValueError(model_name)

    if task_name in POSITIVE_SKEWED and model_name not in {"dummy_mean"}:
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1,
                                          check_inverse=False)
    return model

def choose_splits(df_task, protocol, n_splits):
    n = len(df_task)
    if protocol == "random_row":
        kf = KFold(n_splits=min(n_splits, max(2, n)), shuffle=True, random_state=SEED)
        return [(tr, te, "random_row") for tr, te in kf.split(df_task)]
    if protocol in {"A_old_like_potentially_leaky", "B_leakage_corrected_pair_ratio_group"}:
        groups = make_pair_ratio_group(df_task)
        group_label = "hba+hbd+raw_ratio"
    elif protocol == "C_strict_pair_group":
        groups = make_pair_group(df_task)
        group_label = "hba+hbd"
    elif protocol == "D_leave_one_hba_out":
        groups = make_component_group(df_task, "hba")
        group_label = "hba"
    elif protocol == "D_leave_one_hbd_out":
        groups = make_component_group(df_task, "hbd")
        group_label = "hbd"
    else:
        raise ValueError(protocol)

    n_groups = groups.nunique()
    if n_groups < 2:
        return []
    k = min(n_splits, n_groups)
    splitter = GroupKFold(n_splits=k)
    return [(tr, te, group_label) for tr, te in splitter.split(df_task, groups=groups)]

def run_cv(df, outdir, n_splits=5, fast=False):
    protocols = [
        "A_old_like_potentially_leaky",
        "B_leakage_corrected_pair_ratio_group",
        "C_strict_pair_group",
        "D_leave_one_hba_out",
        "D_leave_one_hbd_out",
    ]
    feature_sets = ["temperature_only", "ratio_only", "descriptors_only", "descriptors_plus_ratio", "full"]
    models = ["dummy_mean", "ridge", "extra_trees", "hist_gradient_boosting"]
    if fast:
        protocols = protocols[:3]
        models = ["dummy_mean", "ridge", "extra_trees"]
        feature_sets = ["temperature_only", "descriptors_only", "full"]
        n_splits = min(n_splits, 3)

    rows, audit_rows, viscosity_rows, fi_rows = [], [], [], []

    for task_name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        df_task = df[df[target_col].notna()].copy()
        y_all = pd.to_numeric(df_task[target_col], errors="coerce")
        df_task = df_task[y_all.notna()].copy()
        y_all = y_all[y_all.notna()].astype(float)

        for protocol in protocols:
            splits = choose_splits(df_task, protocol, n_splits)
            for feature_set in feature_sets:
                feat_cols = infer_feature_columns(df_task, target_col, protocol, feature_set)
                target_in_X = target_col in feat_cols
                property_cols_in_X = [c for c in ALL_TARGET_COLS if c in feat_cols]
                audit_rows.append({
                    "task": task_name, "target_col": target_col, "protocol": protocol,
                    "feature_set": feature_set, "n_rows": len(df_task), "n_features": len(feat_cols),
                    "target_in_X": bool(target_in_X),
                    "property_cols_in_X": ";".join(property_cols_in_X),
                    "first_30_features": ";".join(feat_cols[:30]),
                })
                if not feat_cols or not splits:
                    continue

                X_all = df_task[feat_cols].copy()
                for model_name in models:
                    fold_preds = []
                    for fold, (tr, te, group_label) in enumerate(splits, start=1):
                        model = make_model(model_name, task_name)
                        X_train, X_test = X_all.iloc[tr], X_all.iloc[te]
                        y_train, y_test = y_all.iloc[tr], y_all.iloc[te]
                        try:
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                        except Exception as e:
                            rows.append({
                                "task": task_name, "target_col": target_col, "protocol": protocol,
                                "group_definition": group_label, "feature_set": feature_set,
                                "model": model_name, "fold": fold, "status": f"ERROR: {e}",
                                "n_train": len(tr), "n_test": len(te), "n_features": len(feat_cols),
                                "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                            })
                            continue
                        m = metrics(y_test, pred)
                        row = {
                            "task": task_name, "target_col": target_col, "protocol": protocol,
                            "group_definition": group_label, "feature_set": feature_set,
                            "model": model_name, "fold": fold, "status": "ok",
                            "n_train": len(tr), "n_test": len(te), "n_features": len(feat_cols),
                            **m,
                        }
                        rows.append(row)

                        if task_name == "viscosity" and protocol != "A_old_like_potentially_leaky" and feature_set == "full" and model_name == "extra_trees":
                            tmp = df_task.iloc[te].copy()
                            tmp["y_true"] = y_test.values
                            tmp["y_pred"] = pred
                            tmp["residual"] = tmp["y_pred"] - tmp["y_true"]
                            tmp["abs_error"] = np.abs(tmp["residual"])
                            tmp["fold"] = fold
                            tmp["protocol"] = protocol
                            viscosity_rows.append(tmp[[c for c in ["protocol","fold","measurement_temperature_c","molar_ratio_raw","ratio_hbd_to_hba","y_true","y_pred","residual","abs_error"] if c in tmp.columns]])

                    # permutation importance only for corrected full ExtraTrees, one train/test group split per task/protocol
                    if model_name == "extra_trees" and feature_set == "full" and protocol in {"B_leakage_corrected_pair_ratio_group", "C_strict_pair_group"} and splits:
                        tr, te, group_label = splits[0]
                        try:
                            model = make_model(model_name, task_name)
                            model.fit(X_all.iloc[tr], y_all.iloc[tr])
                            pi = permutation_importance(model, X_all.iloc[te], y_all.iloc[te],
                                                        n_repeats=5, random_state=SEED,
                                                        scoring="neg_mean_absolute_error")
                            order = np.argsort(pi.importances_mean)[::-1][:25]
                            for rank, idx in enumerate(order, start=1):
                                fi_rows.append({
                                    "task": task_name, "protocol": protocol, "model": model_name,
                                    "rank": rank, "feature": feat_cols[idx],
                                    "permutation_importance_mean_delta_mae": float(pi.importances_mean[idx]),
                                    "permutation_importance_std": float(pi.importances_std[idx]),
                                })
                        except Exception as e:
                            fi_rows.append({"task": task_name, "protocol": protocol, "model": model_name,
                                            "rank": None, "feature": f"ERROR: {e}",
                                            "permutation_importance_mean_delta_mae": np.nan,
                                            "permutation_importance_std": np.nan})

    metrics_long = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    vis = pd.concat(viscosity_rows, ignore_index=True) if viscosity_rows else pd.DataFrame()
    fi = pd.DataFrame(fi_rows)

    metrics_long.to_csv(outdir / "diagnostic_metrics_long.csv", index=False)
    audit.to_csv(outdir / "audit_feature_leakage.csv", index=False)
    if not vis.empty:
        vis.to_csv(outdir / "viscosity_diagnostics.csv", index=False)
    if not fi.empty:
        fi.to_csv(outdir / "top_feature_importances.csv", index=False)

    if not metrics_long.empty:
        summary = (metrics_long[metrics_long["status"].eq("ok")]
                   .groupby(["task","target_col","protocol","group_definition","feature_set","model"], as_index=False)
                   .agg(n_folds=("fold","count"),
                        n_test_total=("n_test","sum"),
                        n_features=("n_features","median"),
                        mae_mean=("mae","mean"), mae_std=("mae","std"),
                        rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
                        r2_mean=("r2","mean"), r2_std=("r2","std")))
        # add deltas versus dummy_mean and temperature-only ExtraTrees within same task/protocol
        summary["delta_r2_vs_dummy_mean"] = np.nan
        summary["delta_mae_vs_dummy_mean"] = np.nan
        for idx, r in summary.iterrows():
            base = summary[(summary.task==r.task)&(summary.protocol==r.protocol)&
                           (summary.feature_set==r.feature_set)&(summary.model=="dummy_mean")]
            if not base.empty:
                summary.loc[idx, "delta_r2_vs_dummy_mean"] = r.r2_mean - float(base.iloc[0].r2_mean)
                summary.loc[idx, "delta_mae_vs_dummy_mean"] = float(base.iloc[0].mae_mean) - r.mae_mean
            temp = summary[(summary.task==r.task)&(summary.protocol==r.protocol)&
                           (summary.feature_set=="temperature_only")&(summary.model==r.model)]
            if not temp.empty:
                summary.loc[idx, "delta_r2_vs_temperature_only_same_model"] = r.r2_mean - float(temp.iloc[0].r2_mean)
                summary.loc[idx, "delta_mae_vs_temperature_only_same_model"] = float(temp.iloc[0].mae_mean) - r.mae_mean
        summary.to_csv(outdir / "diagnostic_metrics_summary.csv", index=False)

def dataset_bias(df, outdir):
    rows = []
    for task, target in TARGETS.items():
        if target not in df.columns:
            continue
        d = df[df[target].notna()].copy()
        rows.append({
            "task": task,
            "target_col": target,
            "n": len(d),
            "n_hba": d["hba_name_resolved"].nunique() if "hba_name_resolved" in d else np.nan,
            "n_hbd": d["hbd_name_resolved"].nunique() if "hbd_name_resolved" in d else np.nan,
            "n_pairs": make_pair_group(d).nunique(),
            "temperature_min": pd.to_numeric(d.get("measurement_temperature_c"), errors="coerce").min() if "measurement_temperature_c" in d else np.nan,
            "temperature_median": pd.to_numeric(d.get("measurement_temperature_c"), errors="coerce").median() if "measurement_temperature_c" in d else np.nan,
            "temperature_max": pd.to_numeric(d.get("measurement_temperature_c"), errors="coerce").max() if "measurement_temperature_c" in d else np.nan,
            "target_min": pd.to_numeric(d[target], errors="coerce").min(),
            "target_median": pd.to_numeric(d[target], errors="coerce").median(),
            "target_max": pd.to_numeric(d[target], errors="coerce").max(),
        })
    pd.DataFrame(rows).to_csv(outdir / "dataset_bias_summary.csv", index=False)
    if "hba_name_resolved" in df:
        df["hba_name_resolved"].value_counts(dropna=False).rename_axis("hba").reset_index(name="count").to_csv(outdir / "component_frequency_hba.csv", index=False)
    if "hbd_name_resolved" in df:
        df["hbd_name_resolved"].value_counts(dropna=False).rename_axis("hbd").reset_index(name="count").to_csv(outdir / "component_frequency_hbd.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="diagnostic_audit_outputs")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="Run a lighter 3-protocol/3-model audit for quick diagnosis.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    dataset_bias(df, outdir)
    run_cv(df, outdir, n_splits=args.n_splits, fast=args.fast)

    manifest = {
        "input": str(Path(args.input).resolve()),
        "shape": list(df.shape),
        "targets": TARGETS,
        "seed": SEED,
        "n_splits": args.n_splits,
        "fast": args.fast,
        "notes": [
            "Protocol A intentionally reproduces the old-like feature-selection behavior and may include the current target in X.",
            "Protocols B-D remove all target/property columns from X.",
            "D protocols are component-extrapolative approximations using GroupKFold by HBA or HBD."
        ],
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Outputs written to: {outdir}")

if __name__ == "__main__":
    main()
