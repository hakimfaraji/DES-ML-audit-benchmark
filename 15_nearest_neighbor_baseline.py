#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest-neighbor / kNN baseline for leakage-aware DES property prediction.

Purpose
-------
This script tests whether the audited DES ML models are doing substantially more
than similarity-based interpolation in the available descriptor space. It evaluates
simple nearest-neighbor baselines under the same leakage-safe validation logic used
in the manuscript:

  Protocol B: group by HBA-HBD-ratio composition
  Protocol C: group by HBA-HBD pair
  Protocol D: leave-HBA-out and leave-HBD-out extrapolation

Input
-----
Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Main outputs
------------
knn_baseline_outputs/
  knn_baseline_metrics.csv
  knn_baseline_predictions.csv
  knn_baseline_summary.csv
  FigureS_NN_baseline_R2.png/pdf
  FigureS_NN_baseline_MAE.png/pdf
  nearest_neighbor_run_config.json

Recommended Colab command
-------------------------
!python run_nearest_neighbor_baseline.py \
  --input Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir knn_baseline_outputs

Notes
-----
- Leakage-safe feature matrix: numeric descriptor/condition/composition features only.
- Target and all non-target property columns are removed from X.
- String metadata, source columns, and identifiers are not used as features.
- kNN baselines are evaluated after StandardScaler fit only on the training fold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROPERTY_TARGETS: Dict[str, str] = {
    "Density": "density_g_cm3",
    "Viscosity": "viscosity_mpa_s",
    "Conductivity": "conductivity_ms_cm",
    "Surface tension": "surface_tension_mn_m",
    "Refractive index": "refractive_index",
}

PROPERTY_COLS = set(PROPERTY_TARGETS.values())

# Columns that must never enter X even if numeric.
ALWAYS_EXCLUDE_PATTERNS = [
    r"^unified_row_id$",
    r"^entry_id_local$",
    r"^year$",  # bibliographic year, not a physicochemical feature
    r"^tm_c$",  # melting point may be inconsistently available and can behave like source leakage
    r"inclusion_status$",
    r"manual_review$",
    r"validation_[xy]$",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nearest-neighbor baseline for leakage-aware DES ML.")
    parser.add_argument("--input", required=True, help="Path to Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--outdir", default="knn_baseline_outputs", help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits for Protocol B/C")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used only for deterministic ordering/tie handling")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum non-null samples needed for a property")
    parser.add_argument("--min-train", type=int, default=30, help="Minimum train samples required for a fold")
    parser.add_argument("--min-test", type=int, default=5, help="Minimum test samples required for a fold/component holdout")
    parser.add_argument("--max-d-folds-per-property", type=int, default=10,
                        help="Maximum leave-component folds per property per D protocol; largest test groups are used")
    parser.add_argument("--k-values", default="1,5",
                        help="Comma-separated k values for KNeighborsRegressor. k=1 is nearest-neighbor baseline.")
    parser.add_argument("--weights", default="uniform",
                        help="Comma-separated weighting modes for kNN: uniform,distance")
    parser.add_argument("--include-log-viscosity", action="store_true",
                        help="Also evaluate viscosity on log10 target. Raw viscosity is always evaluated.")
    return parser.parse_args()


def safe_slug(s: object) -> str:
    if pd.isna(s):
        return "NA"
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_+.-]+", "", s)
    return s or "NA"


def parse_ratio_value(x: object) -> float:
    """Parse HBD/HBA molar ratio from common formats: '1:2', '1/2', '2', '1 : 4', etc."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    if not s:
        return np.nan
    s = s.replace("−", "-").replace("–", "-")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan
    vals = [float(v) for v in nums]
    # Convention in this project: ratio range is HBD/HBA. For '1:2', return 2/1.
    if (":" in s or "/" in s) and len(vals) >= 2 and vals[0] != 0:
        return vals[1] / vals[0]
    return vals[0]


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "measurement_temperature_k" not in out.columns:
        if "measurement_temperature_c" in out.columns:
            out["measurement_temperature_k"] = pd.to_numeric(out["measurement_temperature_c"], errors="coerce") + 273.15
    if "parsed_hbd_hba_molar_ratio" not in out.columns:
        if "molar_ratio_raw" in out.columns:
            out["parsed_hbd_hba_molar_ratio"] = out["molar_ratio_raw"].apply(parse_ratio_value)
    return out


def build_groups(df: pd.DataFrame, protocol: str) -> pd.Series:
    hba_col = first_existing_column(df, ["hba_name_resolved", "hba_name_canonical", "hba_name_raw"])
    hbd_col = first_existing_column(df, ["hbd_name_resolved", "hbd_name_canonical", "hbd_name_raw"])
    if hba_col is None or hbd_col is None:
        raise ValueError("Could not find HBA/HBD name columns for grouping.")
    hba = df[hba_col].map(safe_slug)
    hbd = df[hbd_col].map(safe_slug)

    if protocol == "B_pair_ratio":
        ratio = df.get("parsed_hbd_hba_molar_ratio", pd.Series(np.nan, index=df.index))
        # Rounded ratio prevents tiny floating parsing differences from producing false groups.
        ratio_s = pd.to_numeric(ratio, errors="coerce").round(6).astype(str)
        return hba + "||" + hbd + "||r=" + ratio_s
    if protocol == "C_pair":
        return hba + "||" + hbd
    if protocol == "D_leave_HBA":
        return hba
    if protocol == "D_leave_HBD":
        return hbd
    raise ValueError(f"Unknown protocol: {protocol}")


def should_exclude_column(col: str, target_col: str) -> bool:
    if col in PROPERTY_COLS:
        return True
    if col == target_col:
        return True
    for pat in ALWAYS_EXCLUDE_PATTERNS:
        if re.search(pat, col):
            return True
    return False


def build_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cols = []
    for c in numeric_cols:
        if should_exclude_column(c, target_col):
            continue
        cols.append(c)
    # Strong safety audit: remove any suspicious column containing property names.
    forbidden_tokens = ["density", "viscosity", "conductivity", "surface_tension", "refractive_index"]
    safe_cols = []
    for c in cols:
        cl = c.lower()
        if any(tok in cl for tok in forbidden_tokens):
            # allow non-target physicochemical condition only if it is clearly not a property target
            continue
        safe_cols.append(c)
    # Retain explicitly engineered ratio/temp condition columns if numeric and not accidentally filtered.
    for c in ["parsed_hbd_hba_molar_ratio", "measurement_temperature_k", "measurement_temperature_c"]:
        if c in df.columns and c not in safe_cols and c != target_col and c not in PROPERTY_COLS:
            if pd.api.types.is_numeric_dtype(df[c]):
                safe_cols.append(c)
    # Drop all-NaN or constant columns later property-specific.
    return list(dict.fromkeys(safe_cols))


def clean_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep].astype(float)


def make_b_c_splits(groups: pd.Series, n_splits: int, min_train: int, min_test: int) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    n_groups = groups.nunique(dropna=False)
    if n_groups < 2:
        return []
    actual_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=actual_splits)
    dummy_X = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))
    splits = []
    for i, (tr, te) in enumerate(gkf.split(dummy_X, dummy_y, groups=groups), start=1):
        if len(tr) >= min_train and len(te) >= min_test:
            splits.append((f"fold_{i}", tr, te))
    return splits


def make_d_splits(groups: pd.Series, min_train: int, min_test: int, max_folds: int) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    counts = groups.value_counts(dropna=False)
    eligible = counts[counts >= min_test].sort_values(ascending=False)
    if max_folds and len(eligible) > max_folds:
        eligible = eligible.iloc[:max_folds]
    splits = []
    group_values = groups.to_numpy()
    for g, n in eligible.items():
        te = np.where(group_values == g)[0]
        tr = np.where(group_values != g)[0]
        if len(tr) >= min_train and len(te) >= min_test:
            splits.append((f"leave_{str(g)[:80]}", tr, te))
    return splits


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "n_test": int(len(y_true)),
        "r2": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
    }
    if len(y_true) == 0:
        return out
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) >= 2 and np.nanstd(y_true) > 0:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def evaluate_one_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    k: Optional[int] = None,
    weights: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(dtype=float), y.iloc[test_idx].to_numpy(dtype=float)

    if model_name == "DummyMean":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", DummyRegressor(strategy="mean")),
        ])
    else:
        kk = int(k or 1)
        kk = max(1, min(kk, len(train_idx)))
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(n_neighbors=kk, weights=weights or "uniform", metric="minkowski", p=2, n_jobs=-1)),
        ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, metric_dict(y_test, pred)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    group_cols = ["property", "target_variant", "protocol", "baseline", "k", "weights"]
    rows = []
    for keys, g in metrics.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        d.update({
            "n_folds": int(g["fold_id"].nunique()),
            "total_test_predictions": int(g["n_test"].sum()),
            "r2_mean": float(g["r2"].mean(skipna=True)),
            "r2_sd": float(g["r2"].std(skipna=True, ddof=1)) if g["r2"].notna().sum() > 1 else np.nan,
            "mae_mean": float(g["mae"].mean(skipna=True)),
            "mae_sd": float(g["mae"].std(skipna=True, ddof=1)) if g["mae"].notna().sum() > 1 else np.nan,
            "rmse_mean": float(g["rmse"].mean(skipna=True)),
            "rmse_sd": float(g["rmse"].std(skipna=True, ddof=1)) if g["rmse"].notna().sum() > 1 else np.nan,
        })
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["property", "protocol", "baseline", "k", "weights"])


def plot_summary(summary: pd.DataFrame, outdir: Path, metric: str = "r2") -> None:
    if summary.empty:
        return
    # Show compact model set: DummyMean, NN-1, kNN-5 uniform if available.
    def label(row):
        if row["baseline"] == "DummyMean":
            return "Dummy mean"
        if int(row["k"]) == 1:
            return "1-NN"
        return f"{int(row['k'])}-NN {row['weights']}"

    tmp = summary.copy()
    tmp["model_label"] = tmp.apply(label, axis=1)
    preferred = ["Dummy mean", "1-NN", "5-NN uniform", "5-NN distance"]
    tmp = tmp[tmp["model_label"].isin(preferred)]
    if tmp.empty:
        return

    protocols = [p for p in ["B_pair_ratio", "C_pair", "D_leave_HBA", "D_leave_HBD"] if p in tmp["protocol"].unique()]
    properties = [p for p in PROPERTY_TARGETS.keys() if p in tmp["property"].unique()]
    labels = preferred

    for protocol in protocols:
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.18
        x = np.arange(len(properties))
        for j, lab in enumerate(labels):
            vals = []
            errs = []
            for prop in properties:
                row = tmp[(tmp["protocol"] == protocol) & (tmp["property"] == prop) & (tmp["model_label"] == lab)]
                if len(row):
                    vals.append(float(row.iloc[0][f"{metric}_mean"]))
                    errs.append(float(row.iloc[0].get(f"{metric}_sd", np.nan)))
                else:
                    vals.append(np.nan)
                    errs.append(np.nan)
            offset = (j - (len(labels)-1)/2) * width
            ax.bar(x + offset, vals, width, label=lab, yerr=errs, capsize=3)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(properties, rotation=25, ha="right")
        ax.set_ylabel("R²" if metric == "r2" else metric.upper())
        ax.set_title(f"Nearest-neighbor baseline comparison — {protocol}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(outdir / f"FigureS_NN_baseline_{metric.upper()}_{protocol}.{ext}", dpi=300)
        plt.close(fig)

    # Combined file for Protocol B R2 is often the compact SI figure.
    src = outdir / f"FigureS_NN_baseline_{metric.upper()}_B_pair_ratio.png"
    if src.exists():
        # no copy needed; name explicit enough
        pass


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)
    warnings.filterwarnings("ignore", category=UserWarning)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = add_condition_features(df)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    weights_values = [x.strip() for x in args.weights.split(",") if x.strip()]
    weights_values = [w for w in weights_values if w in {"uniform", "distance"}]

    all_metrics: List[Dict[str, object]] = []
    all_preds: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    protocols = ["B_pair_ratio", "C_pair", "D_leave_HBA", "D_leave_HBD"]

    for prop, target_col in PROPERTY_TARGETS.items():
        if target_col not in df.columns:
            continue
        prop_df = df[df[target_col].notna()].copy().reset_index(drop=True)
        prop_df[target_col] = pd.to_numeric(prop_df[target_col], errors="coerce")
        prop_df = prop_df[prop_df[target_col].notna()].reset_index(drop=True)
        if len(prop_df) < args.min_samples:
            continue

        variants = [("raw", prop_df[target_col].copy())]
        if prop == "Viscosity" and args.include_log_viscosity:
            valid = prop_df[target_col] > 0
            log_df = prop_df[valid].copy().reset_index(drop=True)
            variants = [("raw", prop_df[target_col].copy())]
            # Process log separately with aligned dataframe marker by replacing prop_df in loop below.

        feature_cols = build_feature_columns(prop_df, target_col)
        X_all = clean_feature_matrix(prop_df[feature_cols])
        audit_rows.append({
            "property": prop,
            "target_col": target_col,
            "n_samples": len(prop_df),
            "n_features": X_all.shape[1],
            "target_in_X": target_col in X_all.columns,
            "any_property_col_in_X": bool(any(c in PROPERTY_COLS for c in X_all.columns)),
            "feature_columns_preview": ";".join(X_all.columns[:25]),
        })
        if X_all.shape[1] == 0:
            continue

        # Standard raw target evaluation.
        eval_sets: List[Tuple[str, pd.DataFrame, pd.Series]] = [("raw", prop_df, prop_df[target_col])]
        if prop == "Viscosity" and args.include_log_viscosity:
            log_mask = prop_df[target_col] > 0
            log_prop_df = prop_df[log_mask].copy().reset_index(drop=True)
            log_feature_cols = build_feature_columns(log_prop_df, target_col)
            # Features rebuilt for log subset; X calculated in inner loop.
            eval_sets.append(("log10", log_prop_df, np.log10(log_prop_df[target_col].astype(float))))

        for target_variant, eval_df, y_series in eval_sets:
            X = clean_feature_matrix(eval_df[build_feature_columns(eval_df, target_col)])
            y = pd.Series(y_series).reset_index(drop=True)
            if len(eval_df) < args.min_samples or X.shape[1] == 0:
                continue

            for protocol in protocols:
                groups = build_groups(eval_df, protocol)
                if protocol in ["B_pair_ratio", "C_pair"]:
                    splits = make_b_c_splits(groups, args.n_splits, args.min_train, args.min_test)
                else:
                    splits = make_d_splits(groups, args.min_train, args.min_test, args.max_d_folds_per_property)
                if not splits:
                    continue

                for fold_id, train_idx, test_idx in splits:
                    # Dummy mean baseline.
                    pred, met = evaluate_one_fold(X, y, train_idx, test_idx, model_name="DummyMean")
                    row = {
                        "property": prop, "target_col": target_col, "target_variant": target_variant,
                        "protocol": protocol, "fold_id": fold_id, "baseline": "DummyMean",
                        "k": 0, "weights": "NA", "n_train": len(train_idx),
                        **met,
                    }
                    all_metrics.append(row)
                    for local_i, p in zip(test_idx, pred):
                        all_preds.append({
                            "property": prop, "target_variant": target_variant, "protocol": protocol,
                            "fold_id": fold_id, "baseline": "DummyMean", "k": 0, "weights": "NA",
                            "row_index_property_subset": int(local_i), "y_true": float(y.iloc[local_i]), "y_pred": float(p),
                            "abs_error": float(abs(y.iloc[local_i] - p)),
                        })

                    for k in k_values:
                        for weights in weights_values:
                            # For k=1, uniform and distance give identical predictions; keep uniform only.
                            if k == 1 and weights != "uniform":
                                continue
                            pred, met = evaluate_one_fold(X, y, train_idx, test_idx, model_name="KNN", k=k, weights=weights)
                            row = {
                                "property": prop, "target_col": target_col, "target_variant": target_variant,
                                "protocol": protocol, "fold_id": fold_id, "baseline": "KNN",
                                "k": k, "weights": weights, "n_train": len(train_idx),
                                **met,
                            }
                            all_metrics.append(row)
                            for local_i, p in zip(test_idx, pred):
                                all_preds.append({
                                    "property": prop, "target_variant": target_variant, "protocol": protocol,
                                    "fold_id": fold_id, "baseline": "KNN", "k": k, "weights": weights,
                                    "row_index_property_subset": int(local_i), "y_true": float(y.iloc[local_i]), "y_pred": float(p),
                                    "abs_error": float(abs(y.iloc[local_i] - p)),
                                })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.DataFrame(all_preds)
    audit_df = pd.DataFrame(audit_rows)
    summary_df = summarize_metrics(metrics_df) if not metrics_df.empty else pd.DataFrame()

    metrics_df.to_csv(outdir / "knn_baseline_metrics.csv", index=False)
    preds_df.to_csv(outdir / "knn_baseline_predictions.csv", index=False)
    summary_df.to_csv(outdir / "knn_baseline_summary.csv", index=False)
    audit_df.to_csv(outdir / "knn_feature_audit.csv", index=False)

    plot_summary(summary_df, outdir, metric="r2")
    plot_summary(summary_df, outdir, metric="mae")

    config = {
        "input": str(input_path),
        "outdir": str(outdir),
        "n_rows_input": int(len(df)),
        "properties": PROPERTY_TARGETS,
        "protocols": protocols,
        "k_values": k_values,
        "weights": weights_values,
        "n_splits": args.n_splits,
        "min_samples": args.min_samples,
        "min_train": args.min_train,
        "min_test": args.min_test,
        "max_d_folds_per_property": args.max_d_folds_per_property,
        "include_log_viscosity": bool(args.include_log_viscosity),
        "feature_policy": "numeric leakage-safe descriptors/ratio/temperature; all target and cross-property columns removed",
    }
    with open(outdir / "nearest_neighbor_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("[DONE] Nearest-neighbor baseline complete.")
    print(f"Outputs written to: {outdir}")
    print("Key files:")
    print(" - knn_baseline_metrics.csv")
    print(" - knn_baseline_summary.csv")
    print(" - knn_feature_audit.csv")
    print(" - FigureS_NN_baseline_R2_B_pair_ratio.png")
    print(" - FigureS_NN_baseline_MAE_B_pair_ratio.png")


if __name__ == "__main__":
    main()
