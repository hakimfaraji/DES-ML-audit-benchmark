#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applicability-domain / distance-to-training analysis for Line 1 DES ML audit.

Purpose
-------
Quantify whether prediction error increases as test samples move farther from
training data in leakage-safe feature space. The analysis is aligned with the
final Line 1 audit pipeline:

- leakage-safe full numeric feature set only;
- no target or cross-property columns in X;
- Protocol B: HBA-HBD-ratio grouped CV;
- Protocol C: HBA-HBD pair grouped CV;
- Protocol D: leave-HBA-out and leave-HBD-out grouped CV;
- ExtraTrees model, matching the main manuscript benchmark;
- kNN distance to training samples in standardized/imputed feature space.

Expected input
--------------
Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Main outputs
------------
applicability_domain_outputs/
  ad_sample_distances.csv
  distance_error_quartiles.csv
  ad_metrics_by_property_protocol.csv
  ad_spearman_error_distance.csv
  ad_leakage_audit.csv
  FigureS_AD_error_vs_distance.png/pdf
  FigureS_AD_distance_distribution.png/pdf
  README_applicability_domain_outputs.md

Recommended Colab command
-------------------------
!python run_applicability_domain_analysis.py \
  --input Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir applicability_domain_outputs
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42

TARGETS: Dict[str, str] = {
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

PROTOCOLS = [
    "B_leakage_corrected_pair_ratio_group",
    "C_strict_pair_group",
    "D_leave_one_hba_out",
    "D_leave_one_hbd_out",
]

PROTOCOL_LABELS = {
    "B_leakage_corrected_pair_ratio_group": "B: pair+ratio",
    "C_strict_pair_group": "C: pair",
    "D_leave_one_hba_out": "D: leave-HBA-out",
    "D_leave_one_hbd_out": "D: leave-HBD-out",
}

PROPERTY_LABELS = {
    "density": "Density",
    "viscosity": "Viscosity",
    "conductivity": "Conductivity",
    "surface_tension": "Surface tension",
    "refractive_index": "Refractive index",
}


def numeric_nonempty(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            if df[c].notna().any():
                keep.append(c)
    return keep


def infer_full_leakage_safe_features(df: pd.DataFrame, target_col: str) -> List[str]:
    """Full numeric feature set matching the leakage-safe final audit logic."""
    numeric_cols = numeric_nonempty(df, list(df.columns))
    drop = set(META_DROP).union(ALL_TARGET_COLS)
    cols = [c for c in numeric_cols if c not in drop]
    # Final safety filter: exclude columns that look like targets or source identifiers.
    bad_tokens = ["density", "viscosity", "conductivity", "surface_tension", "refractive_index"]
    safe_cols = []
    for c in cols:
        cl = c.lower()
        if c == target_col:
            continue
        if c in ALL_TARGET_COLS:
            continue
        if any(tok in cl for tok in bad_tokens):
            continue
        safe_cols.append(c)
    return safe_cols


def make_pair_group(df: pd.DataFrame) -> pd.Series:
    hba = df.get("hba_name_resolved", pd.Series(["NA"] * len(df), index=df.index)).fillna("NA").astype(str)
    hbd = df.get("hbd_name_resolved", pd.Series(["NA"] * len(df), index=df.index)).fillna("NA").astype(str)
    return hba + " || " + hbd


def make_pair_ratio_group(df: pd.DataFrame) -> pd.Series:
    ratio = df.get("molar_ratio_raw", pd.Series(["NA"] * len(df), index=df.index)).fillna("NA").astype(str)
    return make_pair_group(df) + " || " + ratio


def make_component_group(df: pd.DataFrame, component: str) -> pd.Series:
    col = "hba_name_resolved" if component == "hba" else "hbd_name_resolved"
    return df.get(col, pd.Series(["NA"] * len(df), index=df.index)).fillna("NA").astype(str)


def choose_splits(df_task: pd.DataFrame, protocol: str, n_splits: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    if protocol == "B_leakage_corrected_pair_ratio_group":
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
        raise ValueError(f"Unknown protocol: {protocol}")

    n_groups = int(groups.nunique(dropna=False))
    if n_groups < 2:
        return [], group_label
    k = min(n_splits, n_groups)
    splitter = GroupKFold(n_splits=k)
    return [(tr, te) for tr, te in splitter.split(df_task, groups=groups)], group_label


def make_model(task_name: str) -> Pipeline | TransformedTargetRegressor:
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    if task_name in POSITIVE_SKEWED:
        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return base


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def compute_knn_distance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and nearest distance to k nearest training samples in standardized feature space."""
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    k = max(1, min(k_neighbors, Xtr.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Xtr)
    distances, _ = nn.kneighbors(Xte, return_distance=True)
    return distances.mean(axis=1), distances[:, 0]


def quartile_label_from_series(s: pd.Series) -> pd.Series:
    """Robust quartile labels even when distances have duplicate edges."""
    ranks = s.rank(method="first")
    labels = ["Q1 closest", "Q2", "Q3", "Q4 farthest"]
    return pd.qcut(ranks, q=4, labels=labels)


def run_analysis(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    sample_rows = []
    metric_rows = []
    audit_rows = []

    for task_name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        df_task = df[df[target_col].notna()].copy()
        y = pd.to_numeric(df_task[target_col], errors="coerce")
        valid = y.notna() & np.isfinite(y)
        df_task = df_task.loc[valid].copy().reset_index(drop=True)
        y = y.loc[valid].astype(float).reset_index(drop=True)

        feat_cols = infer_full_leakage_safe_features(df_task, target_col)
        audit_rows.append({
            "property": task_name,
            "target_col": target_col,
            "n_rows": int(len(df_task)),
            "n_features": int(len(feat_cols)),
            "target_in_X": bool(target_col in feat_cols),
            "any_property_col_in_X": bool(any(c in feat_cols for c in ALL_TARGET_COLS)),
            "property_cols_in_X": ";".join([c for c in ALL_TARGET_COLS if c in feat_cols]),
            "first_40_features": ";".join(feat_cols[:40]),
        })
        if not feat_cols:
            print(f"[WARN] No leakage-safe numeric features for {task_name}; skipping.")
            continue

        X = df_task[feat_cols].copy()

        for protocol in PROTOCOLS:
            splits, group_label = choose_splits(df_task, protocol, args.n_splits)
            if not splits:
                continue
            all_y_true, all_y_pred = [], []
            for fold, (tr, te) in enumerate(splits, start=1):
                X_train, X_test = X.iloc[tr], X.iloc[te]
                y_train, y_test = y.iloc[tr], y.iloc[te]

                model = make_model(task_name)
                try:
                    model.fit(X_train, y_train)
                    pred = np.asarray(model.predict(X_test), dtype=float)
                except Exception as exc:
                    print(f"[WARN] Model failed: {task_name} {protocol} fold {fold}: {exc}")
                    continue

                mean_dist, nearest_dist = compute_knn_distance(X_train, X_test, args.k_neighbors)
                abs_err = np.abs(np.asarray(y_test) - pred)
                sq_err = (np.asarray(y_test) - pred) ** 2

                for local_i, orig_i in enumerate(te):
                    sample_rows.append({
                        "property": task_name,
                        "property_label": PROPERTY_LABELS.get(task_name, task_name),
                        "target_col": target_col,
                        "protocol": protocol,
                        "protocol_label": PROTOCOL_LABELS[protocol],
                        "group_definition": group_label,
                        "fold": int(fold),
                        "row_index_within_property": int(orig_i),
                        "unified_row_id": df_task.iloc[orig_i].get("unified_row_id", ""),
                        "hba_name_resolved": df_task.iloc[orig_i].get("hba_name_resolved", ""),
                        "hbd_name_resolved": df_task.iloc[orig_i].get("hbd_name_resolved", ""),
                        "molar_ratio_raw": df_task.iloc[orig_i].get("molar_ratio_raw", ""),
                        "measurement_temperature_c": df_task.iloc[orig_i].get("measurement_temperature_c", np.nan),
                        "y_true": float(np.asarray(y_test)[local_i]),
                        "y_pred": float(pred[local_i]),
                        "absolute_error": float(abs_err[local_i]),
                        "squared_error": float(sq_err[local_i]),
                        "knn_mean_distance": float(mean_dist[local_i]),
                        "knn_nearest_distance": float(nearest_dist[local_i]),
                        "k_neighbors": int(min(args.k_neighbors, len(X_train))),
                        "n_train": int(len(tr)),
                        "n_test_fold": int(len(te)),
                        "n_features": int(len(feat_cols)),
                    })

                all_y_true.extend(np.asarray(y_test, dtype=float).tolist())
                all_y_pred.extend(pred.tolist())

            if all_y_true:
                yy = np.asarray(all_y_true, dtype=float)
                pp = np.asarray(all_y_pred, dtype=float)
                metric_rows.append({
                    "property": task_name,
                    "property_label": PROPERTY_LABELS.get(task_name, task_name),
                    "target_col": target_col,
                    "protocol": protocol,
                    "protocol_label": PROTOCOL_LABELS[protocol],
                    "model": "ExtraTrees_full_leakage_safe",
                    "n_samples": int(len(yy)),
                    "n_folds": int(len(splits)),
                    "n_features": int(len(feat_cols)),
                    "mae": float(mean_absolute_error(yy, pp)),
                    "rmse": rmse(yy, pp),
                    "r2": safe_r2(yy, pp),
                })

    samples = pd.DataFrame(sample_rows)
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)

    if samples.empty:
        raise RuntimeError("No AD samples were generated. Check input columns and target availability.")

    # Assign quartiles within each property/protocol so distance bins are comparable within a task.
    samples["distance_quartile"] = (
        samples.groupby(["property", "protocol"], group_keys=False)["knn_mean_distance"]
        .apply(quartile_label_from_series)
        .astype(str)
    )

    quart = (
        samples.groupby(["property", "property_label", "protocol", "protocol_label", "distance_quartile"], as_index=False)
        .agg(
            n=("absolute_error", "size"),
            mean_knn_distance=("knn_mean_distance", "mean"),
            median_knn_distance=("knn_mean_distance", "median"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            mean_y_true=("y_true", "mean"),
        )
    )

    # Add far/near MAE ratio and delta for compact interpretation.
    extra_rows = []
    for (prop, prot), g in quart.groupby(["property", "protocol"]):
        g_idx = g.set_index("distance_quartile")
        if "Q1 closest" in g_idx.index and "Q4 farthest" in g_idx.index:
            q1 = float(g_idx.loc["Q1 closest", "mean_absolute_error"])
            q4 = float(g_idx.loc["Q4 farthest", "mean_absolute_error"])
            extra_rows.append({
                "property": prop,
                "protocol": prot,
                "q4_minus_q1_mae": q4 - q1,
                "q4_over_q1_mae": (q4 / q1) if q1 > 0 else np.nan,
            })
    extra = pd.DataFrame(extra_rows)
    if not extra.empty:
        quart = quart.merge(extra, on=["property", "protocol"], how="left")

    corr_rows = []
    for (prop, prot), g in samples.groupby(["property", "protocol"]):
        pearson = g[["knn_mean_distance", "absolute_error"]].corr(method="pearson").iloc[0, 1]
        spearman = g[["knn_mean_distance", "absolute_error"]].corr(method="spearman").iloc[0, 1]
        corr_rows.append({
            "property": prop,
            "property_label": PROPERTY_LABELS.get(prop, prop),
            "protocol": prot,
            "protocol_label": PROTOCOL_LABELS.get(prot, prot),
            "n": int(len(g)),
            "pearson_distance_error": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman_distance_error": float(spearman) if pd.notna(spearman) else np.nan,
        })
    corr = pd.DataFrame(corr_rows)

    samples.to_csv(outdir / "ad_sample_distances.csv", index=False)
    quart.to_csv(outdir / "distance_error_quartiles.csv", index=False)
    metrics.to_csv(outdir / "ad_metrics_by_property_protocol.csv", index=False)
    corr.to_csv(outdir / "ad_spearman_error_distance.csv", index=False)
    audit.to_csv(outdir / "ad_leakage_audit.csv", index=False)

    make_error_quartile_figure(quart, outdir)
    make_distance_distribution_figure(samples, outdir)

    manifest = {
        "input": str(input_path),
        "outdir": str(outdir),
        "seed": SEED,
        "k_neighbors": args.k_neighbors,
        "n_splits_requested": args.n_splits,
        "protocols": PROTOCOLS,
        "targets": TARGETS,
        "model": "ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2)",
        "distance_space": "median-imputed + StandardScaler fitted on each training fold; Euclidean kNN mean distance",
        "leakage_policy": "all target and cross-property columns removed before modeling and distance computation",
    }
    with open(outdir / "ad_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = f"""# Applicability-domain outputs\n\nThis folder contains distance-to-training analysis for the DES Line 1 manuscript.\n\n## Method\nFor each property and leakage-safe protocol, an ExtraTrees full-feature model was trained on grouped folds. For every test sample, the mean Euclidean distance to the {args.k_neighbors} nearest training samples was computed after median imputation and standardization fitted only on the training fold. Prediction error was then summarized across distance quartiles.\n\n## Key files\n- `ad_sample_distances.csv`: per-test-sample predictions, absolute errors, and kNN distances.\n- `distance_error_quartiles.csv`: error summarized by within-property/protocol distance quartile.\n- `ad_spearman_error_distance.csv`: Pearson/Spearman association between kNN distance and absolute error.\n- `ad_leakage_audit.csv`: verifies target/cross-property leakage exclusion.\n- `FigureS_AD_error_vs_distance.png/pdf`: main SI figure for error vs distance quartiles.\n- `FigureS_AD_distance_distribution.png/pdf`: supporting SI figure for distance distributions.\n\n## Interpretation rule\nA positive Q4-Q1 MAE difference or positive Spearman correlation indicates that predictions degrade farther from the training domain. Weak or inconsistent trends should be interpreted conservatively as evidence that static descriptors and sparse coverage limit extrapolative reliability, not as proof of a universal distance law.\n"""
    (outdir / "README_applicability_domain_outputs.md").write_text(readme, encoding="utf-8")

    print("[DONE] Applicability-domain analysis complete.")
    print(f"Outputs written to: {outdir}")
    print("Key files:")
    print(" - distance_error_quartiles.csv")
    print(" - ad_spearman_error_distance.csv")
    print(" - FigureS_AD_error_vs_distance.png")
    print(" - FigureS_AD_distance_distribution.png")


def make_error_quartile_figure(quart: pd.DataFrame, outdir: Path) -> None:
    order_props = ["Density", "Refractive index", "Surface tension", "Conductivity", "Viscosity"]
    order_q = ["Q1 closest", "Q2", "Q3", "Q4 farthest"]
    protocols = [p for p in PROTOCOLS if p in quart["protocol"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, protocol in zip(axes, protocols):
        g = quart[quart["protocol"] == protocol].copy()
        x = np.arange(len(order_q))
        width = 0.15
        labels_present = [p for p in order_props if p in set(g["property_label"])]
        offsets = np.linspace(-width * 2, width * 2, max(1, len(labels_present)))
        for off, prop_label in zip(offsets, labels_present):
            gg = g[g["property_label"] == prop_label].set_index("distance_quartile")
            vals = [gg.loc[q, "mean_absolute_error"] if q in gg.index else np.nan for q in order_q]
            ax.bar(x + off, vals, width=width, label=prop_label)
        ax.set_title(PROTOCOL_LABELS.get(protocol, protocol))
        ax.set_xticks(x)
        ax.set_xticklabels(order_q, rotation=20, ha="right")
        ax.set_ylabel("Mean absolute error")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    for j in range(len(protocols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Applicability-domain analysis: prediction error by distance-to-training quartile", fontsize=14)
    fig.savefig(outdir / "FigureS_AD_error_vs_distance.png", dpi=300)
    fig.savefig(outdir / "FigureS_AD_error_vs_distance.pdf")
    plt.close(fig)


def make_distance_distribution_figure(samples: pd.DataFrame, outdir: Path) -> None:
    order_props = ["Density", "Refractive index", "Surface tension", "Conductivity", "Viscosity"]
    protocols = [p for p in PROTOCOLS if p in samples["protocol"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, protocol in zip(axes, protocols):
        g = samples[samples["protocol"] == protocol].copy()
        data = [g.loc[g["property_label"] == p, "knn_mean_distance"].dropna().values for p in order_props]
        labels = [p for p, d in zip(order_props, data) if len(d) > 0]
        data = [d for d in data if len(d) > 0]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(PROTOCOL_LABELS.get(protocol, protocol))
        ax.set_ylabel("Mean kNN distance to training fold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    for j in range(len(protocols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature-space distance distributions under leakage-safe validation", fontsize=14)
    fig.savefig(outdir / "FigureS_AD_distance_distribution.png", dpi=300)
    fig.savefig(outdir / "FigureS_AD_distance_distribution.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run applicability-domain distance analysis for DES Line 1.")
    parser.add_argument("--input", required=True, help="Path to Unified_DES_dataset_GOLD_descriptor_ready_subset.csv")
    parser.add_argument("--outdir", default="applicability_domain_outputs", help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits for each protocol")
    parser.add_argument("--k-neighbors", type=int, default=5, help="k for mean kNN distance")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
