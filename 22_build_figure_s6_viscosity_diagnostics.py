#!/usr/bin/env python3
"""
Build Figure S6: Extended viscosity prediction diagnostics.

Outputs:
  FigureS6_extended_viscosity_diagnostics.png
  FigureS6_extended_viscosity_diagnostics.pdf
  FigureS6_viscosity_oof_predictions.csv
  FigureS6_protocol_metrics.csv

Usage:
  python build_figure_s6_viscosity_diagnostics.py \
    --input Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
    --outdir si_final_figures
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "viscosity_mpa_s"
TEMP_C = "measurement_temperature_c"
HBA = "hba_name_raw"
HBD = "hbd_name_raw"
RATIO_RAW = "molar_ratio_raw"

PROPERTY_COLS = {
    "density_g_cm3", "viscosity_mpa_s", "conductivity_ms_cm",
    "surface_tension_mn_m", "refractive_index"
}


def parse_ratio(value) -> float:
    """Parse common HBA:HBD or HBD/HBA ratio strings into HBD/HBA numeric ratio."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    # remove labels and spaces
    s_clean = re.sub(r"\s+", "", s)
    # patterns like 1:2, 1/2, 1-2 (assumed HBA:HBD -> HBD/HBA = b/a)
    m = re.search(r"(\d+(?:\.\d+)?)\s*[:/\-]\s*(\d+(?:\.\d+)?)", s_clean)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return b / a if a != 0 else np.nan
    # single number
    nums = re.findall(r"\d+(?:\.\d+)?", s_clean)
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def canonical_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of candidate columns found: {candidates}")


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    if "molar_ratio_numeric" not in work.columns:
        work["molar_ratio_numeric"] = work[RATIO_RAW].apply(parse_ratio)
    work["temperature_k"] = work[TEMP_C].astype(float) + 273.15

    desc_cols = [c for c in work.columns if c.startswith("hba_descriptor_") or c.startswith("hbd_descriptor_")]
    # keep only numeric descriptor columns
    numeric_desc = []
    for c in desc_cols:
        x = pd.to_numeric(work[c], errors="coerce")
        # exclude completely empty columns
        if x.notna().sum() > 0:
            work[c] = x
            numeric_desc.append(c)

    feature_cols = numeric_desc + ["molar_ratio_numeric", "temperature_k"]
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X, feature_cols


def groups_for_protocol(df: pd.DataFrame, protocol: str) -> pd.Series:
    hba = df[HBA].fillna("NA").astype(str)
    hbd = df[HBD].fillna("NA").astype(str)
    ratio = df["molar_ratio_numeric"].round(6).astype(str)
    if protocol == "B_pair_ratio":
        return hba + "||" + hbd + "||" + ratio
    if protocol == "C_pair":
        return hba + "||" + hbd
    if protocol == "D_leave_HBA":
        return hba
    if protocol == "D_leave_HBD":
        return hbd
    raise ValueError(protocol)


def make_splits(groups: pd.Series, protocol: str, n_splits_default: int = 5):
    unique_groups = pd.Series(groups).dropna().unique()
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for {protocol}")
    n_splits = min(n_splits_default, n_groups)
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)


def fit_predict_oof(df: pd.DataFrame, protocol: str, seed: int = 42) -> pd.DataFrame:
    X, feature_cols = build_features(df)
    y_raw = pd.to_numeric(df[TARGET], errors="coerce")
    mask = y_raw.notna() & (y_raw > 0) & X.notna().any(axis=1)
    data = df.loc[mask].copy().reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y_raw = y_raw.loc[mask].reset_index(drop=True)
    y_log = np.log10(y_raw)
    data["molar_ratio_numeric"] = X["molar_ratio_numeric"]
    data["temperature_k"] = X["temperature_k"]

    groups = groups_for_protocol(data, protocol)
    preds_log = np.full(len(data), np.nan)
    fold_ids = np.full(len(data), -1)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # Scaling is useful for diagnostics and harmless for tree models.
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(
            n_estimators=80,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
            max_features="sqrt",
        )),
    ])

    for fold, (tr, te) in enumerate(make_splits(groups, protocol), start=1):
        # skip pathological folds with too few train examples
        if len(tr) < 10 or len(te) == 0:
            continue
        model.fit(X.iloc[tr], y_log.iloc[tr])
        preds_log[te] = model.predict(X.iloc[te])
        fold_ids[te] = fold

    out = pd.DataFrame({
        "protocol": protocol,
        "fold": fold_ids,
        "observed_viscosity_mpa_s": y_raw,
        "predicted_log10_viscosity": preds_log,
        "observed_log10_viscosity": y_log,
        "molar_ratio_numeric": data["molar_ratio_numeric"],
        "temperature_k": data["temperature_k"],
        "temperature_c": pd.to_numeric(data[TEMP_C], errors="coerce"),
        "hba_name_raw": data[HBA].astype(str),
        "hbd_name_raw": data[HBD].astype(str),
    })
    out = out[out["predicted_log10_viscosity"].notna()].copy()
    out["predicted_viscosity_mpa_s"] = np.power(10.0, out["predicted_log10_viscosity"])
    out["residual_log10"] = out["predicted_log10_viscosity"] - out["observed_log10_viscosity"]
    out["absolute_error_log10"] = out["residual_log10"].abs()
    out["residual_raw"] = out["predicted_viscosity_mpa_s"] - out["observed_viscosity_mpa_s"]
    out["absolute_error_raw"] = out["residual_raw"].abs()
    return out


def protocol_label(protocol: str) -> str:
    return {
        "B_pair_ratio": "B: pair+ratio",
        "C_pair": "C: pair",
        "D_leave_HBA": "D: leave-HBA-out",
        "D_leave_HBD": "D: leave-HBD-out",
    }[protocol]


def build_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in preds.groupby("protocol"):
        rows.append({
            "Protocol": p,
            "n": len(g),
            "R2_log10": r2_score(g["observed_log10_viscosity"], g["predicted_log10_viscosity"]),
            "MAE_log10": mean_absolute_error(g["observed_log10_viscosity"], g["predicted_log10_viscosity"]),
            "RMSE_log10": math.sqrt(mean_squared_error(g["observed_log10_viscosity"], g["predicted_log10_viscosity"])),
            "MAE_raw_mPa_s": mean_absolute_error(g["observed_viscosity_mpa_s"], g["predicted_viscosity_mpa_s"]),
            "RMSE_raw_mPa_s": math.sqrt(mean_squared_error(g["observed_viscosity_mpa_s"], g["predicted_viscosity_mpa_s"])),
        })
    return pd.DataFrame(rows)


def make_figure(preds: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    protocols = ["B_pair_ratio", "C_pair", "D_leave_HBA", "D_leave_HBD"]
    fig, axes = plt.subplots(len(protocols), 3, figsize=(13.5, 14.5), constrained_layout=True)

    for r, protocol in enumerate(protocols):
        g = preds[preds["protocol"] == protocol]
        label = protocol_label(protocol)

        # Observed vs predicted in log space
        ax = axes[r, 0]
        ax.scatter(g["observed_log10_viscosity"], g["predicted_log10_viscosity"], s=10, alpha=0.45)
        lim_min = np.nanmin([g["observed_log10_viscosity"].min(), g["predicted_log10_viscosity"].min()])
        lim_max = np.nanmax([g["observed_log10_viscosity"].max(), g["predicted_log10_viscosity"].max()])
        pad = 0.05 * (lim_max - lim_min) if lim_max > lim_min else 0.1
        ax.plot([lim_min-pad, lim_max+pad], [lim_min-pad, lim_max+pad], linestyle="--", linewidth=1)
        ax.set_xlim(lim_min-pad, lim_max+pad); ax.set_ylim(lim_min-pad, lim_max+pad)
        ax.set_title(f"{label}: observed vs predicted")
        ax.set_xlabel("Observed log10 viscosity")
        ax.set_ylabel("Predicted log10 viscosity")

        # Residual vs ratio
        ax = axes[r, 1]
        ax.scatter(g["molar_ratio_numeric"], g["residual_log10"], s=10, alpha=0.45)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{label}: residual vs ratio")
        ax.set_xlabel("HBD/HBA molar ratio")
        ax.set_ylabel("Residual log10(pred − obs)")

        # Residual vs temperature
        ax = axes[r, 2]
        ax.scatter(g["temperature_k"], g["residual_log10"], s=10, alpha=0.45)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{label}: residual vs temperature")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Residual log10(pred − obs)")

    fig.suptitle("Extended viscosity prediction diagnostics", fontsize=16, y=1.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Unified DES dataset CSV")
    ap.add_argument("--outdir", default="si_final_figures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    # Only rows with viscosity
    df = df[df[TARGET].notna()].copy()
    if "molar_ratio_numeric" not in df.columns:
        df["molar_ratio_numeric"] = df[RATIO_RAW].apply(parse_ratio)

    all_preds = []
    for p in ["B_pair_ratio", "C_pair", "D_leave_HBA", "D_leave_HBD"]:
        print(f"[INFO] Running {p}...")
        all_preds.append(fit_predict_oof(df, p, seed=args.seed))
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(outdir / "FigureS6_viscosity_oof_predictions.csv", index=False)
    metrics = build_metrics(preds)
    metrics.to_csv(outdir / "FigureS6_protocol_metrics.csv", index=False)
    make_figure(preds, outdir / "FigureS6_extended_viscosity_diagnostics.png", outdir / "FigureS6_extended_viscosity_diagnostics.pdf")
    print(f"[DONE] Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
