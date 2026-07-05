# ============================================================
# Line 1 viscosity deep diagnostic
# Project: AI_DES formation model — Line 1 manuscript
#
# Purpose:
#   Diagnose why viscosity is weakly learnable under leakage-aware
#   validation: raw-vs-log target, validation sensitivity, residual
#   structure versus temperature/ratio/magnitude, and feature importance.
#
# Input:
#   --input Unified_DES_dataset_GOLD_descriptor_ready_subset.csv
# Output directory:
#   viscosity_deep_diagnostic_outputs/
# ============================================================

from __future__ import annotations
import argparse, json, warnings, math, re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "viscosity_mpa_s"
ALL_TARGET_COLS = ["density_g_cm3", "viscosity_mpa_s", "conductivity_ms_cm", "surface_tension_mn_m", "refractive_index"]
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

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def r2(y, p): return float(r2_score(y, p)) if len(y) > 1 else np.nan

def parse_ratio_hbd_to_hba(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*[:/]\s*([0-9]*\.?[0-9]+)\s*$", s)
    if not m: return np.nan
    a, b = float(m.group(1)), float(m.group(2))
    return b / a if a != 0 else np.nan

def pair_group(df):
    hba = df.get("hba_name_resolved", pd.Series("NA", index=df.index)).fillna("NA").astype(str)
    hbd = df.get("hbd_name_resolved", pd.Series("NA", index=df.index)).fillna("NA").astype(str)
    return hba + " || " + hbd

def pair_ratio_group(df):
    return pair_group(df) + " || " + df.get("molar_ratio_raw", pd.Series("NA", index=df.index)).fillna("NA").astype(str)

def component_group(df, which):
    col = "hba_name_resolved" if which == "hba" else "hbd_name_resolved"
    return df.get(col, pd.Series("NA", index=df.index)).fillna("NA").astype(str)

def numeric_cols(df):
    return [c for c in df.columns if (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])) and df[c].notna().any()]

def feature_cols(df, feature_set):
    drop = set(META_DROP).union(ALL_TARGET_COLS)
    eligible = [c for c in numeric_cols(df) if c not in drop]
    temp = [c for c in eligible if c in {"measurement_temperature_c", "temperature_c"} or "temperature" in c.lower()]
    ratio = [c for c in eligible if c.startswith("ratio_") or "ratio_" in c or c == "ratio_hbd_to_hba_parsed"]
    desc = [c for c in eligible if "descriptor_" in c or c in {"tm_c"}]
    pair = [c for c in eligible if c.startswith("pair_") or c.startswith("ratio_weighted_")]
    if feature_set == "temperature_only": cols = temp
    elif feature_set == "ratio_only": cols = ratio
    elif feature_set == "descriptors_only": cols = desc
    elif feature_set == "descriptors_plus_ratio_temp": cols = sorted(set(desc + ratio + temp))
    elif feature_set == "full": cols = eligible
    else: raise ValueError(feature_set)
    return [c for c in cols if c in df.columns]

def model_factory(name):
    if name == "dummy_mean":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))])
    if name == "ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "extra_trees":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1))])
    if name == "hist_gradient_boosting":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=0.01, random_state=SEED))])
    raise ValueError(name)

def splits(df, protocol, n_splits):
    if protocol == "random_row":
        k = min(n_splits, len(df))
        return [(tr, te, "row") for tr, te in KFold(n_splits=k, shuffle=True, random_state=SEED).split(df)]
    if protocol == "B_pair_ratio_group":
        g, label = pair_ratio_group(df), "hba+hbd+raw_ratio"
    elif protocol == "C_pair_group":
        g, label = pair_group(df), "hba+hbd"
    elif protocol == "D_leave_hba_out":
        g, label = component_group(df, "hba"), "hba"
    elif protocol == "D_leave_hbd_out":
        g, label = component_group(df, "hbd"), "hbd"
    else: raise ValueError(protocol)
    k = min(n_splits, g.nunique())
    if k < 2: return []
    return [(tr, te, label) for tr, te in GroupKFold(n_splits=k).split(df, groups=g)]

def fit_predict(model_name, transform, Xtr, ytr, Xte):
    model = model_factory(model_name)
    if transform == "raw":
        model.fit(Xtr, ytr)
        return model.predict(Xte), model
    if transform == "log1p":
        yy = np.log1p(ytr)
        model.fit(Xtr, yy)
        return np.expm1(model.predict(Xte)), model
    raise ValueError(transform)

def summarize_bins(preds, outdir):
    p = preds.copy()
    p["log_y_true"] = np.log10(p["y_true"].clip(lower=1e-12))
    p["log_abs_error"] = np.abs(np.log1p(p["y_pred"].clip(lower=0)) - np.log1p(p["y_true"]))
    # bins by true viscosity magnitude
    p["viscosity_range"] = pd.cut(p["y_true"], bins=[0, 20, 100, 1000, 10000, np.inf], labels=["<=20", "20-100", "100-1000", "1000-10000", ">10000"], include_lowest=True)
    temp = pd.to_numeric(p.get("measurement_temperature_c"), errors="coerce")
    p["temperature_bin"] = pd.cut(temp, bins=[-np.inf, 25, 40, 60, np.inf], labels=["<=25C", "25-40C", "40-60C", ">60C"])
    ratio = pd.to_numeric(p.get("ratio_hbd_to_hba_parsed"), errors="coerce")
    p["ratio_bin"] = pd.cut(ratio, bins=[-np.inf, 1, 2, 5, np.inf], labels=["<=1", "1-2", "2-5", ">5"])
    p.to_csv(outdir / "viscosity_all_oof_predictions.csv", index=False)
    for col, fn in [("viscosity_range", "error_by_viscosity_range.csv"), ("temperature_bin", "error_by_temperature_bin.csv"), ("ratio_bin", "error_by_ratio_bin.csv")]:
        s = (p.groupby(["protocol","feature_set","model","target_transform",col], observed=False)
               .agg(n=("y_true","size"), mae=("abs_error","mean"), median_ae=("abs_error","median"),
                    rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
                    log_mae=("log_abs_error","mean"), bias=("residual","mean"))
               .reset_index())
        s.to_csv(outdir / fn, index=False)

def make_plots(preds, summary, outdir):
    # Use main clean model for figures: B/C full ExtraTrees log1p and raw if available
    for protocol in ["B_pair_ratio_group", "C_pair_group", "D_leave_hba_out", "D_leave_hbd_out"]:
        sub = preds[(preds.protocol == protocol) & (preds.feature_set == "full") & (preds.model == "extra_trees") & (preds.target_transform == "log1p")]
        if sub.empty: continue
        plt.figure(figsize=(6,5))
        plt.scatter(sub["y_true"], sub["y_pred"], s=12, alpha=0.6)
        lim = [max(1e-1, min(sub["y_true"].min(), sub["y_pred"].min())), max(sub["y_true"].max(), sub["y_pred"].max())]
        plt.plot(lim, lim)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Observed viscosity (mPa s)"); plt.ylabel("Predicted viscosity (mPa s)")
        plt.title(f"Viscosity observed vs predicted — {protocol}, ExtraTrees log1p")
        plt.tight_layout(); plt.savefig(outdir / f"observed_vs_predicted_{protocol}_extratrees_log1p.png", dpi=220); plt.close()

        plt.figure(figsize=(6,5))
        x = pd.to_numeric(sub.get("measurement_temperature_c"), errors="coerce")
        plt.scatter(x, sub["residual"], s=12, alpha=0.6)
        plt.axhline(0)
        plt.xlabel("Temperature (°C)"); plt.ylabel("Residual: predicted - observed (mPa s)")
        plt.title(f"Residual vs temperature — {protocol}")
        plt.tight_layout(); plt.savefig(outdir / f"residual_vs_temperature_{protocol}.png", dpi=220); plt.close()

        plt.figure(figsize=(6,5))
        x = pd.to_numeric(sub.get("ratio_hbd_to_hba_parsed"), errors="coerce")
        plt.scatter(x, sub["residual"], s=12, alpha=0.6)
        plt.axhline(0)
        plt.xlabel("Parsed HBD:HBA ratio"); plt.ylabel("Residual: predicted - observed (mPa s)")
        plt.title(f"Residual vs ratio — {protocol}")
        plt.tight_layout(); plt.savefig(outdir / f"residual_vs_ratio_{protocol}.png", dpi=220); plt.close()

    # summary bar-ish line of R2 by protocol/transform for full ExtraTrees
    sub = summary[(summary.feature_set == "full") & (summary.model == "extra_trees")]
    if not sub.empty:
        plt.figure(figsize=(8,5))
        labels = sub["protocol"] + "\n" + sub["target_transform"]
        plt.bar(range(len(sub)), sub["r2_mean"])
        plt.xticks(range(len(sub)), labels, rotation=45, ha="right")
        plt.ylabel("Mean CV R²")
        plt.title("Validation sensitivity for viscosity — full ExtraTrees")
        plt.tight_layout(); plt.savefig(outdir / "validation_sensitivity_full_extratrees_r2.png", dpi=220); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="viscosity_deep_diagnostic_outputs")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if TARGET not in df.columns: raise SystemExit(f"Missing target: {TARGET}")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)].copy()
    df["ratio_hbd_to_hba_parsed"] = df["molar_ratio_raw"].map(parse_ratio_hbd_to_hba) if "molar_ratio_raw" in df else np.nan

    dataset_summary = {
        "n": int(len(df)),
        "n_hba": int(df["hba_name_resolved"].nunique()) if "hba_name_resolved" in df else None,
        "n_hbd": int(df["hbd_name_resolved"].nunique()) if "hbd_name_resolved" in df else None,
        "n_pairs": int(pair_group(df).nunique()),
        "n_pair_ratios": int(pair_ratio_group(df).nunique()),
        "viscosity_mpa_s_describe": df[TARGET].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_dict(),
        "temperature_c_describe": pd.to_numeric(df.get("measurement_temperature_c"), errors="coerce").describe(percentiles=[.05,.25,.5,.75,.95]).to_dict() if "measurement_temperature_c" in df else {},
        "molar_ratio_n_unique_raw": int(df["molar_ratio_raw"].nunique()) if "molar_ratio_raw" in df else None,
    }
    (outdir / "viscosity_dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    protocols = ["random_row", "B_pair_ratio_group", "C_pair_group", "D_leave_hba_out", "D_leave_hbd_out"]
    feature_sets = ["temperature_only", "ratio_only", "descriptors_only", "descriptors_plus_ratio_temp", "full"]
    models = ["dummy_mean", "ridge", "extra_trees", "hist_gradient_boosting"]
    transforms = ["raw", "log1p"]

    all_rows, pred_rows, audit_rows = [], [], []
    for protocol in protocols:
        spl = splits(df, protocol, args.n_splits)
        for feature_set in feature_sets:
            cols = feature_cols(df, feature_set)
            audit_rows.append({"protocol": protocol, "feature_set": feature_set, "n_features": len(cols), "target_in_X": TARGET in cols, "property_cols_in_X": ";".join([c for c in ALL_TARGET_COLS if c in cols]), "first_40_features": ";".join(cols[:40])})
            if not cols or not spl: continue
            X = df[cols]
            y = df[TARGET].astype(float)
            for model_name in models:
                for transform in transforms:
                    if model_name == "dummy_mean" and transform == "log1p":
                        continue
                    fold_metrics = []
                    for fold, (tr, te, group_label) in enumerate(spl, start=1):
                        try:
                            pred, fitted = fit_predict(model_name, transform, X.iloc[tr], y.iloc[tr], X.iloc[te])
                            pred = np.asarray(pred, dtype=float)
                            pred[pred < 0] = 0.0
                            yt = y.iloc[te].to_numpy(dtype=float)
                            m = {"mae": mean_absolute_error(yt, pred), "rmse": rmse(yt, pred), "r2": r2(yt, pred), "log_mae": mean_absolute_error(np.log1p(yt), np.log1p(pred))}
                            fold_metrics.append(m)
                            tmp = df.iloc[te][[c for c in ["unified_row_id","hba_name_resolved","hbd_name_resolved","molar_ratio_raw","ratio_hbd_to_hba_parsed","measurement_temperature_c"] if c in df.columns]].copy()
                            tmp["protocol"] = protocol; tmp["group_definition"] = group_label; tmp["feature_set"] = feature_set; tmp["model"] = model_name; tmp["target_transform"] = transform; tmp["fold"] = fold
                            tmp["y_true"] = yt; tmp["y_pred"] = pred; tmp["residual"] = pred - yt; tmp["abs_error"] = np.abs(pred - yt); tmp["sq_error"] = (pred - yt)**2
                            pred_rows.append(tmp)
                        except Exception as e:
                            all_rows.append({"protocol": protocol, "feature_set": feature_set, "model": model_name, "target_transform": transform, "status": f"ERROR: {e}"})
                    if fold_metrics:
                        d = pd.DataFrame(fold_metrics)
                        all_rows.append({"protocol": protocol, "group_definition": group_label, "feature_set": feature_set, "model": model_name, "target_transform": transform, "status": "ok", "n_folds": len(d), "n_test_total": sum(len(te) for _, te, _ in spl), "n_features": len(cols),
                                         "mae_mean": d.mae.mean(), "mae_std": d.mae.std(), "rmse_mean": d.rmse.mean(), "rmse_std": d.rmse.std(), "r2_mean": d.r2.mean(), "r2_std": d.r2.std(), "log_mae_mean": d.log_mae.mean(), "log_mae_std": d.log_mae.std()})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(outdir / "viscosity_model_summary.csv", index=False)
    audit = pd.DataFrame(audit_rows); audit.to_csv(outdir / "viscosity_feature_audit.csv", index=False)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if not preds.empty:
        summarize_bins(preds, outdir)
        make_plots(preds, summary[summary.status.eq("ok")].copy(), outdir)

    # permutation importance for B/C full ExtraTrees log1p using first split
    fi_rows = []
    for protocol in ["B_pair_ratio_group", "C_pair_group"]:
        cols = feature_cols(df, "full")
        spl = splits(df, protocol, args.n_splits)
        if not cols or not spl: continue
        tr, te, group_label = spl[0]
        pred, fitted = fit_predict("extra_trees", "log1p", df[cols].iloc[tr], df[TARGET].iloc[tr], df[cols].iloc[te])
        # score in raw MAE, permutation_importance maximizes scoring, so neg_mean_absolute_error reduction means important when positive.
        pi = permutation_importance(fitted, df[cols].iloc[te], np.log1p(df[TARGET].iloc[te]), n_repeats=8, random_state=SEED, scoring="neg_mean_absolute_error")
        order = np.argsort(pi.importances_mean)[::-1][:30]
        for rank, idx in enumerate(order, 1):
            fi_rows.append({"protocol": protocol, "model": "extra_trees", "target_transform": "log1p", "rank": rank, "feature": cols[idx], "importance_mean_delta_log_mae": float(pi.importances_mean[idx]), "importance_std": float(pi.importances_std[idx])})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(outdir / "viscosity_permutation_importance.csv", index=False)

    (outdir / "run_manifest.json").write_text(json.dumps({"input": str(Path(args.input).resolve()), "target": TARGET, "seed": SEED, "n_splits": args.n_splits, "notes": ["All target/property columns are removed from X.", "raw and log1p target variants are compared.", "OOF predictions are provided for residual diagnostics."]}, indent=2), encoding="utf-8")
    print(f"Done. Outputs written to: {outdir}")

if __name__ == "__main__":
    main()
