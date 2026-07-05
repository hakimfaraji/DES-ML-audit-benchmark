#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table S5: leakage-audit summary for the DES ML audit workflow.

Purpose
-------
This script inspects the GOLD descriptor-ready dataset and verifies that the
feature matrices used in the leakage-safe workflows do not contain:
  1) the target column itself,
  2) any non-target property column,
  3) obvious metadata / identity columns as numeric predictors.

It also reports the grouping definition and group counts for Protocols B/C/D.

Inputs
------
  Unified_DES_dataset_GOLD_descriptor_ready_subset.csv

Outputs
-------
  <outdir>/Table_S5_leakage_audit_summary.csv       compact SI-ready table
  <outdir>/leakage_audit_detailed.csv               target-level detailed audit
  <outdir>/leakage_audit_summary.json               machine-readable summary

Example
-------
python run_leakage_audit_table.py \
  --input Unified_DES_dataset_GOLD_descriptor_ready_subset.csv \
  --outdir si_cleanup_outputs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

TARGET_COLS = [
    "density_g_cm3",
    "viscosity_mpa_s",
    "conductivity_ms_cm",
    "surface_tension_mn_m",
    "refractive_index",
    "tm_c",
]

MODELED_TARGETS = [
    "density_g_cm3",
    "viscosity_mpa_s",
    "conductivity_ms_cm",
    "surface_tension_mn_m",
    "refractive_index",
]

PROPERTY_LABELS = {
    "density_g_cm3": "Density",
    "viscosity_mpa_s": "Viscosity",
    "conductivity_ms_cm": "Conductivity",
    "surface_tension_mn_m": "Surface tension",
    "refractive_index": "Refractive index",
}

HBA_COLS = [
    "hba_name_resolved",
    "hba_name_canonical",
    "hba_canonical_name",
    "hba_slug_canonical",
    "hba_canonical_slug",
    "hba_name_raw",
]
HBD_COLS = [
    "hbd_name_resolved",
    "hbd_name_canonical",
    "hbd_canonical_name",
    "hbd_slug_canonical",
    "hbd_canonical_slug",
    "hbd_name_raw",
]

META_PATTERNS = [
    r"^unified_row_id$",
    r"source_",
    r"article",
    r"journal",
    r"doi",
    r"reference",
    r"filename",
    r"entry_id",
    r"traceability",
    r"note",
    r"raw$",
    r"schema",
    r"inclusion",
    r"reason",
    r"status$",
    r"validation",
    r"smiles",
    r"canonical_name$",
    r"canonical_slug",
    r"component_registry",
    r"preferred_role",
    r"observed_roles",
    r"name_",
    r"_name$",
    r"slug",
    r"role",
]

PROTOCOLS = {
    "B_pair_ratio": "HBA–HBD–ratio grouping",
    "C_pair": "HBA–HBD pair grouping",
    "D_leave_HBA": "Leave-HBA-out grouping",
    "D_leave_HBD": "Leave-HBD-out grouping",
}

FEATURE_SETS = {
    "Full descriptor + ratio + temperature": "full_model",
    "Descriptors only": "descriptors_only",
    "Descriptors + ratio": "descriptors_ratio",
    "Descriptors + temperature": "descriptors_temperature",
    "Temperature-only baseline": "temperature_only",
    "Ratio-only baseline": "ratio_only",
    "Temperature + ratio baseline": "temperature_ratio",
}


def pick_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_ratio_value(x) -> float:
    if pd.isna(x):
        return np.nan
    nums = re.findall(r"[-+]?\d*\.?\d+", str(x))
    if not nums:
        return np.nan
    vals = [float(v) for v in nums]
    if len(vals) >= 2 and vals[1] != 0:
        # The project convention is HBD/HBA.
        return vals[0] / vals[1]
    return vals[0]


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "molar_ratio_numeric" not in df.columns:
        if "molar_ratio_raw" in df.columns:
            df["molar_ratio_numeric"] = df["molar_ratio_raw"].map(parse_ratio_value)
        else:
            df["molar_ratio_numeric"] = np.nan
    if "temperature_k" not in df.columns and "measurement_temperature_c" in df.columns:
        df["temperature_k"] = pd.to_numeric(df["measurement_temperature_c"], errors="coerce") + 273.15
    return df


def is_meta_col(col: str) -> bool:
    return any(re.search(pat, col, flags=re.IGNORECASE) for pat in META_PATTERNS)


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def build_feature_columns(df: pd.DataFrame, feature_set_key: str, target: str) -> List[str]:
    nums = numeric_cols(df)
    all_targets = [c for c in TARGET_COLS if c in df.columns]
    safe_nums = [c for c in nums if c not in all_targets and not is_meta_col(c)]

    descriptor_cols = [c for c in safe_nums if "descriptor" in c.lower()]
    ratio_cols = [c for c in safe_nums if c == "molar_ratio_numeric" or "ratio" in c.lower()]
    temperature_cols = [c for c in safe_nums if c in ["measurement_temperature_c", "temperature_k"]]

    if feature_set_key == "full_model":
        cols = descriptor_cols + ratio_cols + temperature_cols
    elif feature_set_key == "descriptors_only":
        cols = descriptor_cols
    elif feature_set_key == "descriptors_ratio":
        cols = descriptor_cols + ratio_cols
    elif feature_set_key == "descriptors_temperature":
        cols = descriptor_cols + temperature_cols
    elif feature_set_key == "temperature_only":
        cols = temperature_cols
    elif feature_set_key == "ratio_only":
        cols = ratio_cols
    elif feature_set_key == "temperature_ratio":
        cols = temperature_cols + ratio_cols
    else:
        raise ValueError(f"Unknown feature set: {feature_set_key}")

    # Preserve order, remove duplicates.
    return list(dict.fromkeys(cols))


def make_groups(df: pd.DataFrame, protocol: str) -> pd.Series:
    hba_col = pick_existing(df, HBA_COLS)
    hbd_col = pick_existing(df, HBD_COLS)
    if hba_col is None or hbd_col is None:
        return pd.Series(["MISSING_GROUP"] * len(df), index=df.index)

    hba = df[hba_col].fillna("UNKNOWN_HBA").astype(str)
    hbd = df[hbd_col].fillna("UNKNOWN_HBD").astype(str)
    ratio = df.get("molar_ratio_numeric", pd.Series(np.nan, index=df.index)).round(6).astype(str)

    if protocol == "B_pair_ratio":
        return hba + " || " + hbd + " || " + ratio
    if protocol == "C_pair":
        return hba + " || " + hbd
    if protocol == "D_leave_HBA":
        return hba
    if protocol == "D_leave_HBD":
        return hbd
    raise ValueError(f"Unknown protocol: {protocol}")


def group_stats(groups: pd.Series) -> Dict[str, float]:
    vc = groups.value_counts(dropna=False)
    return {
        "n_groups": int(vc.shape[0]),
        "group_size_min": int(vc.min()) if len(vc) else 0,
        "group_size_median": float(vc.median()) if len(vc) else np.nan,
        "group_size_max": int(vc.max()) if len(vc) else 0,
    }


def audit_one(df: pd.DataFrame, target: str, protocol: str, feature_label: str, feature_key: str) -> Dict:
    d = df[df[target].notna()].copy()
    features = build_feature_columns(d, feature_key, target)
    leaked_target = [c for c in features if c == target]
    leaked_other_properties = [c for c in features if c in TARGET_COLS and c != target]
    meta_in_X = [c for c in features if is_meta_col(c)]
    hba_col = pick_existing(d, HBA_COLS)
    hbd_col = pick_existing(d, HBD_COLS)
    identity_cols = [c for c in [hba_col, hbd_col, "molar_ratio_numeric"] if c is not None and c in features]
    groups = make_groups(d, protocol)
    gstats = group_stats(groups)

    return {
        "target": PROPERTY_LABELS.get(target, target),
        "target_col": target,
        "workflow": feature_label,
        "protocol": protocol,
        "group_definition": PROTOCOLS[protocol],
        "n_rows": int(len(d)),
        "n_features": int(len(features)),
        "target_in_X": bool(leaked_target),
        "any_other_property_col_in_X": bool(leaked_other_properties),
        "leaked_property_columns": "; ".join(leaked_other_properties),
        "metadata_or_identity_cols_in_X": bool(meta_in_X or identity_cols),
        "metadata_identity_columns": "; ".join(list(dict.fromkeys(meta_in_X + identity_cols))),
        **gstats,
        "audit_status": "PASS" if not leaked_target and not leaked_other_properties else "FAIL",
    }


def make_compact_table(detailed: pd.DataFrame) -> pd.DataFrame:
    # Compact SI table: one row per workflow/protocol, aggregated over all targets.
    rows = []
    for (workflow, protocol, group_definition), g in detailed.groupby(["workflow", "protocol", "group_definition"], dropna=False):
        rows.append({
            "Workflow / feature set": workflow,
            "Protocol": protocol,
            "Grouping definition": group_definition,
            "Targets audited": int(g["target"].nunique()),
            "Feature count range": f"{int(g['n_features'].min())}–{int(g['n_features'].max())}",
            "Target in X": "No" if not g["target_in_X"].any() else "Yes",
            "Other property columns in X": "No" if not g["any_other_property_col_in_X"].any() else "Yes",
            "Metadata / identity columns in X": "No" if not g["metadata_or_identity_cols_in_X"].any() else "Yes",
            "Group count range": f"{int(g['n_groups'].min())}–{int(g['n_groups'].max())}",
            "Audit status": "PASS" if (g["audit_status"] == "PASS").all() else "FAIL",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="GOLD descriptor-ready CSV")
    ap.add_argument("--outdir", default="si_cleanup_outputs")
    ap.add_argument("--targets", nargs="+", default=MODELED_TARGETS)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = add_engineered_columns(pd.read_csv(args.input))

    rows = []
    for target in args.targets:
        if target not in df.columns:
            print(f"[WARN] target not found, skipping: {target}")
            continue
        for protocol in PROTOCOLS:
            for feature_label, feature_key in FEATURE_SETS.items():
                rows.append(audit_one(df, target, protocol, feature_label, feature_key))

    detailed = pd.DataFrame(rows)
    compact = make_compact_table(detailed)

    detailed.to_csv(outdir / "leakage_audit_detailed.csv", index=False)
    compact.to_csv(outdir / "Table_S5_leakage_audit_summary.csv", index=False)

    summary = {
        "n_detailed_rows": int(len(detailed)),
        "n_compact_rows": int(len(compact)),
        "overall_status": "PASS" if (detailed["audit_status"] == "PASS").all() else "FAIL",
        "outputs": [
            "Table_S5_leakage_audit_summary.csv",
            "leakage_audit_detailed.csv",
        ],
    }
    (outdir / "leakage_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] Leakage audit complete.")
    print(f"[OUT] {outdir / 'Table_S5_leakage_audit_summary.csv'}")
    print(f"[OUT] {outdir / 'leakage_audit_detailed.csv'}")
    print(f"[STATUS] {summary['overall_status']}")


if __name__ == "__main__":
    main()
