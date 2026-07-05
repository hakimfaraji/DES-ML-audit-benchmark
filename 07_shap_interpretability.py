#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHAP interpretability V3 hotfix for AI_DES Line 1.

Fixes SHAP `isfinite` dtype errors by converting the final X matrix to
finite float64 arrays and median-imputing before fitting/explaining.
"""
from __future__ import annotations
import argparse, json, math, re, sys, warnings
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
warnings.filterwarnings('ignore')

TARGET_COLS = ['density_g_cm3','viscosity_mpa_s','conductivity_ms_cm','surface_tension_mn_m','refractive_index','tm_c']
HBA_COLS = ['hba_name_resolved','hba_name_canonical','hba_canonical_name','hba_slug_canonical','hba_canonical_slug']
HBD_COLS = ['hbd_name_resolved','hbd_name_canonical','hbd_canonical_name','hbd_slug_canonical','hbd_canonical_slug']
META_PATTERNS = [r'^unified_row_id$', r'source_', r'article', r'journal', r'doi', r'reference', r'filename', r'entry_id', r'traceability', r'note', r'raw$', r'schema', r'inclusion', r'reason', r'status$', r'validation', r'smiles', r'canonical_name$', r'canonical_slug', r'component_registry', r'preferred_role', r'observed_roles']

def safe_slug(s):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s)).strip('_')

def pick(df, cols):
    return next((c for c in cols if c in df.columns), None)

def parse_ratio_value(x):
    if pd.isna(x): return np.nan
    nums = re.findall(r'[-+]?\d*\.?\d+', str(x))
    if not nums: return np.nan
    vals = [float(v) for v in nums]
    if len(vals) >= 2 and vals[1] != 0: return vals[0] / vals[1]
    return vals[0]

def add_engineered(df):
    df = df.copy()
    if 'molar_ratio_numeric' not in df.columns:
        df['molar_ratio_numeric'] = df['molar_ratio_raw'].map(parse_ratio_value) if 'molar_ratio_raw' in df.columns else np.nan
    if 'measurement_temperature_c' in df.columns:
        df['temperature_k'] = pd.to_numeric(df['measurement_temperature_c'], errors='coerce') + 273.15
    return df

def is_meta(c):
    return any(re.search(p, c, flags=re.I) for p in META_PATTERNS)

def build_features(df, feature_set):
    targets = [c for c in TARGET_COLS if c in df.columns]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in targets]
    desc = [c for c in numeric if 'descriptor' in c.lower()]
    temp = [c for c in numeric if c in ['measurement_temperature_c','temperature_k']]
    ratio = [c for c in numeric if 'ratio' in c.lower() or c == 'molar_ratio_numeric']
    safe_numeric = [c for c in numeric if not is_meta(c)]
    if feature_set == 'descriptors_only': feat = desc
    elif feature_set == 'descriptors_ratio': feat = desc + ratio
    elif feature_set == 'descriptors_temp': feat = desc + temp
    elif feature_set == 'descriptors_ratio_temp': feat = desc + ratio + temp
    elif feature_set == 'full_safe': feat = safe_numeric
    else: raise ValueError(feature_set)
    return list(dict.fromkeys([c for c in feat if c not in targets]))

def sanitize_X(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    Xn = pd.DataFrame(index=X.index)
    for c in X.columns:
        Xn[c] = pd.to_numeric(X[c], errors='coerce')
    Xn = Xn.replace([np.inf, -np.inf], np.nan).astype('float64')
    keep = [c for c in Xn.columns if not Xn[c].isna().all()]
    dropped = [c for c in Xn.columns if c not in keep]
    return Xn[keep], dropped

def make_groups(df, protocol):
    hba_col, hbd_col = pick(df,HBA_COLS), pick(df,HBD_COLS)
    if protocol == 'random': return None
    if hba_col is None or hbd_col is None: raise ValueError('Missing HBA/HBD identity columns')
    hba = df[hba_col].fillna('UNKNOWN_HBA').astype(str)
    hbd = df[hbd_col].fillna('UNKNOWN_HBD').astype(str)
    if protocol == 'pair_group': return hba + ' || ' + hbd
    if protocol == 'pair_ratio_group':
        ratio = df.get('molar_ratio_numeric', pd.Series(np.nan,index=df.index)).round(4).astype(str)
        return hba + ' || ' + hbd + ' || ' + ratio
    if protocol == 'leave_hba_out': return hba
    if protocol == 'leave_hbd_out': return hbd
    raise ValueError(protocol)

def make_model(name, seed):
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    if name == 'HistGB':
        return HistGradientBoostingRegressor(random_state=seed, max_iter=250, learning_rate=0.05, l2_regularization=0.01)
    raise ValueError(name)

def impute_arrays(Xtr, Xte):
    imp = SimpleImputer(strategy='median')
    a = imp.fit_transform(Xtr)
    b = imp.transform(Xte)
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return a,b

def sample_array(A, n, seed):
    if A.shape[0] <= n: return A.copy()
    rng = np.random.default_rng(seed)
    return A[rng.choice(A.shape[0], size=n, replace=False)].copy()

def run_shap(model, model_name, Xtr_np, Xte_np, feature_names, max_background, max_explain, seed):
    import shap
    bg = sample_array(Xtr_np, max_background, seed)
    xe = sample_array(Xte_np, max_explain, seed+1)
    if model_name == 'ExtraTrees':
        explainer = shap.TreeExplainer(model, feature_names=feature_names)
        vals = explainer.shap_values(xe, check_additivity=False)
        arr = np.asarray(vals, dtype=np.float64)
    else:
        masker = shap.maskers.Independent(bg)
        explainer = shap.Explainer(model.predict, masker, feature_names=feature_names)
        vals = explainer(xe)
        arr = np.asarray(vals.values, dtype=np.float64)
    if arr.ndim == 3: arr = arr[:,:,0]
    return pd.DataFrame({'feature': feature_names, 'mean_abs_shap': np.nanmean(np.abs(arr), axis=0), 'mean_signed_shap': np.nanmean(arr, axis=0), 'n_explained': xe.shape[0], 'n_background': bg.shape[0]})

def rmse(y,p): return math.sqrt(mean_squared_error(y,p))

def run_one(df, target, protocol, model_name, feature_set, args):
    d = add_engineered(df[df[target].notna()].copy())
    y_raw = pd.to_numeric(d[target], errors='coerce')
    valid = y_raw.notna()
    if target == 'viscosity_mpa_s' and args.log_viscosity:
        valid &= y_raw > 0
        y = np.log10(y_raw[valid]); target_used = 'log10_viscosity_mpa_s'
    else:
        y = y_raw[valid]; target_used = target
    d = d.loc[valid].reset_index(drop=True); y = pd.Series(y).reset_index(drop=True)
    feat_cols = build_features(d, feature_set)
    X, dropped_sanitize = sanitize_X(d[feat_cols])
    feat_cols = list(X.columns)
    if not feat_cols: raise ValueError('No usable features after sanitization')
    groups = make_groups(d, protocol)
    if groups is None:
        splits = list(KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed).split(X,y))
    else:
        groups = groups.reset_index(drop=True)
        nsp = min(args.n_splits, int(groups.nunique()))
        if nsp < 2: raise ValueError('Not enough groups')
        splits = list(GroupKFold(n_splits=nsp).split(X,y,groups=groups))
    prefix = f'{safe_slug(target)}__{safe_slug(protocol)}__{safe_slug(model_name)}__{safe_slug(feature_set)}'
    outdir = Path(args.outdir); sdir = outdir/'shap_by_run'; sdir.mkdir(parents=True, exist_ok=True)
    leakage = {'target_in_X': target in feat_cols, 'any_property_col_in_X': any(c in feat_cols for c in TARGET_COLS), 'property_cols_in_X': [c for c in TARGET_COLS if c in feat_cols]}
    mets=[]; shaps=[]
    for fold,(tr,te) in enumerate(splits, start=1):
        if args.max_folds and fold > args.max_folds: break
        Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y.iloc[tr], y.iloc[te]
        Xtr_np, Xte_np = impute_arrays(Xtr, Xte)
        model = make_model(model_name, args.seed+fold); model.fit(Xtr_np, ytr)
        pred = model.predict(Xte_np)
        mets.append({'target':target,'target_used':target_used,'protocol':protocol,'model':model_name,'feature_set':feature_set,'fold':fold,'n_train':len(tr),'n_test':len(te),'n_features':len(feat_cols),'r2':r2_score(yte,pred) if len(yte)>1 else np.nan,'mae':mean_absolute_error(yte,pred),'rmse':rmse(yte,pred), **leakage})
        try:
            sd = run_shap(model, model_name, Xtr_np, Xte_np, feat_cols, args.max_background, args.max_explain, args.seed+fold)
            sd.insert(0,'fold',fold); shaps.append(sd)
        except Exception as e:
            print(f'[WARN] SHAP failed for {prefix} fold {fold}: {e}', file=sys.stderr)
    if shaps:
        sdf = pd.concat(shaps, ignore_index=True)
        sdf.insert(0,'target',target); sdf.insert(1,'protocol',protocol); sdf.insert(2,'model',model_name); sdf.insert(3,'feature_set',feature_set)
        sdf.to_csv(sdir/f'{prefix}__fold_shap.csv', index=False)
        g = sdf.groupby(['target','protocol','model','feature_set','feature'], as_index=False).agg(mean_abs_shap=('mean_abs_shap','mean'), sd_abs_shap=('mean_abs_shap','std'), mean_signed_shap=('mean_signed_shap','mean'), n_folds=('fold','nunique')).sort_values('mean_abs_shap', ascending=False)
        g.to_csv(sdir/f'{prefix}__global_shap.csv', index=False)
    else:
        g = pd.DataFrame()
    audit = {'target':target,'target_used':target_used,'protocol':protocol,'model':model_name,'feature_set':feature_set,'n_rows':len(d),'n_features':len(feat_cols),'features':feat_cols,'dropped_all_nan_after_sanitize':dropped_sanitize,'leakage':leakage,'max_background':args.max_background,'max_explain':args.max_explain}
    (sdir/f'{prefix}__audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return pd.DataFrame(mets), g

def make_plots(outdir):
    import matplotlib.pyplot as plt
    p = outdir/'shap_global_summary.csv'
    if not p.exists(): return
    df = pd.read_csv(p); figdir=outdir/'figures'; figdir.mkdir(exist_ok=True)
    for key,g in df.groupby(['target','protocol','model','feature_set']):
        top = g.sort_values('mean_abs_shap', ascending=False).head(20).iloc[::-1]
        if top.empty: continue
        plt.figure(figsize=(8, max(4,0.28*len(top))))
        plt.barh(top['feature'], top['mean_abs_shap'])
        plt.xlabel('Mean |SHAP value|'); plt.title(' | '.join(map(str,key)))
        plt.tight_layout(); plt.savefig(figdir/('shap_bar__'+'__'.join(safe_slug(x) for x in key)+'.png'), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--outdir', default='shap_outputs_v3')
    ap.add_argument('--properties', nargs='+', default=['density_g_cm3','viscosity_mpa_s','conductivity_ms_cm','surface_tension_mn_m','refractive_index'])
    ap.add_argument('--protocols', nargs='+', default=['pair_group','pair_ratio_group'], choices=['random','pair_group','pair_ratio_group','leave_hba_out','leave_hbd_out'])
    ap.add_argument('--models', nargs='+', default=['ExtraTrees'], choices=['ExtraTrees','HistGB'])
    ap.add_argument('--feature-set', default='descriptors_ratio_temp', choices=['descriptors_only','descriptors_ratio','descriptors_temp','descriptors_ratio_temp','full_safe'])
    ap.add_argument('--n-splits', type=int, default=5); ap.add_argument('--max-folds', type=int, default=2)
    ap.add_argument('--max-background', type=int, default=80); ap.add_argument('--max-explain', type=int, default=120); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-log-viscosity', dest='log_viscosity', action='store_false'); ap.add_argument('--make-plots', action='store_true')
    args = ap.parse_args();
    if args.max_folds == 0: args.max_folds = None
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = add_engineered(pd.read_csv(args.dataset))
    allm=[]; alls=[]
    for target in args.properties:
        if target not in df.columns:
            print(f'[WARN] missing target: {target}', file=sys.stderr); continue
        for protocol in args.protocols:
            for model in args.models:
                print(f'[RUN] target={target} protocol={protocol} model={model} feature_set={args.feature_set}')
                try:
                    m,s = run_one(df,target,protocol,model,args.feature_set,args); allm.append(m)
                    if not s.empty: alls.append(s)
                except Exception as e:
                    print(f'[ERROR] failed target={target} protocol={protocol} model={model}: {e}', file=sys.stderr)
    if allm:
        met = pd.concat(allm, ignore_index=True); met.to_csv(outdir/'shap_model_metrics_by_fold.csv', index=False)
        met.groupby(['target','target_used','protocol','model','feature_set'], as_index=False).agg(n_folds=('fold','nunique'), n_features=('n_features','max'), r2_mean=('r2','mean'), r2_sd=('r2','std'), mae_mean=('mae','mean'), rmse_mean=('rmse','mean'), target_in_X=('target_in_X','max'), any_property_col_in_X=('any_property_col_in_X','max')).to_csv(outdir/'shap_model_metrics_summary.csv', index=False)
    if alls:
        sh = pd.concat(alls, ignore_index=True); sh.to_csv(outdir/'shap_global_summary.csv', index=False)
        sh.groupby(['target','protocol','model','feature_set']).head(25).to_csv(outdir/'shap_top25_by_run.csv', index=False)
    if args.make_plots: make_plots(outdir)
    print(f'Done. Outputs written to: {outdir}')
if __name__ == '__main__': main()
