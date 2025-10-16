# src/preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocess ICU lymphoma dataset:
- Load ingested dataset (parquet/csv)
- Ensure binary outcome (0/1), print prevalence
- Clip outliers for continuous features
- One-hot encode categoricals (drop_first)
- Persist processed feature matrix (pre-imputation, pre-scaling)
- Emit missingness overview and feature schema

Usage:
    python -m src.preprocess --config conf/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# -----------------------
# I/O helpers
# -----------------------
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_ingested_frame(processed_dir: Path) -> pd.DataFrame:
    p_parquet = processed_dir / "s1_data.parquet"
    p_csv = processed_dir / "s1_data.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(
        f"未找到已摄取的数据文件：{p_parquet} 或 {p_csv}。请先运行 `python -m src.ingest`。"
    )


# -----------------------
# Core transforms
# -----------------------
def coerce_binary_outcome(df: pd.DataFrame, outcome_col: str) -> Tuple[pd.DataFrame, Dict]:
    """Map outcome to {0,1} if needed. Returns (df, mapping_meta)."""
    if outcome_col not in df.columns:
        raise KeyError(f"结局列 `{outcome_col}` 不存在。")

    s = df[outcome_col]
    uniq = pd.Series(s.dropna().unique()).astype(str).str.lower().tolist()

    # If already 0/1, leave as is
    if set(pd.unique(s.dropna())) <= {0, 1}:
        mapping = {"applied": False, "note": "already binary 0/1"}
        return df, mapping

    # Common truthy/falsy mappings
    truthy = {"1", "true", "yes", "y", "死亡", "dead", "deceased", "case", "positive", "pos", "2"}
    falsy = {"0", "false", "no", "n", "生存", "alive", "survive", "control", "negative", "neg"}

    out = s.copy()
    applied = False
    try:
        # Try numeric coercion first (handles {1,2} -> map 2->1, 1->0 as a last resort)
        sn = pd.to_numeric(s, errors="coerce")
        sn_uniq = set(sn.dropna().unique())
        if sn_uniq and not (sn_uniq <= {0, 1}):
            # Heuristic: if values are {1,2} or {0,2}
            if sn_uniq <= {1, 2}:
                out = sn.map({1: 0, 2: 1})
                applied = True
            elif sn_uniq <= {0, 2}:
                out = sn.map({0: 0, 2: 1})
                applied = True
        elif sn_uniq <= {0, 1} and not s.equals(sn):
            out = sn
            applied = True
    except Exception:
        pass

    if not applied:
        lower = s.astype(str).str.lower()
        out = lower.map(lambda x: 1 if x in truthy else (0 if x in falsy else np.nan))
        # Only accept if we didn't create conflicts
        if out.dropna().nunique() == 2 or out.dropna().nunique() == 1:
            applied = True

    if not applied:
        raise ValueError(
            f"无法将 `{outcome_col}` 安全映射为二分类 0/1。唯一取值: {sorted(uniq)}。"
        )

    df = df.copy()
    df[outcome_col] = out.astype("float").astype("Int64").astype("float").astype(int)  # safe cast
    mapping = {"applied": True, "original_uniques": uniq}
    return df, mapping


def clip_outliers(df: pd.DataFrame, cols: List[str], q_low: float, q_high: float) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        # numeric coercion to avoid object issues
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lo, hi = df[c].quantile(q_low), df[c].quantile(q_high)
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def missingness_overview(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in df.columns:
        miss = df[col].isna().sum()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_missing": int(miss),
                "pct_missing": float(100.0 * miss / n if n else np.nan),
            }
        )
    ov = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    return ov


def one_hot_encode(
    df: pd.DataFrame, cat_cols: List[str], drop_first: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    if not cat_cols:
        return df.copy(), []
    # ensure category dtype for stability
    df_cat = df.copy()
    for c in cat_cols:
        if c in df_cat.columns:
            df_cat[c] = df_cat[c].astype("category")
    df_enc = pd.get_dummies(df_cat, columns=[c for c in cat_cols if c in df_cat.columns], drop_first=drop_first)
    new_cat_cols = [c for c in df_enc.columns if c not in df.columns]
    return df_enc, new_cat_cols


# -----------------------
# Main pipeline
# -----------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess ICU lymphoma dataset before imputation & scaling.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml", help="Path to YAML config.")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    figures_dir = Path(cfg["output"]["figures"])
    tables_dir = Path(cfg["output"]["tables"])

    for d in (processed_dir, artifacts_dir, figures_dir, tables_dir):
        ensure_dir(d)

    outcome_col = cfg["data"]["outcome_col"]
    positive_label = cfg["data"].get("positive_label", 1)

    # 1) load ingested frame
    print("[step] 读取 ingest 输出数据 ...")
    df = read_ingested_frame(processed_dir)
    print(f"[info] 数据形状: {df.shape}")

    # 2) standardize outcome to binary 0/1
    print("[step] 统一结局列为 0/1 ...")
    df, outcome_map = coerce_binary_outcome(df, outcome_col)
    prev = df[outcome_col].mean()
    print(f"[stat] 结局阳性率 (label={positive_label} 视为阳性)：{prev:.4f} ({prev*100:.2f}%)")
    if outcome_map.get("applied", False):
        print(f"[info] 已对 `{outcome_col}` 做二值化映射。原始取值: {outcome_map.get('original_uniques')}")

    # 3) gather configured features
    cont_cols = list(cfg["features"].get("continuous", []) or [])
    cat_cols = list(cfg["features"].get("categorical", []) or [])
    comorb_cols = list(cfg["features"].get("comorbidities", []) or [])

    # sanity: coerce comorbidities to numeric (0/1) if they look like booleans/ints
    for c in comorb_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) outlier clipping (continuous)
    q_low, q_high = cfg["preprocess"]["outlier_clip_quantiles"]
    print(f"[step] 连续变量分位数截尾: q_low={q_low}, q_high={q_high}")
    df = clip_outliers(df, cont_cols, q_low, q_high)

    # 5) one-hot encode categoricals
    enc_mode = cfg["preprocess"].get("categorical_encoding", "onehot_drop_first")
    if enc_mode not in ("onehot_drop_first",):
        print(f"[warn] 未实现的分类编码模式: {enc_mode}，将回退到 onehot_drop_first。", file=sys.stderr)
        enc_mode = "onehot_drop_first"

    print("[step] 分类变量独热编码（drop_first=True） ...")
    df_enc, new_cat_cols = one_hot_encode(df, cat_cols, drop_first=True)

    # 6) build feature matrix order
    # continuous + comorbidities + expanded categoricals (new dummies)
    feature_cols = [c for c in cont_cols if c in df_enc.columns] \
                   + [c for c in comorb_cols if c in df_enc.columns] \
                   + [c for c in df_enc.columns if c not in cont_cols + comorb_cols and c != outcome_col and any(c.startswith(cc + "_") for cc in cat_cols)]

    # Keep only relevant columns + outcome
    keep_cols = feature_cols + [outcome_col]
    Xy = df_enc.loc[:, [c for c in keep_cols if c in df_enc.columns]].copy()

    # 7) missingness overview
    print("[step] 生成缺失概览 ...")
    miss_ov = missingness_overview(Xy)
    miss_path = processed_dir / "missingness_overview.csv"
    miss_ov.to_csv(miss_path, index=False)
    print(f"[ok] 缺失概览已导出: {miss_path}")

    # print top-10 missing variables for quick check
    top_miss = miss_ov.sort_values("pct_missing", ascending=False).head(10)
    print("[stat] 缺失率前10变量：")
    print(top_miss.to_string(index=False))

    # 8) save processed matrix (pre-imputation, pre-scaling)
    out_parquet = processed_dir / "features_processed.parquet"
    try:
        Xy.to_parquet(out_parquet, index=False)
        print(f"[ok] 预处理设计矩阵已写入: {out_parquet}")
    except Exception as e:
        csv_fallback = out_parquet.with_suffix(".csv")
        Xy.to_csv(csv_fallback, index=False)
        print(f"[warn] Parquet 写入失败（{e}），已回退为 CSV: {csv_fallback}", file=sys.stderr)

    # 9) persist feature schema
    schema = {
        "outcome_col": outcome_col,
        "continuous": [c for c in cont_cols if c in Xy.columns],
        "categorical_raw": [c for c in cat_cols if c in df.columns],
        "categorical_expanded": [c for c in feature_cols if c not in cont_cols + comorb_cols],
        "comorbidities": [c for c in comorb_cols if c in Xy.columns],
        "feature_order": feature_cols,
        "n_samples": int(len(Xy)),
        "n_features": int(len(feature_cols)),
    }
    schema_path = Path(cfg["project"]["artifacts_dir"]) / "feature_schema.json"
    ensure_dir(schema_path.parent)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"[ok] 特征架构已写入: {schema_path}")

    print("[note] 当前矩阵仍包含缺失值（按配置将由多重插补或简单插补处理）；缩放在插补之后执行。")
    print("[done] Preprocess 完成。后续可运行：`make mi` 或 `make train MODEL=catboost`。")


if __name__ == "__main__":
    main()



# python -m src.preprocess --config conf/config.yaml
