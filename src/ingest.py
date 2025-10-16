# src/ingest.py
# -*- coding: utf-8 -*-
"""
Ingest raw data, generate cached stratified split indices, and persist basic artifacts.

Usage:
    python -m src.ingest --config conf/config.yaml [--split-only]

Outputs:
    - data_processed/s1_data.parquet
    - data_processed/columns_overview.csv
    - outputs/artifacts/split_idx.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# -----------------------
# Utils
# -----------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # sklearn 受 numpy 种子影响；其他库在各自模块设置


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_columns_overview(df: pd.DataFrame, out_csv: Path) -> None:
    records = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        miss = s.isna().sum()
        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(miss),
            "pct_missing": float(miss / n * 100.0 if n > 0 else np.nan),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        }
        records.append(rec)
    overview = pd.DataFrame.from_records(records).sort_values(["pct_missing", "column"], ascending=[False, True])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(out_csv, index=False)


def read_one_file(path: Path, input_format: str) -> pd.DataFrame:
    if input_format.lower() == "excel":
        # 默认读取第一张表；若需要指定 sheet，可在此扩展
        return pd.read_excel(path)
    elif input_format.lower() == "csv":
        return pd.read_csv(path)
    elif input_format.lower() == "parquet":
        return pd.read_parquet(path)
    elif input_format.lower() == "sql":
        raise NotImplementedError("SQL 输入暂未实现；请先导出为 CSV/Parquet/Excel。")
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")


def load_raw_dataset(cfg: Dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    input_files = cfg["data"]["input_files"]
    input_format = cfg["data"]["input_format"]

    if not input_files:
        raise ValueError("config.data.input_files 为空。")

    dfs = []
    for fn in input_files:
        p = raw_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"未找到原始数据文件: {p}")
        df = read_one_file(p, input_format=input_format)
        dfs.append(df)

    if len(dfs) == 1:
        df_all = dfs[0]
    else:
        # 按行纵向合并
        df_all = pd.concat(dfs, axis=0, ignore_index=True)

    return df_all


def assert_binary_outcome(df: pd.DataFrame, outcome_col: str, positive_label: Optional[int | str]) -> None:
    if outcome_col not in df.columns:
        raise KeyError(f"结局列 `{outcome_col}` 不存在于数据集中。可在 conf/config.yaml 的 data.outcome_col 修改。")

    vals = df[outcome_col].dropna().unique()
    if len(vals) == 0:
        raise ValueError(f"结局列 `{outcome_col}` 全部缺失。")
    # 允许 {0,1} 或 {False,True} 或 {1,2} 等，若不是 {0,1} 则尝试智能映射
    # 此处仅做提示，不强制映射（映射逻辑通常放到 preprocess）
    if len(vals) > 2:
        raise ValueError(f"结局列 `{outcome_col}` 不是二分类。唯一值: {sorted(vals)}")

    if positive_label is not None and positive_label not in vals:
        print(f"[warn] positive_label={positive_label} 不在实际取值 {sorted(vals)} 中，请确认。", file=sys.stderr)


def stratified_split_indices(
    y: pd.Series,
    test_size: float,
    random_state: int,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y,
    )
    return np.array(train_idx), np.array(test_idx)


# -----------------------
# Core routine
# -----------------------
@dataclass
class IngestPaths:
    processed_dir: Path
    cache_dir: Path
    artifacts_dir: Path
    split_cache: Path
    parquet_out: Path
    columns_overview_csv: Path


def build_paths(cfg: Dict) -> IngestPaths:
    processed_dir = Path(cfg["data"]["processed_dir"])
    cache_dir = Path(cfg["project"]["cache_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    split_cache = Path(cfg["split"]["index_cache"])
    parquet_out = processed_dir / "s1_data.parquet"
    columns_overview_csv = processed_dir / "columns_overview.csv"

    for p in (processed_dir, cache_dir, artifacts_dir, split_cache.parent):
        ensure_dir(Path(p))

    return IngestPaths(
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        artifacts_dir=artifacts_dir,
        split_cache=split_cache,
        parquet_out=parquet_out,
        columns_overview_csv=columns_overview_csv,
    )


def persist_split_cache(path: Path, train_idx: np.ndarray, test_idx: np.ndarray, y: pd.Series, cfg: Dict) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": cfg["split"]["random_state"],
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "y_train_pos_rate": float(y.iloc[train_idx].mean()),
        "y_test_pos_rate": float(y.iloc[test_idx].mean()),
        "outcome_col": cfg["data"]["outcome_col"],
    }
    obj = {"train_idx": train_idx, "test_idx": test_idx, "meta": meta}
    joblib.dump(obj, path)
    print(f"[ok] 分割索引已写入: {path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def load_or_create_split(df: pd.DataFrame, cfg: Dict, paths: IngestPaths) -> Dict:
    outcome_col = cfg["data"]["outcome_col"]
    pos_label = cfg["data"].get("positive_label", 1)
    assert_binary_outcome(df, outcome_col, pos_label)
    y = df[outcome_col]

    cache_path = paths.split_cache
    if cache_path.exists():
        obj = joblib.load(cache_path)
        tr, te = obj["train_idx"], obj["test_idx"]
        print(f"[info] 发现已有分割索引缓存，直接复用: {cache_path}")
        print(json.dumps(obj.get("meta", {}), indent=2, ensure_ascii=False))
        return {"train_idx": tr, "test_idx": te, "meta": obj.get("meta", {})}

    # create new
    train_idx, test_idx = stratified_split_indices(
        y=y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        shuffle=cfg["split"].get("shuffle", True),
    )
    persist_split_cache(cache_path, train_idx, test_idx, y, cfg)
    return {"train_idx": train_idx, "test_idx": test_idx}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest raw ICU lymphoma dataset and create cached splits.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml", help="Path to YAML config.")
    parser.add_argument("--split-only", action="store_true", help="Only create/ensure split indices, skip exporting parquet/overview.")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")

    cfg = load_yaml(cfg_path)

    # Seeds & dirs
    seed = int(cfg["project"].get("seed", 42))
    set_global_seed(seed)
    paths = build_paths(cfg)

    # Load raw
    print("[step] 读取原始数据 ...")
    df = load_raw_dataset(cfg)
    print(f"[info] 原始数据形状: {df.shape}")

    # Ensure outcome exists & show quick prevalence
    outcome_col = cfg["data"]["outcome_col"]
    pos_label = cfg["data"].get("positive_label", 1)
    assert_binary_outcome(df, outcome_col, pos_label)

    # Create or reuse split
    print("[step] 检查/生成固定的分层 70/30 分割索引 ...")
    split_obj = load_or_create_split(df, cfg, paths)

    if args.split_only:
        print("[done] 仅生成/复用分割索引（--split-only）。")
        return

    # Persist parquet & overview for downstream steps
    print("[step] 导出数据副本与列概览 ...")
    ensure_dir(paths.processed_dir)
    try:
        df.to_parquet(paths.parquet_out, index=False)
    except Exception as e:
        # 某些环境可能缺少 pyarrow/fastparquet，可降级为 csv
        print(f"[warn] 写入 Parquet 失败（{e}），降级为 CSV。建议安装 pyarrow 以提升性能。", file=sys.stderr)
        csv_fallback = paths.parquet_out.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        print(f"[ok] 已改为写入 CSV: {csv_fallback}")
    save_columns_overview(df, paths.columns_overview_csv)
    print(f"[ok] 列概览: {paths.columns_overview_csv}")

    # Print quick stats
    y = df[outcome_col]
    pos_rate = float((y == pos_label).mean())
    print(f"[stat] 全样本阳性率(=label {pos_label}) = {pos_rate:.4f}  ({pos_rate*100:.2f}%)")
    print(f"[stat] 预期论文锚点：样本量≈1591，院内死亡≈21.5%（用于后续 self-check）")

    print("[done] Ingest 完成。下游可运行：`make train MODEL=catboost` 或 `make mi`。")


if __name__ == "__main__":
    main()



# python -m src.ingest --config conf/config.yaml --split-only
# # 或
# python -m src.ingest --config conf/config.yaml
