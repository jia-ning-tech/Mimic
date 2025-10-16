# -*- coding: utf-8 -*-
"""
Multiple Imputation (MICE) pipeline for ICU lymphoma ML reproduction.

- Loads preprocessed feature matrix (pre-imputation, pre-scaling)
- Uses train/test split indices cached by src.ingest
- Runs M imputations with IterativeImputer (BayesianRidge or ExtraTreesRegressor)
- Fits scaler on train (continuous columns only) AFTER imputation (to avoid leakage)
- Saves each m-run as a joblib artifact and writes an index.json for downstream training
- Shows progress bar for M runs and heartbeat logs during long fit/transform steps

Usage:
    python -m src.multiple_imputation --config conf/config.yaml [--m 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# -----------------------
# Utils / I/O
# -----------------------
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_feature_schema(artifacts_dir: Path) -> Dict:
    schema_path = artifacts_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到特征架构: {schema_path}，请先运行 `python -m src.preprocess`。")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_processed_matrix(processed_dir: Path) -> pd.DataFrame:
    p = processed_dir / "features_processed.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p_csv = processed_dir / "features_processed.csv"
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(
        f"未找到预处理设计矩阵：{p} 或 {p_csv}。请先运行 `python -m src.preprocess`。"
    )


def read_split_indices(split_cache: Path) -> Dict:
    if not split_cache.exists():
        raise FileNotFoundError(f"未找到分割索引缓存: {split_cache}。请先运行 `python -m src.ingest --split-only`。")
    return joblib.load(split_cache)


def pick_scaler(name: str):
    name = (name or "standard").lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"未知 scaler: {name}")


def make_imputer(estimator_name: str, max_iter: int, sample_posterior: bool, random_state: int) -> IterativeImputer:
    est = estimator_name.lower()
    if est in ("bayesian_ridge", "bayes", "br"):
        base = BayesianRidge()
        # sample_posterior 仅在 BayesianRidge 生效，带来多重插补的随机性
        return IterativeImputer(
            estimator=base,
            max_iter=max_iter,
            sample_posterior=bool(sample_posterior),
            random_state=random_state,
            skip_complete=False,
            add_indicator=False,
            tol=1e-3,
            verbose=0,
        )
    elif est in ("random_forest", "extratrees", "et"):
        # 使用 ExtraTreesRegressor 近似 RF，速度更快、鲁棒性好
        base = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )
        return IterativeImputer(
            estimator=base,
            max_iter=max_iter,
            sample_posterior=False,       # 对基学习器为树时无效
            random_state=random_state,
            skip_complete=False,
            add_indicator=False,
            tol=1e-3,
            verbose=0,
        )
    else:
        raise ValueError(f"不支持的 imputer estimator: {estimator_name}")


def to_float32(df: pd.DataFrame) -> pd.DataFrame:
    # 所有特征转 float32，节省内存（29GB 服务器更稳）
    for c in df.columns:
        if c != "__y__":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


# -----------------------
# Heartbeat helper
# -----------------------
class Heartbeat:
    """
    A simple heartbeat that prints a status line every `interval` seconds while a long task is running.
    It runs in a daemon thread and stops when exiting the context.
    """

    def __init__(self, label: str, interval: float = 5.0):
        self.label = label
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = time.time()
        tick = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - t0)
            # 使用 \r 就地刷新同一行；避免刷屏
            sys.stdout.write(f"\r[hb] {self.label} ... {elapsed}s")
            sys.stdout.flush()
            tick += 1
            self._stop.wait(self.interval)
        # 清掉心跳行，换行
        sys.stdout.write("\r" + " " * (len(self.label) + 24) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1)


# -----------------------
# Core
# -----------------------
@dataclass
class CfgPaths:
    processed_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    split_cache: Path
    mi_dir: Path


def build_paths(cfg: Dict) -> CfgPaths:
    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    figures_dir = Path(cfg["output"]["figures"])
    tables_dir = Path(cfg["output"]["tables"])
    split_cache = Path(cfg["split"]["index_cache"])
    mi_dir = Path(cfg["missing_data"]["mice"]["mice_output_dir"])
    for d in (processed_dir, artifacts_dir, figures_dir, tables_dir, mi_dir):
        ensure_dir(d)
    return CfgPaths(processed_dir, artifacts_dir, figures_dir, tables_dir, split_cache, mi_dir)


def run_single_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cont_cols: List[str],
    imputer_cfg: Dict,
    preprocess_cfg: Dict,
    base_seed: int,
    m_idx: int,
    heartbeat_sec: float = 5.0,
) -> Dict:
    # 1) build imputer
    est_name = imputer_cfg.get("estimator", "bayesian_ridge")
    max_iter = int(imputer_cfg.get("max_iter", 15))
    sample_posterior = bool(imputer_cfg.get("sample_posterior", False))
    fit_scope = imputer_cfg.get("fit_scope", "train_only")
    rs = int(imputer_cfg.get("random_state", 42)) + int(m_idx)

    imputer = make_imputer(estimator_name=est_name, max_iter=max_iter, sample_posterior=sample_posterior, random_state=rs)

    # 2) fit/transform
    t0 = time.time()
    if fit_scope == "full_sample":
        full = pd.concat([X_train, X_test], axis=0)
        print(f"[m={m_idx}] 拟合插补器（{est_name}, max_iter={max_iter}）于全样本，样本数={len(full)}，特征数={full.shape[1]}")
        with Heartbeat(label=f"m={m_idx} imputer.fit(full)", interval=heartbeat_sec):
            imputer.fit(full)
        t1 = time.time()
        print(f"[m={m_idx}] imputer.fit 完成，用时 {t1 - t0:.1f}s")
        with Heartbeat(label=f"m={m_idx} imputer.transform(full)", interval=heartbeat_sec):
            full_imp = pd.DataFrame(imputer.transform(full), columns=full.columns, index=full.index)
        t2 = time.time()
        print(f"[m={m_idx}] imputer.transform 完成，用时 {t2 - t1:.1f}s")
        X_train_imp = full_imp.loc[X_train.index].copy()
        X_test_imp = full_imp.loc[X_test.index].copy()
    else:
        print(f"[m={m_idx}] 拟合插补器（{est_name}, max_iter={max_iter}）于训练集，n_train={len(X_train)}, n_test={len(X_test)}")
        with Heartbeat(label=f"m={m_idx} imputer.fit(train)", interval=heartbeat_sec):
            imputer.fit(X_train)
        t1 = time.time()
        print(f"[m={m_idx}] imputer.fit 完成，用时 {t1 - t0:.1f}s")
        with Heartbeat(label=f"m={m_idx} transform(train)", interval=heartbeat_sec):
            X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
        t2 = time.time()
        print(f"[m={m_idx}] transform(train) 完成，用时 {t2 - t1:.1f}s")
        with Heartbeat(label=f"m={m_idx} transform(test)", interval=heartbeat_sec):
            X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        t3 = time.time()
        print(f"[m={m_idx}] transform(test) 完成，用时 {t3 - t2:.1f}s")

    # 3) scaling (continuous only), fit on train then transform both
    scaler_state = {}
    if preprocess_cfg.get("scale_continuous", True):
        scaler_name = preprocess_cfg.get("scaler", "standard")
        scaler = pick_scaler(scaler_name)
        cont_in_data = [c for c in cont_cols if c in X_train_imp.columns]
        if cont_in_data:
            print(f"[m={m_idx}] 缩放连续变量（{scaler_name}），列数={len(cont_in_data)}")
            with Heartbeat(label=f"m={m_idx} scaler.fit(train[cont])", interval=max(2.0, heartbeat_sec)):
                scaler.fit(X_train_imp[cont_in_data])
            X_train_imp.loc[:, cont_in_data] = scaler.transform(X_train_imp[cont_in_data])
            X_test_imp.loc[:, cont_in_data] = scaler.transform(X_test_imp[cont_in_data])
            scaler_state = {
                "scaler": scaler_name,
                "fitted_on": "train",
                "columns": cont_in_data,
                "mean_": getattr(scaler, "mean_", None) if hasattr(scaler, "mean_") else None,
                "scale_": getattr(scaler, "scale_", None) if hasattr(scaler, "scale_") else None,
            }

    # 4) sanity checks
    nan_train = int(np.isnan(X_train_imp.values).sum())
    nan_test = int(np.isnan(X_test_imp.values).sum())
    if nan_train or nan_test:
        print(f"[warn][m={m_idx}] 插补后仍存在 NaN：train={nan_train}, test={nan_test}")

    meta = {
        "m_index": int(m_idx),
        "estimator": est_name,
        "max_iter": max_iter,
        "sample_posterior": sample_posterior,
        "fit_scope": fit_scope,
        "random_state": rs,
        "nan_after_impute_train": nan_train,
        "nan_after_impute_test": nan_test,
        "dur_total_sec": round(time.time() - t0, 1),
    }
    return {
        "X_train": X_train_imp.astype("float32"),
        "X_test": X_test_imp.astype("float32"),
        "scaler_state": scaler_state,
        "imputer_state": {
            "estimator": est_name,
            "max_iter": max_iter,
            "sample_posterior": sample_posterior,
            "fit_scope": fit_scope,
            "random_state": rs,
        },
        "meta": meta,
    }


# -----------------------
# Entry
# -----------------------
@dataclass
class CfgPaths:
    processed_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    tables_dir: Path
    split_cache: Path
    mi_dir: Path


def build_paths(cfg: Dict) -> CfgPaths:
    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    figures_dir = Path(cfg["output"]["figures"])
    tables_dir = Path(cfg["output"]["tables"])
    split_cache = Path(cfg["split"]["index_cache"])
    mi_dir = Path(cfg["missing_data"]["mice"]["mice_output_dir"])
    for d in (processed_dir, artifacts_dir, figures_dir, tables_dir, mi_dir):
        ensure_dir(d)
    return CfgPaths(processed_dir, artifacts_dir, figures_dir, tables_dir, split_cache, mi_dir)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Multiple Imputation (MICE) with train-only fit and scaling (with progress/heartbeat).")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml", help="Path to YAML config.")
    parser.add_argument("--model", "-mname", type=str, default=None, help="Model name (兼容命令行；此处不使用，仅透传至元数据).")
    parser.add_argument("--m", type=int, default=None, help="覆盖 config 中的插补次数 M。")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    paths = build_paths(cfg)

    # Load data & schema
    df = read_processed_matrix(paths.processed_dir)
    schema = read_feature_schema(paths.artifacts_dir)

    outcome_col = schema["outcome_col"]
    feature_order = schema["feature_order"]
    cont_cols = schema["continuous"]

    # Guard: ensure all needed columns exist
    cols_needed = feature_order + [outcome_col]
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"预处理矩阵缺少列: {missing_cols}。请检查 preprocess 流程。")

    # Build X/y
    X = df[feature_order].copy()
    y = df[outcome_col].astype(int).copy()

    # Load split indices
    split_obj = read_split_indices(Path(cfg["split"]["index_cache"]))
    tr_idx = np.array(split_obj["train_idx"])
    te_idx = np.array(split_obj["test_idx"])

    X_train = to_float32(X.iloc[tr_idx].reset_index(drop=True))
    X_test  = to_float32(X.iloc[te_idx].reset_index(drop=True))
    y_train = y.iloc[tr_idx].reset_index(drop=True)
    y_test  = y.iloc[te_idx].reset_index(drop=True)

    # MICE config
    mice_cfg = cfg["missing_data"]["mice"]
    preprocess_cfg = cfg["preprocess"]
    base_seed = int(mice_cfg.get("random_state", 42))
    M = int(args.m if args.m is not None else mice_cfg.get("m", 20))

    # Heartbeat interval（优先环境变量，其次config，默认5秒）
    hb_env = os.environ.get("MI_HEARTBEAT_SEC")
    heartbeat_sec = float(hb_env) if hb_env is not None else float(mice_cfg.get("heartbeat_sec", 5.0))

    print(f"[info] 开始多重插补：M={M}, estimator={mice_cfg.get('estimator')}, fit_scope={mice_cfg.get('fit_scope')}, heartbeat={heartbeat_sec}s")

    runs_meta: List[Dict] = []
    ensure_dir(paths.mi_dir)

    # 进度条：M 次插补
    for m_idx in tqdm(range(1, M + 1), desc="MI runs", unit="run"):
        out = run_single_imputation(
            X_train=X_train,
            X_test=X_test,
            cont_cols=cont_cols,
            imputer_cfg=mice_cfg,
            preprocess_cfg=preprocess_cfg,
            base_seed=base_seed,
            m_idx=m_idx,
            heartbeat_sec=heartbeat_sec,
        )

        # Save artifact for this m
        art = {
            "X_train": out["X_train"],
            "X_test": out["X_test"],
            "y_train": y_train.values.astype("int32"),
            "y_test": y_test.values.astype("int32"),
            "feature_order": feature_order,
            "scaler_state": out["scaler_state"],
            "imputer_state": out["imputer_state"],
            "meta": {
                **out["meta"],
                "outcome_col": outcome_col,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "model_name": args.model,
            },
        }
        save_path = paths.mi_dir / f"mi_m{m_idx:02d}.joblib"
        joblib.dump(art, save_path)
        print(f"[ok] 已保存 MI 工件：{save_path} （总耗时 {out['meta']['dur_total_sec']}s）")
        runs_meta.append(
            {
                "m_index": m_idx,
                "path": str(save_path),
                **art["meta"],
            }
        )

    # Write index.json
    index_path = paths.mi_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_by": "src.multiple_imputation",
                "M": M,
                "paths": runs_meta,
                "feature_order": feature_order,
                "outcome_col": outcome_col,
                "continuous": cont_cols,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] 多重插补完成，索引文件：{index_path}")
    print("[hint] 后续在 train.py 中读取 mi_runs/index.json 循环训练，并于 evaluate.py 中做 Rubin 合并。")


if __name__ == "__main__":
    main()



# python -m src.multiple_imputation --config conf/config.yaml --m 3
