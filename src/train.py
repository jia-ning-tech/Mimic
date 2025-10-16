# src/train.py
# -*- coding: utf-8 -*-
"""
Train & evaluate models on ICU lymphoma dataset with MI ensemble.

- Load MI runs (outputs/mi_runs/index.json)
- Optionally restrict to selected_features.json
- Train one model per MI run -> get test probabilities -> average across M
- Optionally apply CalibratedClassifierCV (isotonic, cv=KFOLDS) on train only (version-compatible)
- Save metrics, per-m AUC table, ROC curve, calibration plot, and fitted models
- Heartbeat logs during long fits; tqdm progress over M runs
"""

from __future__ import annotations

import argparse
import json
import inspect
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import make_model


# ========= Utilities =========
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_mi_index(mi_dir: Path) -> Dict:
    idx_path = mi_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"未找到 MI 索引：{idx_path}；请先运行 `python -m src.multiple_imputation`。")
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_selected_features(artifacts_dir: Path) -> Optional[List[str]]:
    sel_path = artifacts_dir / "selected_features.json"
    if not sel_path.exists():
        return None
    with open(sel_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    feats = obj.get("selected_features")
    if isinstance(feats, list) and len(feats) > 0:
        return feats
    return None


def load_best_params(outputs_dir: Path, model_name: str) -> Optional[Dict]:
    p = outputs_dir / f"best_params_{model_name}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("best_params", None)


def get_class_weight(cfg: Dict) -> Optional[Dict[int, float]]:
    imb = cfg.get("imbalance", {}) or {}
    if imb.get("use_class_weight", True):
        return {0: 1.0, 1: float(imb.get("positive_class_weight", 1.0))}
    return None


def class_weight_to_sample_weight(y: np.ndarray, cw: Dict[int, float]) -> np.ndarray:
    w0 = float(cw.get(0, 1.0))
    w1 = float(cw.get(1, 1.0))
    return np.where(y == 1, w1, w0).astype("float32")


def get_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        ps = est.predict_proba(X)
        return ps[:, 1] if ps.ndim == 2 else ps
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return s  # ROC AUC 对单调变换不变
    preds = est.predict(X)
    return preds.astype(float)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, path_png: Path, title: str) -> None:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    ensure_dir(path_png.parent)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, path_png: Path, title: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="upper left")
    ensure_dir(path_png.parent)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()


# ========= Calibrator (version compatibility) =========
def make_calibrator(base_estimator, method: str, cv: int) -> CalibratedClassifierCV:
    """
    Create CalibratedClassifierCV compatible with both old (base_estimator=...)
    and new (estimator=...) sklearn signatures.
    """
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    params = list(sig.parameters.keys())
    try:
        if "estimator" in params:
            return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
        else:
            return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)
    except TypeError:
        # Fallback once more using the other name
        try:
            return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)
        except TypeError:
            return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)


# ========= Heartbeat =========
class Heartbeat:
    """
    Periodically print a heartbeat line while a long operation is running.
    Usage:
        with Heartbeat(prefix="[hb] m=1 training", interval=5.0):
            est.fit(...)
    """
    def __init__(self, prefix: str, interval: float = 5.0):
        self.prefix = prefix
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._t = None
        self._t0 = None

    def _runner(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._t0
            print(f"{self.prefix} ... {elapsed:.1f}s elapsed", flush=True)

    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._runner, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.1)


# ========= Robust CSV append helpers =========
def append_metrics_row(sink_csv: Path, row_dict: Dict, key_col: str = "model", key_val: Optional[str] = None):
    """
    Append/replace a row into sink_csv robustly:
    - If file not exist: create
    - If file exists but missing key_col: just append (align columns)
    - If key_col present: drop rows with key==key_val, then append
    """
    new_df = pd.DataFrame([row_dict])
    if not sink_csv.exists():
        new_df.to_csv(sink_csv, index=False)
        return

    try:
        old = pd.read_csv(sink_csv)
    except Exception:
        # Corrupted or incompatible: overwrite with new
        new_df.to_csv(sink_csv, index=False)
        return

    # Align columns (union)
    all_cols = list(dict.fromkeys(list(old.columns) + list(new_df.columns)))
    old = old.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    if key_val is not None and key_col in old.columns:
        old = old[old[key_col] != key_val]

    pd.concat([old, new_df], axis=0, ignore_index=True).to_csv(sink_csv, index=False)


# ========= Main =========
def main(argv=None):
    # tqdm optional
    try:
        from tqdm import tqdm
        has_tqdm = True
    except Exception:
        def tqdm(x, **kwargs):
            return x
        has_tqdm = False

    parser = argparse.ArgumentParser(description="Train & evaluate with MI ensemble.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml", help="Path to config.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name (e.g., random_forest).")
    parser.add_argument("--use_m_all", action="store_true", help="Use all MI runs (default).")
    parser.add_argument("--m_only", type=int, default=None, help="Use only specific m index (1-based).")
    parser.add_argument("--no_calib", action="store_true", help="Force disable calibration regardless of config.")
    parser.add_argument("--heartbeat", type=float, default=5.0, help="Heartbeat interval in seconds (default 5s).")
    args = parser.parse_args(argv)

    # ---- config & paths
    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs")
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures"))
    tables_dir = Path(cfg.get("output", {}).get("tables", "outputs/tables"))
    ensure_dir(outputs_dir); ensure_dir(figures_dir); ensure_dir(tables_dir)

    # artifacts dir with default fallback
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))
    ensure_dir(artifacts_dir)

    # MI dir with default fallback
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    # feature subset
    selected = read_selected_features(artifacts_dir)

    # calibration flag
    do_calib = bool(cfg.get("evaluation", {}).get("calibration", True)) and (not args.no_calib)
    kfolds = int(cfg.get("models", {}).get("optimization", {}).get("cv_folds", 5))

    # class weight
    cw = get_class_weight(cfg)

    # load best params (if any)
    best_params = load_best_params(outputs_dir, args.model) or {}

    # ---- choose MI runs
    if args.m_only is not None:
        idx = max(1, int(args.m_only))
        if idx > len(m_paths):
            raise ValueError(f"--m_only={idx} 超出范围（共有 M={len(m_paths)}）。")
        use_paths = [m_paths[idx - 1]]
    else:
        use_paths = m_paths  # 默认用全部 M
    M = len(use_paths)
    print(f"[info] 训练模型: {args.model}，使用 M={M} 个插补集成（{'校准' if do_calib else '不校准'}，K={kfolds}）")

    # ---- containers for ensemble
    per_m_auc: List[float] = []
    test_probas_accum = None  # 用于最终平均
    y_test_ref = None

    models_dir = outputs_dir / "models"
    ensure_dir(models_dir)

    # ---- loop over each MI run (with progress)
    for i, p in enumerate(tqdm(use_paths, desc="MI runs", unit="m"), start=1):
        art = joblib.load(p)
        feat_order = art["feature_order"]
        X_tr_df = pd.DataFrame(art["X_train"], columns=feat_order)
        X_te_df = pd.DataFrame(art["X_test"], columns=feat_order)
        y_tr = np.asarray(art["y_train"]).astype(int)
        y_te = np.asarray(art["y_test"]).astype(int)

        if y_test_ref is None:
            y_test_ref = y_te
        else:
            if len(y_test_ref) != len(y_te):
                raise RuntimeError("不同 m 的测试集样本数不一致，请检查 split 索引。")

        # restrict to selected features if present
        if selected:
            use_cols = [c for c in selected if c in X_tr_df.columns]
            if not use_cols:
                raise RuntimeError("selected_features.json 中的列在训练矩阵中不存在。")
            X_tr_df = X_tr_df[use_cols]
            X_te_df = X_te_df[use_cols]

        X_tr = X_tr_df.values.astype("float32")
        X_te = X_te_df.values.astype("float32")

        # build estimator
        est = make_model(args.model, cfg, custom_params=best_params.copy())

        # sample_weight fallback for models without class_weight
        fit_kwargs = {}
        if cw is not None:
            if hasattr(est, "class_weight") or ("class_weight" in est.get_params().keys()):
                pass  # 已在 make_model 设置
            else:
                fit_kwargs["sample_weight"] = class_weight_to_sample_weight(y_tr, cw)

        # optional calibration (on train only, K-fold internal)
        if do_calib:
            base = est
            est = make_calibrator(base_estimator=base, method="isotonic", cv=kfolds)

        # fit with heartbeat
        with Heartbeat(prefix=f"[hb] m={i} training", interval=float(args.heartbeat)):
            est.fit(X_tr, y_tr, **fit_kwargs)

        # save fitted model per m
        model_path = models_dir / f"{args.model}_m{i:02d}.joblib"
        joblib.dump(est, model_path)

        # proba on test
        proba_te = get_proba(est, X_te)
        # normalize to [0,1] if decision_function returned
        if proba_te.min() < 0 or proba_te.max() > 1:
            s = proba_te
            proba_te = 1.0 / (1.0 + np.exp(-s))

        auc_m = roc_auc_score(y_te, proba_te)
        per_m_auc.append(float(auc_m))
        print(f"[m={i}] test AUC = {auc_m:.4f}")

        if test_probas_accum is None:
            test_probas_accum = proba_te.astype("float64")
        else:
            test_probas_accum += proba_te.astype("float64")

    # ---- ensemble averaging
    y_te = y_test_ref
    yhat_prob = (test_probas_accum / M).astype("float64")

    # ---- metrics
    auc = roc_auc_score(y_te, yhat_prob)
    thr = 0.5  # 默认0.5阈值（阈值扫描另有模块）
    yhat = (yhat_prob >= thr).astype(int)

    metrics = {
        "model": args.model,
        "M": M,
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_te, yhat)),
        "f1": float(f1_score(y_te, yhat)),
        "precision": float(precision_score(y_te, yhat)),
        "recall": float(recall_score(y_te, yhat)),
        "brier": float(brier_score_loss(y_te, yhat_prob)),
    }

    # ---- save metrics & per-m AUC table (robust)
    outputs_dir = Path("outputs")
    model_metrics_csv = outputs_dir / f"metrics_test_{args.model}.csv"
    metrics_csv = outputs_dir / "metrics_test.csv"
    pd.DataFrame([metrics]).to_csv(model_metrics_csv, index=False)

    # robust append/replace by 'model'
    append_metrics_row(metrics_csv, metrics, key_col="model", key_val=args.model)

    # per-m auc + ensemble auc
    auc_tab = pd.DataFrame({
        "model": [args.model]*M,
        "m_index": list(range(1, M+1)),
        "auc": per_m_auc
    })
    # 0 行为 ensemble
    auc_tab.loc[len(auc_tab)] = [args.model, 0, auc]
    auc_csv = outputs_dir / "model_auc_test.csv"
    # robust append (union columns); here we just append (allow multiple models)
    if auc_csv.exists():
        try:
            old = pd.read_csv(auc_csv)
            all_cols = list(dict.fromkeys(list(old.columns) + list(auc_tab.columns)))
            old = old.reindex(columns=all_cols)
            auc_tab = auc_tab.reindex(columns=all_cols)
            combined = pd.concat([old, auc_tab], axis=0, ignore_index=True)
            combined.to_csv(auc_csv, index=False)
        except Exception:
            auc_tab.to_csv(auc_csv, index=False)
    else:
        auc_tab.to_csv(auc_csv, index=False)

    # ---- plots
    roc_png = outputs_dir / f"roc_test_{args.model}.png"
    plot_roc(y_te, yhat_prob, roc_png, f"ROC (test) - {args.model} (M={M})")
    roc_alias = outputs_dir / "roc_test.png"
    try:
        # 再生成一份通用名，方便比较不同模型时覆盖
        plot_roc(y_te, yhat_prob, roc_alias, f"ROC (test) - {args.model} (M={M})")
    except Exception:
        pass

    if do_calib:
        calib_png = outputs_dir / "calibration_test.png"
        plot_calibration(y_te, yhat_prob, calib_png, f"Calibration (test) - {args.model} (M={M})")

    # ---- ensemble manifest
    models_dir = outputs_dir / "models"
    ensure_dir(models_dir)
    ensemble_json = models_dir / f"{args.model}_ensemble.json"
    with open(ensemble_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "M": M,
                "mi_paths": [str(p) for p in use_paths],
                "per_m_auc": per_m_auc,
                "ensemble_auc": float(auc),
                "selected_features": selected,
                "best_params_used": best_params,
                "calibration": do_calib,
                "kfolds": kfolds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- final log
    print("[ok] 评估完成：")
    for k, v in metrics.items():
        if k == "model":
            continue
        print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
    print(f"[save] 指标: {model_metrics_csv}")
    print(f"[save] 汇总: {metrics_csv}")
    print(f"[save] AUC表: {auc_csv}")
    print(f"[save] ROC图: {roc_png}")
    if do_calib:
        print(f"[save] 校准图: {outputs_dir / 'calibration_test.png'}")
    print(f"[save] 模型工件: {models_dir}/*.joblib, {ensemble_json}")


if __name__ == "__main__":
    # Local imports required by metrics functions (avoid circular imports at top)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
    main()



# python -m src.train --config conf/config.yaml --model random_forest --heartbeat 2
