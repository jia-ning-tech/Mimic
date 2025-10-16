# src/calibrate.py
# -*- coding: utf-8 -*-
"""
Cross-validated probability calibration on train-only, evaluate on test.

- Load MI artifacts and (optionally) best_params_<model>.json
- For each m: fit base model on train -> predict test (pre-calib)
           -> fit CalibratedClassifierCV on train (K-fold) -> predict test (post-calib)
- Average post-calib test probabilities across M as final ensemble
- Save robust CSV metrics, per-m & ensemble AUC tables, ROC + Calibration plots
- Heartbeat during long fits; tqdm progress over M

Usage:
    python -m src.calibrate --config conf/config.yaml --model random_forest --method isotonic --kfolds 5
"""

from __future__ import annotations

import argparse
import inspect
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, brier_score_loss
)

from src.models import make_model


# ========= I/O utils =========
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
    w0 = float(cw.get(0, 1.0)); w1 = float(cw.get(1, 1.0))
    return np.where(y == 1, w1, w0).astype("float32")


def get_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return s
    y = est.predict(X)
    return y.astype(float)


# ========= plots =========
def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    ensure_dir(out_png.parent); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_calib_compare(y_true: np.ndarray, prob_pre: np.ndarray, prob_post: np.ndarray, out_png: Path, title: str):
    frac_pre, mean_pre = calibration_curve(y_true, prob_pre, n_bins=10, strategy="quantile")
    frac_post, mean_post = calibration_curve(y_true, prob_post, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5))
    plt.plot(mean_pre, frac_pre, 'o-', label="Before")
    plt.plot(mean_post, frac_post, 'o-', label="After")
    plt.plot([0,1],[0,1],'--', label="Perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(title); plt.legend(loc="upper left")
    ensure_dir(out_png.parent); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


# ========= Calibrator compatibility =========
def make_calibrator(base_estimator, method: str, cv: int) -> CalibratedClassifierCV:
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    params = list(sig.parameters.keys())
    try:
        if "estimator" in params:
            return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
        else:
            return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)
    except TypeError:
        try:
            return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)
        except TypeError:
            return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)


# ========= Heartbeat =========
class Heartbeat:
    def __init__(self, prefix: str, interval: float = 5.0):
        self.prefix = prefix; self.interval = max(1.0, float(interval))
        self._stop = threading.Event(); self._t = None; self._t0 = None
    def _runner(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._t0
            print(f"{self.prefix} ... {elapsed:.1f}s elapsed", flush=True)
    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._runner, daemon=True); self._t.start(); return self
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t: self._t.join(timeout=0.1)


# ========= Robust CSV helpers =========
def append_row(sink_csv: Path, row: Dict, key_col: Optional[str] = None, key_val: Optional[str] = None):
    new_df = pd.DataFrame([row])
    if not sink_csv.exists():
        new_df.to_csv(sink_csv, index=False); return
    try:
        old = pd.read_csv(sink_csv)
    except Exception:
        new_df.to_csv(sink_csv, index=False); return
    all_cols = list(dict.fromkeys(list(old.columns) + list(new_df.columns)))
    old = old.reindex(columns=all_cols); new_df = new_df.reindex(columns=all_cols)
    if key_col and (key_col in old.columns) and (key_val is not None):
        old = old[old[key_col] != key_val]
    pd.concat([old, new_df], axis=0, ignore_index=True).to_csv(sink_csv, index=False)


# ========= Main =========
def main(argv=None):
    # tqdm optional
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kwargs):  # type: ignore
            return x

    parser = argparse.ArgumentParser(description="Cross-validated probability calibration.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--method", type=str, default="isotonic", choices=["isotonic", "sigmoid"])
    parser.add_argument("--kfolds", type=int, default=None, help="Calibration CV folds; default from config.")
    parser.add_argument("--heartbeat", type=float, default=5.0, help="Heartbeat seconds.")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables",  "outputs/tables"));  ensure_dir(tables_dir)

    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts")); ensure_dir(artifacts_dir)
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    selected = read_selected_features(artifacts_dir)
    kfolds = int(args.kfolds) if args.kfolds is not None else int(cfg.get("models", {}).get("optimization", {}).get("cv_folds", 5))
    cw = get_class_weight(cfg)
    best_params = load_best_params(outputs_dir, args.model) or {}

    # containers
    per_m_auc_pre: List[float] = []
    per_m_auc_post: List[float] = []
    prob_sum_post = None
    y_test_ref = None

    # loop over M
    for i, p in enumerate(tqdm(m_paths, desc="Calib M", unit="m"), start=1):
        art = joblib.load(p)
        feat_order = art["feature_order"]
        X_tr_df = pd.DataFrame(art["X_train"], columns=feat_order)
        X_te_df = pd.DataFrame(art["X_test"],  columns=feat_order)
        y_tr = np.asarray(art["y_train"]).astype(int)
        y_te = np.asarray(art["y_test"]).astype(int)

        if y_test_ref is None:
            y_test_ref = y_te
        else:
            if len(y_test_ref) != len(y_te):
                raise RuntimeError("不同 m 的测试集样本数不一致，请检查 split 索引。")

        if selected:
            use_cols = [c for c in selected if c in X_tr_df.columns]
            if not use_cols:
                raise RuntimeError("selected_features.json 中的列在训练/测试矩阵中不存在。")
            X_tr_df = X_tr_df[use_cols]; X_te_df = X_te_df[use_cols]

        X_tr = X_tr_df.values.astype("float32")
        X_te = X_te_df.values.astype("float32")

        # base estimator
        base = make_model(args.model, cfg, custom_params=best_params.copy())
        fit_kwargs = {}
        if cw is not None:
            if hasattr(base, "class_weight") or ("class_weight" in base.get_params().keys()):
                pass
            else:
                fit_kwargs["sample_weight"] = class_weight_to_sample_weight(y_tr, cw)

        # --- fit base (train only) with heartbeat
        with Heartbeat(prefix=f"[hb] m={i} base.fit", interval=float(args.heartbeat)):
            base.fit(X_tr, y_tr, **fit_kwargs)

        # pre-calib proba on test
        prob_pre = get_proba(base, X_te)
        if prob_pre.min() < 0 or prob_pre.max() > 1:
            prob_pre = 1.0 / (1.0 + np.exp(-prob_pre))
        auc_pre = roc_auc_score(y_te, prob_pre)
        per_m_auc_pre.append(float(auc_pre))
        print(f"[m={i}] AUC before calib = {auc_pre:.4f}")

        # --- fit calibrator on train only (K-fold internal)
        calib = make_calibrator(base_estimator=base, method=args.method, cv=kfolds)
        with Heartbeat(prefix=f"[hb] m={i} calibrator.fit", interval=float(args.heartbeat)):
            calib.fit(X_tr, y_tr)  # CV happens internally on training set

        # post-calib on test
        prob_post = get_proba(calib, X_te)
        if prob_post.min() < 0 or prob_post.max() > 1:
            prob_post = 1.0 / (1.0 + np.exp(-prob_post))
        auc_post = roc_auc_score(y_te, prob_post)
        per_m_auc_post.append(float(auc_post))
        print(f"[m={i}] AUC after  calib = {auc_post:.4f}")

        # save calibrated model per m
        models_dir = outputs_dir / "models"; ensure_dir(models_dir)
        model_path = models_dir / f"{args.model}_{args.method}_m{i:02d}.joblib"
        joblib.dump(calib, model_path)

        # accumulate post-calib prob
        if prob_sum_post is None:
            prob_sum_post = prob_post.astype("float64")
        else:
            prob_sum_post += prob_post.astype("float64")

        # per-m calibration compare figure
        plot_calib_compare(
            y_te, prob_pre, prob_post,
            outputs_dir / f"calibration_{args.model}_{args.method}_m{i:02d}.png",
            f"Calibration (m={i}) - {args.model} [{args.method}]"
        )

    # ensemble post-calib
    y_te = y_test_ref
    prob_ens = (prob_sum_post / len(m_paths)).astype("float64")
    auc_ens = roc_auc_score(y_te, prob_ens)
    ap_ens  = average_precision_score(y_te, prob_ens)
    thr     = 0.5
    yhat    = (prob_ens >= thr).astype(int)

    metrics = {
        "model": f"{args.model}_{args.method}",
        "base_model": args.model,
        "method": args.method,
        "M": len(m_paths),
        "roc_auc": float(auc_ens),
        "avg_precision": float(ap_ens),
        "accuracy": float(accuracy_score(y_te, yhat)),
        "f1": float(f1_score(y_te, yhat)),
        "precision": float(precision_score(y_te, yhat)),
        "recall": float(recall_score(y_te, yhat)),
        "brier": float(brier_score_loss(y_te, prob_ens)),
        "thr": thr,
    }

    # save metrics (per-model file + global)
    append_row(outputs_dir / f"metrics_test_{args.model}_{args.method}.csv", metrics)
    append_row(outputs_dir / "metrics_test.csv", metrics, key_col="model", key_val=f"{args.model}_{args.method}")

    # auc table: per-m (before/after) + ensemble(after)
    auc_rows = []
    for i, (a0, a1) in enumerate(zip(per_m_auc_pre, per_m_auc_post), start=1):
        auc_rows.append({"model": f"{args.model}_{args.method}", "m_index": i, "auc_before": a0, "auc_after": a1})
    auc_rows.append({"model": f"{args.model}_{args.method}", "m_index": 0, "auc_before": np.nan, "auc_after": float(auc_ens)})
    auc_tab = pd.DataFrame(auc_rows)
    auc_csv = outputs_dir / "model_auc_test.csv"
    if auc_csv.exists():
        try:
            old = pd.read_csv(auc_csv)
            all_cols = list(dict.fromkeys(list(old.columns) + list(auc_tab.columns)))
            old = old.reindex(columns=all_cols)
            auc_tab = auc_tab.reindex(columns=all_cols)
            pd.concat([old, auc_tab], axis=0, ignore_index=True).to_csv(auc_csv, index=False)
        except Exception:
            auc_tab.to_csv(auc_csv, index=False)
    else:
        auc_tab.to_csv(auc_csv, index=False)

    # plots: ROC & Calibration compare (ensemble)
    plot_roc(y_te, prob_ens, outputs_dir / f"roc_test_{args.model}_{args.method}.png",
             f"ROC (test) - {args.model} [{args.method}] (M={len(m_paths)})")
    plot_calib_compare(y_te, prob_ens, prob_ens,  # 这里为了统一产物名，再单独出一张对比：使用“Before=After=ensemble post”会是一条‘After’曲线
                       outputs_dir / f"calibration_test_{args.model}_{args.method}.png",
                       f"Calibration (test) - {args.model} [{args.method}] (M={len(m_paths)})")

    # ensemble manifest
    models_dir = outputs_dir / "models"; ensure_dir(models_dir)
    with open(models_dir / f"{args.model}_{args.method}_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "method": args.method,
                "M": len(m_paths),
                "mi_paths": [str(p) for p in m_paths],
                "per_m_auc_before": per_m_auc_pre,
                "per_m_auc_after": per_m_auc_post,
                "ensemble_auc_after": float(auc_ens),
                "selected_features": selected,
                "best_params_used": best_params,
                "kfolds": kfolds,
            },
            f, ensure_ascii=False, indent=2
        )

    # final log
    print("[ok] Calibration 完成：")
    print(f"  - AUC(after): {auc_ens:.4f} | AP(after): {ap_ens:.4f}")
    print(f"  - Acc/F1 at 0.5: {metrics['accuracy']:.4f}/{metrics['f1']:.4f}")
    print(f"  - Brier(after): {metrics['brier']:.4f}")
    print(f"[save] 指标: outputs/metrics_test_{args.model}_{args.method}.csv")
    print(f"[save] 汇总: outputs/metrics_test.csv")
    print(f"[save] AUC表: outputs/model_auc_test.csv")
    print(f"[save] ROC图: outputs/roc_test_{args.model}_{args.method}.png")
    print(f"[save] 校准图: outputs/calibration_test_{args.model}_{args.method}.png")
    print(f"[save] 模型工件: outputs/models/{args.model}_{args.method}_mXX.joblib, outputs/models/{args.model}_{args.method}_ensemble.json")


if __name__ == "__main__":
    main()



# python -m src.calibrate --config conf/config.yaml --model random_forest --method isotonic --kfolds 5
