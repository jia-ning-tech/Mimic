# src/train_mi.py
# -*- coding: utf-8 -*-
"""
Train & evaluate per-imputation models, then combine across M imputations.
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, precision_score, recall_score, f1_score,
    accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

# tqdm optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# 尝试导入你仓库的 models
try:
    from .models import get_model as _get_model  # 可能是无参或有参
except Exception:
    _get_model = None  # 允许兜底

# ----------------- utils -----------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_mi_index(mi_dir: Path) -> Dict[str, Any]:
    idx_path = mi_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"未找到 MI 索引：{idx_path}；请先运行 `python -m src.multiple_imputation`。"
        )
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_selected_features(artifacts_dir: Path) -> Optional[List[str]]:
    sel_path = artifacts_dir / "selected_features.json"
    if not sel_path.exists():
        return None
    with open(sel_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    feats = obj.get("selected_features")
    return feats if isinstance(feats, list) and feats else None

def build_class_weight(cfg: Dict[str, Any]) -> Optional[Any]:
    imb = cfg.get("imbalance", {}) or {}
    if not imb.get("use_class_weight", False):
        return None
    if "positive_class_weight" in imb:
        return {0: 1.0, 1: float(imb["positive_class_weight"])}
    return "balanced"

def try_inject_class_weight(estimator, class_weight: Any):
    if class_weight is None:
        return estimator
    try:
        if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
            params = estimator.get_params(deep=False)
            if "class_weight" in params:
                estimator.set_params(class_weight=class_weight)
    except Exception:
        pass
    return estimator

def try_set_params(estimator, **params):
    """存在才设置；避免签名不兼容。"""
    try:
        if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
            valid = estimator.get_params(deep=False)
            to_set = {k: v for k, v in params.items() if v is not None and k in valid}
            if to_set:
                estimator.set_params(**to_set)
    except Exception:
        pass
    return estimator

def make_base_estimator(model_name: str):
    """
    统一适配 src.models.get_model 的不同实现：
      - 若 get_model() 是无参，视为返回注册表 dict：{name: ctor|callable}
      - 若 get_model(name) 是有参，直接调用
      - 若都失败，提供 random_forest 的兜底
    """
    # 1) 有 get_model 时，探测其签名
    if _get_model is not None:
        import inspect
        try:
            sig = inspect.signature(_get_model)
            if len(sig.parameters) == 0:
                # 无参：期望返回注册表
                registry = _get_model()
                if isinstance(registry, dict) and model_name in registry:
                    ctor = registry[model_name]
                    # ctor 可能是类/可调用；尽量调用无参构造
                    try:
                        return ctor()
                    except TypeError:
                        # 有些可能要求关键字参数，先无参尝试，不行再直接返回原对象
                        return ctor
                # 注册表无此模型，继续兜底
            else:
                # 有参：按我们预期的方式调用
                return _get_model(model_name)
        except Exception:
            # get_model 不可用，走兜底
            pass

    # 2) 兜底：至少支持 random_forest
    from sklearn.ensemble import RandomForestClassifier
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=500, random_state=42)
    # 进一步可加其它兜底模型
    raise ValueError(f"无法构建模型：{model_name}；请检查 src.models.get_model 的实现或改为受支持的名称。")

def get_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if getattr(p, "ndim", 1) == 2 else p
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    y = est.predict(X)
    return y.astype(float)

# --------------- metrics & plots ---------------
def _safe_auc_ap(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    auc, ap = _safe_auc_ap(y_true, prob)
    brier = float(brier_score_loss(y_true, prob))
    pred = (prob >= thr).astype(int)
    acc = float(accuracy_score(y_true, pred))
    f1  = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec  = float(recall_score(y_true, pred, zero_division=0))
    return dict(roc_auc=auc, average_precision=ap, accuracy=acc, f1=f1,
                precision=prec, recall=rec, brier=brier)

def plot_roc_pr(y_true: np.ndarray, prob: np.ndarray, out_png: Path, title: str):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        ax[0].plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
        ax[0].plot([0, 1], [0, 1], ls="--", c="gray")
        ax[0].legend(loc="lower right")
    else:
        ax[0].text(0.5, 0.5, "ROC未定义（单类）", ha="center", va="center")
    ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")

    # PR
    if np.unique(y).size == 2:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax[1].plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
        ax[1].legend(loc="lower left")
    else:
        ax[1].text(0.5, 0.5, "PR未定义（单类）", ha="center", va="center")
    ax[1].set_title("PR"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")

    plt.suptitle(title)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_calibration(y_true: np.ndarray, prob: np.ndarray, out_png: Path, bins: int = 10, title: str = "Calibration"):
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    try:
        df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=min(bins, max(2, df["p"].nunique())), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        y_mean=("y", "mean"), p_mean=("p", "mean"), n=("y", "size")
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.scatter(grouped["p_mean"], grouped["y_mean"])
    for _, r in grouped.iterrows():
        ax.text(r["p_mean"], r["y_mean"], str(int(r["n"])), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title(title)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

# --------------- heartbeat ---------------
class Heartbeat:
    def __init__(self, prefix: str, interval: float = 5.0):
        self.prefix = prefix
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._t = None
        self._t0 = None

    def _runner(self):
        while not self._stop.wait(self.interval):
            dt = time.time() - self._t0
            print(f"{self.prefix} ... {dt:.1f}s elapsed", flush=True)

    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._runner, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.1)

# --------------- Rubin pooling (simple) ---------------
def rubin_pool_scalar(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    M = len(arr)
    mean = float(np.mean(arr))
    if M <= 1:
        return {"mean": mean, "std_pool": 0.0}
    B = float(np.var(arr, ddof=1))
    U_bar = B  # 近似
    T = U_bar + (1 + 1.0 / M) * B
    std_pool = float(np.sqrt(max(T, 0.0)))
    return {"mean": mean, "std_pool": std_pool}

# ----------------- main -----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train per-imputation and pool results.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--kfolds", type=int, default=5, help="K folds for calibration.")
    ap.add_argument("--heartbeat", type=float, default=5.0)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    models_dir  = outputs_dir / "models"; ensure_dir(models_dir)
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables", "outputs/tables")); ensure_dir(tables_dir)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))
    probs_dir   = outputs_dir / "probs"; ensure_dir(probs_dir)

    # MI
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    M = len(m_paths)
    if M == 0:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    # feature selection（严格对齐）
    selected = read_selected_features(artifacts_dir)

    # model settings
    n_jobs = int(cfg.get("project", {}).get("n_jobs", -1))
    class_weight = build_class_weight(cfg)
    suffix = "" if args.method == "raw" else f"_{args.method}"

    # 保存总体标签（检查各 m 一致性）
    y_test_ref: Optional[np.ndarray] = None
    prob_list: List[np.ndarray] = []
    metrics_rows: List[Dict[str, Any]] = []

    print(f"[info] 多重插补训练: M={M}, model={args.model}, method={args.method}, kfolds={args.kfolds}")

    for i, pth in enumerate(tqdm(m_paths, desc="MI", unit="m"), start=1):
        art = joblib.load(pth)
        feat_order = list(art["feature_order"])
        X_tr = np.asarray(art["X_train"]).astype("float32")
        y_tr = np.asarray(art["y_train"]).astype(int)
        X_te = np.asarray(art["X_test"]).astype("float32")
        y_te = np.asarray(art["y_test"]).astype(int)

        # 检查 y_test 一致性
        if y_test_ref is None:
            y_test_ref = y_te
        else:
            if not np.array_equal(y_test_ref, y_te):
                raise RuntimeError("不同 m 的 y_test 不一致，可能分割索引未固定。")

        # 特征对齐
        if selected:
            keep_idx = [feat_order.index(c) for c in selected if c in feat_order]
            if len(keep_idx) != len(selected):
                missing = [c for c in selected if c not in feat_order]
                raise RuntimeError(f"[m={i}] 选中特征缺失：{missing}")
            X_tr = X_tr[:, keep_idx]
            X_te = X_te[:, keep_idx]
            feat_used = selected
        else:
            feat_used = feat_order

        # === 构建基学习器（自适配 get_model 实现） ===
        base = make_base_estimator(args.model)
        # 安全注入参数（存在才设置）
        base = try_set_params(base, n_jobs=n_jobs, random_state=42 + i)
        base = try_inject_class_weight(base, class_weight)

        # 概率校准
        if args.method in ("isotonic", "sigmoid"):
            cv = int(args.kfolds)
            est = CalibratedClassifierCV(estimator=base, method=args.method, cv=cv)
        else:
            est = base

        with Heartbeat(prefix=f"[hb] m={i} training", interval=float(args.heartbeat)):
            est.fit(X_tr, y_tr)

        prob = get_proba(est, X_te)
        prob_list.append(prob.astype("float64"))

        # —— 保存模型 —— #
        model_path = models_dir / (
            f"{args.model}_{'m%02d' % i}.joblib" if args.method == "raw"
            else f"{args.model}_{args.method}_m{'%02d' % i}.joblib"
        )
        joblib.dump(est, model_path)

        # —— 保存 per-m 概率 —— #
        np.save(probs_dir / f"probs_m{i:02d}_{args.model}{suffix}.npy", prob)

        # —— 指标 —— #
        met = compute_metrics(y_te, prob, thr=0.5)
        row = dict(m=i, model=args.model, method=args.method, n_test=len(y_te), features=len(feat_used), **met)
        metrics_rows.append(row)

        per_m_csv = mi_dir / f"metrics_m{i:02d}_{args.model}{suffix}.csv"
        pd.DataFrame([row]).to_csv(per_m_csv, index=False)

        print(f"[m={i}] AUC={met['roc_auc']:.4f}  AP={met['average_precision']:.4f}  "
              f"Acc={met['accuracy']:.4f}  F1={met['f1']:.4f}  Brier={met['brier']:.4f}")

    assert y_test_ref is not None
    y_te = y_test_ref

    # —— 概率均值集成 —— #
    prob_stack = np.vstack(prob_list)  # [M, N]
    prob_ens = prob_stack.mean(axis=0)
    np.save(probs_dir / f"probs_ensemble_{args.model}{suffix}.npy", prob_ens)

    met_ens = compute_metrics(y_te, prob_ens, thr=0.5)
    print(f"[ens] AUC={met_ens['roc_auc']:.4f}  AP={met_ens['average_precision']:.4f}  "
          f"Acc={met_ens['accuracy']:.4f}  F1={met_ens['f1']:.4f}  Brier={met_ens['brier']:.4f}")

    # —— Rubin pooling（标量近似） —— #
    pool_cols = ["roc_auc", "average_precision", "brier", "accuracy", "f1", "precision", "recall"]
    pooled = {}
    df_m = pd.DataFrame(metrics_rows)
    for col in pool_cols:
        pooled[col] = rubin_pool_scalar(df_m[col].tolist())

    # —— 汇总保存（outputs/tables） —— #
    pooled_flat = {f"{k}_mean": v["mean"] for k, v in pooled.items()}
    pooled_flat.update({f"{k}_std": v["std_pool"] for k, v in pooled.items()})
    ens_flat = {f"ens_{k}": v for k, v in met_ens.items()}

    out_row = dict(model=args.model, method=args.method, M=len(prob_list), **pooled_flat, **ens_flat)
    out_csv = tables_dir / f"metrics_test_{args.model}_mi.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        keep = old[~((old.get("model") == args.model) & (old.get("method") == args.method))]
        df_new = pd.concat([keep, pd.DataFrame([out_row])], axis=0, ignore_index=True)
    else:
        df_new = pd.DataFrame([out_row])
    ensure_dir(out_csv.parent)
    df_new.to_csv(out_csv, index=False)

    # 图像（使用集成概率展示） -> outputs/figures
    roc_png = figures_dir / f"roc_test_{args.model}_mi.png"
    plot_roc_pr(y_te, prob_ens, roc_png,
                title=f"ROC/PR – {args.model}{'' if args.method=='raw' else ' | '+args.method} (MI ensemble)")

    cal_png = figures_dir / f"calibration_test_{args.model}_mi.png"
    plot_calibration(y_te, prob_ens, cal_png, bins=10,
                     title=f"Calibration – {args.model}{'' if args.method=='raw' else ' | '+args.method} (MI ensemble)")

    print("[ok] train_mi 完成：")
    print(f"  - per-m metrics: {mi_dir}/metrics_mXX_{args.model}{suffix}.csv")
    print(f"  - pooled table : {out_csv}")
    print(f"  - ROC/PR figure: {roc_png}")
    print(f"  - Calib figure : {cal_png}")

if __name__ == "__main__":
    main()


# # 逐 m 训练 + 合并（raw 概率）
# python -m src.train_mi --config conf/config.yaml --model random_forest

# # 使用 isotonic 校准（K=5）
# python -m src.train_mi --config conf/config.yaml --model random_forest --method isotonic --kfolds 5

# # 其它模型同理（支持你配置中的 15 个）
# python -m src.train_mi --config conf/config.yaml --model lightgbm --method sigmoid
