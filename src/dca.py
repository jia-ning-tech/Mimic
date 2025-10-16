# src/dca.py
# -*- coding: utf-8 -*-
"""
Decision Curve Analysis (DCA) with MI-ensemble probabilities.

Refactor highlights:
- Import shared utilities from data_utils (atomic writes / heartbeat / style)
- Keep previous behavior and outputs unchanged
- Output both Net Benefit (NB) and Standardized NB (sNB = NB / prevalence)
- Vertical labels kept on axis (no change), unified style applied

Outputs:
  - outputs/dca_<model>[ _<method>].csv
  - outputs/dca_nb_<model>[ _<method>].png
  - outputs/dca_snb_<model>[ _<method>].png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---- import shared tools
from .data_utils import (
    load_yaml, ensure_dir,
    Heartbeat, apply_mpl_style,
    atomic_to_csv, atomic_savefig
)

# ---------------- helpers (local) ----------------
def read_mi_index(mi_dir: Path) -> Dict[str, Any]:
    idx_path = mi_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"未找到 MI 索引：{idx_path}；请先运行 `python -m src.multiple_imputation`。")
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_selected_features(artifacts_dir: Path) -> Optional[List[str]]:
    sel_path = artifacts_dir / "selected_features.json"
    if not sel_path.exists():
        return None
    obj = json.loads(sel_path.read_text(encoding="utf-8"))
    feats = obj.get("selected_features")
    return feats if isinstance(feats, list) and feats else None

def list_model_path(model_name: str, method: str, m_index: int) -> Path:
    models_dir = Path("outputs") / "models"
    if method == "raw":
        return models_dir / f"{model_name}_m{m_index:02d}.joblib"
    else:
        return models_dir / f"{model_name}_{method}_m{m_index:02d}.joblib"

def get_proba(est, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) from estimator or calibrated wrapper."""
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    y = est.predict(X)
    return y.astype(float)

def average_probs_across_m(
    model_name: str,
    method: str,
    m_paths: List[Path],
    selected_for_pred: Optional[List[str]],
    heartbeat_sec: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      y_test (1d), prob_ensemble (1d), used_features (list)
    """
    prob_sum = None
    y_ref = None
    feat_used: Optional[List[str]] = None

    for i, pth in enumerate(tqdm(m_paths, desc="Eval M", unit="m"), start=1):
        art = joblib.load(pth)
        feat_order = list(art["feature_order"])
        X_te_df = pd.DataFrame(art["X_test"], columns=feat_order)
        y_te = np.asarray(art["y_test"]).astype(int)

        if selected_for_pred:
            use_cols = [c for c in selected_for_pred if c in X_te_df.columns]
            if len(use_cols) != len(selected_for_pred):
                miss = [c for c in selected_for_pred if c not in X_te_df.columns]
                raise RuntimeError(f"selected_features.json 中的列缺失于测试矩阵：{miss}")
            X_pred = X_te_df[use_cols].values.astype("float32")
            feat_used = use_cols
        else:
            X_pred = X_te_df.values.astype("float32")
            feat_used = feat_order

        if y_ref is None:
            y_ref = y_te
        else:
            if len(y_ref) != len(y_te):
                raise RuntimeError("不同 m 的测试集样本数不一致，请检查 split 索引。")

        model_path = list_model_path(model_name, method, i)
        if not model_path.exists():
            hint = "请先运行 `python -m src.train`" if method == "raw" \
                   else f"请先运行 `python -m src.calibrate --model {model_name} --method {method}`"
            raise FileNotFoundError(f"未找到模型文件：{model_path}；{hint}。")
        est = joblib.load(model_path)

        with Heartbeat(prefix=f"[hb] m={i} predicting", interval=heartbeat_sec):
            prob = get_proba(est, X_pred)

        if prob_sum is None:
            prob_sum = prob.astype("float64")
        else:
            prob_sum += prob.astype("float64")

    prob_ens = (prob_sum / len(m_paths)).astype("float64")
    assert y_ref is not None and feat_used is not None
    return y_ref, prob_ens, feat_used

# ---------------- DCA core ----------------
def decision_curve(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Compute Net Benefit (NB) and Standardized NB (sNB) for model and reference strategies.
    NB(thr) = TP/N - FP/N * (thr / (1-thr))
    sNB = NB / prevalence
    Treat-All: NB_all = prevalence - (1 - prevalence) * (thr / (1-thr))
    Treat-None: NB_none = 0
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    N = float(len(y))
    prev = float(np.mean(y))

    rows = []
    for t in thresholds:
        if not (0.0 < t < 1.0):
            continue
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = t / (1.0 - t)
        nb_model = (tp / N) - (fp / N) * w
        nb_all   = prev - (1.0 - prev) * w
        rows.append({
            "threshold": float(t),
            "prevalence": prev,
            "nb_model": nb_model,
            "nb_all": nb_all,
            "nb_none": 0.0,
            "snb_model": (nb_model / prev) if prev > 0 else np.nan,
            "snb_all":   (nb_all / prev) if prev > 0 else np.nan,
            "snb_none":  0.0
        })
    return pd.DataFrame(rows)

# ---------------- plotting ----------------
def plot_dca_nb(df: pd.DataFrame, out_png: Path, title: str):
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(df["threshold"], df["nb_model"], label="Model", lw=2)
    ax.plot(df["threshold"], df["nb_all"],   label="Treat All", ls="--", lw=1.5)
    ax.plot(df["threshold"], df["nb_none"],  label="Treat None", ls=":", lw=1.5)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png)
    plt.close(fig)

def plot_dca_snb(df: pd.DataFrame, out_png: Path, title: str):
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(df["threshold"], df["snb_model"], label="Model", lw=2)
    ax.plot(df["threshold"], df["snb_all"],   label="Treat All", ls="--", lw=1.5)
    ax.plot(df["threshold"], df["snb_none"],  label="Treat None", ls=":", lw=1.5)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Standardized Net Benefit")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png)
    plt.close(fig)

# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Decision Curve Analysis (DCA) with MI-ensemble.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--thr_min", type=float, default=0.01)
    ap.add_argument("--thr_max", type=float, default=0.99)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--heartbeat", type=float, default=5.0)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables", "outputs/tables")); ensure_dir(tables_dir)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))

    # MI artifacts
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    selected = read_selected_features(artifacts_dir)

    # collect probs across M
    with Heartbeat(prefix=f"[hb] dca ({args.model})", interval=float(args.heartbeat)):
        y_te, prob_ens, feat_used = average_probs_across_m(
            model_name=args.model, method=args.method, m_paths=m_paths,
            selected_for_pred=selected, heartbeat_sec=float(args.heartbeat)
        )

    prev = float(np.mean(y_te))
    thr_min = max(1e-6, float(args.thr_min))
    thr_max = min(1 - 1e-6, float(args.thr_max))
    step = float(args.thr_step)
    thresholds = np.arange(thr_min, thr_max + 1e-12, step, dtype=float)

    # DCA
    dca_df = decision_curve(y_te, prob_ens, thresholds)

    # save
    suffix = "" if args.method == "raw" else f"_{args.method}"
    dca_csv = outputs_dir / f"dca_{args.model}{suffix}.csv"
    dca_nb_png  = outputs_dir / f"dca_nb_{args.model}{suffix}.png"
    dca_snb_png = outputs_dir / f"dca_snb_{args.model}{suffix}.png"

    atomic_to_csv(dca_df, dca_csv)
    plot_dca_nb(dca_df, dca_nb_png,  title=f"DCA Net Benefit – {args.model}{'' if args.method=='raw' else ' | '+args.method}")
    plot_dca_snb(dca_df, dca_snb_png, title=f"DCA Standardized Net Benefit – {args.model}{'' if args.method=='raw' else ' | '+args.method}")

    # console
    print("[ok] DCA 完成：")
    print(f"  - N={len(y_te)}, prevalence={prev:.4f}")
    print(f"  - 阈值范围: [{thr_min:.3f}, {thr_max:.3f}] step={step:.3f} (n={len(dca_df)})")
    print(f"[save] DCA表 : {dca_csv}")
    print(f"[save] 曲线-NB : {dca_nb_png}")
    print(f"[save] 曲线-sNB: {dca_snb_png}")

if __name__ == "__main__":
    main()


# # 原始概率的 DCA（0.01~0.99 每 0.01）
# python -m src.dca --config conf/config.yaml --model random_forest

# # 对校准后概率做 DCA（等同你之前的用法）
# python -m src.dca --config conf/config.yaml --model random_forest --method isotonic

# # 改阈值网格（例如论文常用 0.05~0.7）
# python -m src.dca --config conf/config.yaml --model random_forest --method isotonic \
#   --thr_min 0.05 --thr_max 0.70 --thr_step 0.01
