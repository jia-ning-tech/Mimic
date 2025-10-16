# src/evaluate.py
# -*- coding: utf-8 -*-
"""
Evaluate MI-ensemble probabilities on the held-out test set.

Refactor highlights:
- Import shared utilities from data_utils (atomic writes / heartbeat / style)
- Save tables to outputs/tables/ and figures to outputs/figures/
- Keep computational behavior unchanged

Outputs (now under proper folders):
  Tables (outputs/tables/):
    - metrics_test_<model>[ _<method>].csv
    - metrics_test.csv  (append/update by model+method)
    - model_auc_test.csv
    - thresholds_<model>[ _<method>].csv
  Figures (outputs/figures/):
    - roc_test_<model>[ _<method>].png
    - pr_test_<model>[ _<method>].png
    - calibration_test_<model>[ _<method>].png
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

# sklearn
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---- common utils ----
from .data_utils import (
    load_yaml, ensure_dir,
    Heartbeat, apply_mpl_style,
    threshold_metrics,
    atomic_to_csv, atomic_savefig,
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X_test_df (full columns), y_test (1d), prob_ensemble (1d), used_features (list)
    """
    prob_sum = None
    y_ref = None
    feat_used: Optional[List[str]] = None
    X_ref_df: Optional[pd.DataFrame] = None

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
            X_ref_df = X_te_df.copy()
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
    assert y_ref is not None and feat_used is not None and X_ref_df is not None
    return X_ref_df, y_ref, prob_ens, feat_used

def bootstrap_auc_ap_ci(
    y_true: np.ndarray, prob: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return (auc, lo, hi), (ap, lo, hi) with 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    n = len(y)
    auc_list: List[float] = []
    ap_list: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]; pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yb, pb)
        auc_list.append(auc(fpr, tpr))
        ap_list.append(average_precision_score(yb, pb))
    auc_val = float(np.nan)
    ap_val  = float(np.nan)
    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = float(auc(fpr, tpr))
        ap_val  = float(average_precision_score(y, p))
    if len(auc_list) == 0:
        return (auc_val, float("nan"), float("nan")), (ap_val, float("nan"), float("nan"))
    lo_auc = float(np.nanquantile(auc_list, 0.025))
    hi_auc = float(np.nanquantile(auc_list, 0.975))
    lo_ap  = float(np.nanquantile(ap_list, 0.025))
    hi_ap  = float(np.nanquantile(ap_list, 0.975))
    return (auc_val, lo_auc, hi_auc), (ap_val, lo_ap, hi_ap)

def plot_roc_pr(
    y_true: np.ndarray, prob: np.ndarray,
    roc_png: Path, pr_png: Path,
    title_suffix: str = "", n_boot: int = 0, plot_ci: bool = False, seed: int = 42
):
    apply_mpl_style()
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)

    # --- ROC ---
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.4))
    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f"Model (AUC={roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        # CI band
        if plot_ci and n_boot > 0:
            rng = np.random.default_rng(seed)
            grid = np.linspace(0, 1, 101)
            tpr_samples = []
            for _ in tqdm(range(n_boot), desc="ROC-Boot", unit="it"):
                idx = rng.integers(0, len(y), size=len(y))
                yb = y[idx]; pb = p[idx]
                if np.unique(yb).size < 2:
                    continue
                f, t, _ = roc_curve(yb, pb)
                tpr_interp = np.interp(grid, f, t)
                tpr_samples.append(tpr_interp)
            if len(tpr_samples) > 10:
                band = np.vstack(tpr_samples)
                lo = np.nanquantile(band, 0.025, axis=0)
                hi = np.nanquantile(band, 0.975, axis=0)
                ax1.fill_between(grid, lo, hi, alpha=0.15, label="95% CI")
        ax1.set_title(f"ROC Curve{title_suffix}")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.25)
    else:
        ax1.text(0.5, 0.5, "ROC 未定义（测试集中仅单一类别）", ha="center", va="center")
    plt.tight_layout()
    ensure_dir(roc_png.parent)
    atomic_savefig(fig1, roc_png)
    plt.close(fig1)

    # --- PR ---
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
    precision, recall, _ = precision_recall_curve(y, p)
    ap_val = average_precision_score(y, p) if np.unique(y).size == 2 else np.nan
    ax2.plot(recall, precision, lw=2, label=f"Model (AP={ap_val:.3f})")
    if plot_ci and n_boot > 0 and np.unique(y).size == 2:
        rng = np.random.default_rng(seed)
        grid = np.linspace(0, 1, 101)
        prec_samples = []
        for _ in tqdm(range(n_boot), desc="PR-Boot", unit="it"):
            idx = rng.integers(0, len(y), size=len(y))
            yb = y[idx]; pb = p[idx]
            if np.unique(yb).size < 2:
                continue
            pr, rc, _ = precision_recall_curve(yb, pb)
            pr_interp = np.interp(grid, rc[::-1], pr[::-1])  # ensure increasing rc
            prec_samples.append(pr_interp)
        if len(prec_samples) > 10:
            band = np.vstack(prec_samples)
            lo = np.nanquantile(band, 0.025, axis=0)
            hi = np.nanquantile(band, 0.975, axis=0)
            ax2.fill_between(grid, lo, hi, alpha=0.15, label="95% CI")
    ax2.set_title(f"Precision–Recall Curve{title_suffix}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(pr_png.parent)
    atomic_savefig(fig2, pr_png)
    plt.close(fig2)

def plot_calibration(
    y_true: np.ndarray, prob: np.ndarray,
    calib_png: Path, title_suffix: str = "", n_bins: int = 10
):
    apply_mpl_style()
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # calibration curve
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, lw=2, label="Model")
    ax.set_title(f"Calibration Curve{title_suffix}")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(calib_png.parent)
    atomic_savefig(fig, calib_png)
    plt.close(fig)

def small_threshold_table(
    y_true: np.ndarray, prob: np.ndarray, thr_min: float = 0.01, thr_max: float = 0.99, step: float = 0.01
) -> pd.DataFrame:
    thrs = np.arange(max(1e-6, thr_min), min(1-1e-6, thr_max) + 1e-12, step, dtype=float)
    rows = []
    for t in thrs:
        m = threshold_metrics(y_true, prob, float(t))
        m["thr"] = float(t)
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["thr", "sensitivity", "specificity", "precision", "npv", "accuracy", "f1", "brier", "tp", "fp", "tn", "fn"]
    return df[cols]

def append_or_update_summary(summary_csv: Path, metrics_row: Dict[str, Any], keys: List[str]):
    """
    Append or update by unique key (e.g., ["model","method"]).
    Ensure 'model' column exists.
    """
    ensure_dir(summary_csv.parent)
    if summary_csv.exists():
        try:
            df_old = pd.read_csv(summary_csv)
            # ensure columns
            for k in keys:
                if k not in df_old.columns:
                    df_old[k] = np.nan
            mask = np.ones(len(df_old), dtype=bool)
            for k in keys:
                mask &= (df_old[k].astype(str) == str(metrics_row[k]))
            df_new = pd.concat([df_old.loc[~mask], pd.DataFrame([metrics_row])], ignore_index=True)
        except Exception:
            df_new = pd.DataFrame([metrics_row])
    else:
        df_new = pd.DataFrame([metrics_row])
    atomic_to_csv(df_new, summary_csv)

# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate MI-ensemble probabilities on test set.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations for AUC/AP CI and optional band.")
    ap.add_argument("--plot_ci", action="store_true", help="Plot ROC/PR bands with bootstrap.")
    ap.add_argument("--heartbeat", type=float, default=5.0)
    # light threshold table
    ap.add_argument("--thr_min", type=float, default=0.01)
    ap.add_argument("--thr_max", type=float, default=0.99)
    ap.add_argument("--thr_step", type=float, default=0.01)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    # 根目录仍可创建，但不再用于表格/图像写盘
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)

    # 规范目录：表格与图像
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
    with Heartbeat(prefix=f"[hb] evaluate ({args.model})", interval=float(args.heartbeat)):
        X_ref_df, y_te, prob_ens, feat_used = average_probs_across_m(
            model_name=args.model, method=args.method, m_paths=m_paths,
            selected_for_pred=selected, heartbeat_sec=float(args.heartbeat)
        )

    y = y_te.astype(int)
    p = prob_ens.astype(float)

    # ---- point metrics ----
    # AUC / AP
    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = float(auc(fpr, tpr))
        ap_val  = float(average_precision_score(y, p))
    else:
        auc_val = float("nan"); ap_val = float("nan")
    # Calibration (Brier)
    brier = float(brier_score_loss(y, p))
    # Default threshold 0.5 for point metrics table
    met = threshold_metrics(y, p, 0.5)

    # ---- bootstrap CI for AUC/AP ----
    (auc_pt, auc_lo, auc_hi), (ap_pt, ap_lo, ap_hi) = bootstrap_auc_ap_ci(
        y, p, n_boot=int(args.n_boot), seed=42
    )

    # ---- save metrics_test_<model>.csv ----
    suffix = "" if args.method == "raw" else f"_{args.method}"
    metrics_csv = tables_dir / f"metrics_test_{args.model}{suffix}.csv"
    metrics_row = {
        "model": args.model,
        "method": args.method,
        "n_test": int(len(y)),
        "prevalence": float(np.mean(y)),
        "roc_auc": auc_val,
        "roc_auc_lo": auc_lo,
        "roc_auc_hi": auc_hi,
        "ap": ap_val,
        "ap_lo": ap_lo,
        "ap_hi": ap_hi,
        "brier": brier,
        "accuracy": met["accuracy"],
        "f1": met["f1"],
        "precision": met["precision"],
        "recall": met["sensitivity"],  # 与早期命名保持一致
    }
    atomic_to_csv(pd.DataFrame([metrics_row]), metrics_csv)

    # ---- append / update summary metrics_test.csv (by model+method) ----
    summary_csv = tables_dir / "metrics_test.csv"
    append_or_update_summary(summary_csv, metrics_row, keys=["model", "method"])

    # ---- model_auc_test.csv (overall table for quick glance) ----
    auc_table_csv = tables_dir / "model_auc_test.csv"
    row_auc = {"model": args.model, "method": args.method, "roc_auc": auc_val, "ap": ap_val}
    append_or_update_summary(auc_table_csv, row_auc, keys=["model", "method"])

    # ---- thresholds table (small sweep; separate from full threshold_scan) ----
    thr_csv = tables_dir / f"thresholds_{args.model}{suffix}.csv"
    thr_df = small_threshold_table(y, p, thr_min=float(args.thr_min), thr_max=float(args.thr_max), step=float(args.thr_step))
    atomic_to_csv(thr_df, thr_csv)

    # ---- plots ----
    title_suffix = f" – {args.model}{'' if args.method=='raw' else ' | '+args.method}"
    roc_png = figures_dir / f"roc_test_{args.model}{suffix}.png"
    pr_png  = figures_dir / f"pr_test_{args.model}{suffix}.png"
    calib_png = figures_dir / f"calibration_test_{args.model}{suffix}.png"

    plot_roc_pr(
        y, p, roc_png, pr_png, title_suffix=title_suffix,
        n_boot=int(args.n_boot), plot_ci=bool(args.plot_ci), seed=42
    )
    plot_calibration(y, p, calib_png, title_suffix=title_suffix)

    # ---- console ----
    print("[ok] Evaluate 完成：")
    print(f"  - AUC: {auc_val:.4f}  (95% CI {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"  - AP : {ap_val:.4f}  (95% CI {ap_lo:.4f}–{ap_hi:.4f})")
    print(f"  - Acc: {met['accuracy']:.4f} at thr=0.5")
    print(f"  - F1 : {met['f1']:.4f} at thr=0.5")
    print(f"  - Brier: {brier:.4f}")
    print(f"[save] 主指标: {metrics_csv}")
    print(f"[save] 汇总表: {summary_csv}")
    print(f"[save] AUC表: {auc_table_csv}")
    print(f"[save] 阈值表: {thr_csv}")
    print(f"[save] ROC图: {roc_png}")
    print(f"[save] PR图 : {pr_png}")
    print(f"[save] 校准图: {calib_png}")

if __name__ == "__main__":
    main()



# # 原始（raw）概率
# python -m src.evaluate --config conf/config.yaml --model random_forest

# # 对校准后的概率评估（isotonic/sigmoid）
# python -m src.evaluate --config conf/config.yaml --model random_forest --method isotonic

# # 若希望在图里加 bootstrap 置信带：
# python -m src.evaluate --config conf/config.yaml --model random_forest --method isotonic --plot_ci --n_boot 1000
