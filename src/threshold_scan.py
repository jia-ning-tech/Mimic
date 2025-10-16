# -*- coding: utf-8 -*-
"""
Threshold scanning with robust I/O, heartbeat, tqdm, and auto-threshold recommendations.
This version imports shared utils from data_utils and adds robust de-overlap label placement.

Outputs (unchanged):
  - outputs/threshold_scan_<model>[_<method>].csv
  - outputs/threshold_scan_<model>[_<method>]_summary.json
  - outputs/thr_metrics_<model>[_<method>].png
  - outputs/thr_fscore_<model>[_<method>].png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_fscore_support

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---- shared utils ----
from .data_utils import (
    load_yaml, ensure_dir,
    Heartbeat, apply_mpl_style,
    threshold_metrics,
    atomic_to_csv, atomic_savefig,
)

# ---------------- helpers ----------------
def list_model_path(model_name: str, method: str, m_index: int) -> Path:
    models_dir = Path("outputs") / "models"
    if method == "raw":
        return models_dir / f"{model_name}_m{m_index:02d}.joblib"
    else:
        return models_dir / f"{model_name}_{method}_m{m_index:02d}.joblib"

def get_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    y = est.predict(X)
    return y.astype(float)

def load_mi_index(mi_dir: Path) -> Dict[str, Any]:
    idx_path = mi_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"未找到 MI 索引：{idx_path}；请先运行 `python -m src.multiple_imputation`。")
    return json.loads(idx_path.read_text(encoding="utf-8"))

def read_selected_features(artifacts_dir: Path) -> Optional[List[str]]:
    sel_path = artifacts_dir / "selected_features.json"
    if not sel_path.exists():
        return None
    obj = json.loads(sel_path.read_text(encoding="utf-8"))
    feats = obj.get("selected_features")
    return feats if isinstance(feats, list) and feats else None

def average_probs_across_m(
    model_name: str,
    method: str,
    m_paths: List[Path],
    selected_for_pred: Optional[List[str]],
    heartbeat_sec: float = 5.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    import joblib

    prob_sum = None
    y_ref = None
    feat_used: Optional[List[str]] = None
    X_ref_df: Optional[pd.DataFrame] = None

    for i, pth in enumerate(tqdm(m_paths, desc="Scan M", unit="m"), start=1):
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

def recommend_thresholds(y: np.ndarray, p: np.ndarray, sens_targets: List[float], spec_targets: List[float]) -> Dict[str, Any]:
    # Youden
    fpr, tpr, thr = roc_curve(y, p)
    youden = tpr - fpr
    youden_idx = int(np.argmax(youden))
    youden_thr = float(thr[youden_idx]) if youden_idx < len(thr) else 0.5
    # F1-max（扫一遍 0.01~0.99 步长 0.001）
    grid = np.arange(0.01, 0.991, 0.001)
    f1_vals = []
    for t in grid:
        m = threshold_metrics(y, p, float(t))
        f1_vals.append(m["f1"])
    f1_idx = int(np.nanargmax(f1_vals))
    f1_thr = float(grid[f1_idx])

    # 固定敏感度、特异度
    def first_thr_by(metric_name: str, target: float, ascending: bool) -> Optional[float]:
        for t in grid:
            m = threshold_metrics(y, p, float(t))
            val = m["sensitivity"] if metric_name == "sens" else m["specificity"]
            if (not ascending and val >= target) or (ascending and val <= target):
                return float(t)
        return None

    sens_list = []
    for s in sens_targets:
        t = first_thr_by("sens", s, ascending=False)
        sens_list.append((s, t))
    spec_list = []
    for s in spec_targets:
        # 找到使 specificity >= s 的最小阈值
        t = None
        for thr0 in grid:
            m = threshold_metrics(y, p, float(thr0))
            if m["specificity"] >= s:
                t = float(thr0)
                break
        spec_list.append((s, t))

    return {
        "youden_thr": youden_thr,
        "f1_thr": f1_thr,
        "sens_targets": sens_list,
        "spec_targets": spec_list,
    }

# ---------- label placer to avoid overlaps ----------
class LabelPlacer:
    """
    Place vertical labels under x-axis with de-overlap.
    Each new label checks existing x's; if too close, it uses a deeper layer.
    """
    def __init__(self, x_tol: float = 0.02, base_y: float = -0.10, step: float = 0.04, max_layers: int = 6):
        self.x_positions: List[float] = []
        self.layers: List[int] = []  # same length as x_positions
        self.x_tol = float(x_tol)
        self.base_y = float(base_y)
        self.step = float(step)
        self.max_layers = int(max_layers)

    def pick_layer(self, x: float) -> int:
        # find the smallest layer not colliding with existing labels near x
        for layer in range(self.max_layers):
            ok = True
            for xp, lp in zip(self.x_positions, self.layers):
                if abs(x - xp) <= self.x_tol and lp == layer:
                    ok = False
                    break
            if ok:
                self.x_positions.append(float(x))
                self.layers.append(layer)
                return layer
        # if exceeded, just put at last layer
        self.x_positions.append(float(x))
        self.layers.append(self.max_layers - 1)
        return self.max_layers - 1

    def bottom_margin(self) -> float:
        """Suggest bottom margin based on how many layers used."""
        if not self.layers:
            return 0.10
        deepest = max(self.layers)
        # base 0.14 + step*layers
        return 0.14 + deepest * 0.045

def annotate_vertical_label(ax: plt.Axes, placer: LabelPlacer, x: float, text: str, color: str):
    layer = placer.pick_layer(float(x))
    y = placer.base_y - layer * placer.step
    ax.axvline(x, color=color, ls="--", lw=1.2, alpha=0.8)
    ax.text(
        x, y, text, color=color, rotation=90,
        ha="center", va="top", fontsize=9,
        transform=ax.get_xaxis_transform()
    )

# ---------------- plotting ----------------
def plot_metrics_with_labels(
    x_grid: np.ndarray, curves: Dict[str, np.ndarray],
    rec: Dict[str, Any], out_png: Path, title: str
):
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    # lines
    ax.plot(x_grid, curves["sensitivity"], label="Sensitivity", lw=2.2)
    ax.plot(x_grid, curves["specificity"], label="Specificity", lw=2.2)
    ax.plot(x_grid, curves["precision"],   label="Precision",   lw=2.2)
    ax.plot(x_grid, curves["accuracy"],    label="Accuracy",    lw=2.2)
    ax.plot(x_grid, curves["brier"],       label="Brier",       lw=2.2)

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    # de-overlap labels under x-axis
    placer = LabelPlacer(x_tol=0.02, base_y=-0.10, step=0.045, max_layers=8)

    # Youden
    y_thr = float(rec["youden_thr"])
    annotate_vertical_label(ax, placer, y_thr, f"Youden\n@{y_thr:.3f}", color="#7b1fa2")  # purple-ish

    # sens targets
    for s, t in rec["sens_targets"]:
        if t is not None:
            annotate_vertical_label(ax, placer, float(t), f"sens≥{s:.2f}\n@{t:.3f}", color="#2e7d32")  # green

    # spec targets
    for s, t in rec["spec_targets"]:
        if t is not None:
            annotate_vertical_label(ax, placer, float(t), f"spec≥{s:.2f}\n@{t:.3f}", color="#1565c0")  # blue

    # adjust bottom margin so labels are visible
    bottom = placer.bottom_margin()
    fig.subplots_adjust(bottom=bottom)

    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png, dpi=180)
    plt.close(fig)

def plot_fscore_with_labels(
    x_grid: np.ndarray, f1: np.ndarray, rec: Dict[str, Any], out_png: Path, title: str
):
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x_grid, f1, lw=2.4, label="F1.0")
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1.0")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    placer = LabelPlacer(x_tol=0.02, base_y=-0.10, step=0.045, max_layers=8)

    # F1 max
    f1_thr = float(rec["f1_thr"])
    annotate_vertical_label(ax, placer, f1_thr, f"F1-max\n@{f1_thr:.3f}", color="#7b1fa2")

    # sens/spec targets（可视化同上，方便对齐）
    for s, t in rec["sens_targets"]:
        if t is not None:
            annotate_vertical_label(ax, placer, float(t), f"sens≥{s:.2f}\n@{t:.3f}", color="#2e7d32")
    for s, t in rec["spec_targets"]:
        if t is not None:
            annotate_vertical_label(ax, placer, float(t), f"spec≥{s:.2f}\n@{t:.3f}", color="#1565c0")

    fig.subplots_adjust(bottom=placer.bottom_margin())
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png, dpi=180)
    plt.close(fig)

# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Threshold scanning")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="isotonic", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--thr_min", type=float, default=0.01)
    ap.add_argument("--thr_max", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.001)
    ap.add_argument("--sens_targets", type=float, nargs="*", default=[0.90])
    ap.add_argument("--spec_targets", type=float, nargs="*", default=[0.80])
    ap.add_argument("--heartbeat", type=float, default=5.0)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables", "outputs/tables")); ensure_dir(tables_dir)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))

    # MI
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = load_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    selected = read_selected_features(artifacts_dir)

    # collect proba
    X_ref_df, y_te, prob_ens, feat_used = average_probs_across_m(
        model_name=args.model, method=args.method, m_paths=m_paths,
        selected_for_pred=selected, heartbeat_sec=float(args.heartbeat)
    )
    y = y_te.astype(int)
    p = prob_ens.astype(float)

    # scan
    thr_min = max(1e-6, float(args.thr_min))
    thr_max = min(1 - 1e-6, float(args.thr_max))
    step = float(args.step)
    grid = np.arange(thr_min, thr_max + 1e-12, step, dtype=float)

    rows = []
    for t in grid:
        m = threshold_metrics(y, p, float(t))
        m["thr"] = float(t)
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["thr", "sensitivity", "specificity", "precision", "npv", "accuracy", "f1", "brier", "tp", "fp", "tn", "fn"]
    df = df[cols]

    # recommendations
    rec = recommend_thresholds(y, p, args.sens_targets, args.spec_targets)

    # save csv/json
    suffix = "" if args.method == "raw" else f"_{args.method}"
    scan_csv = outputs_dir / f"threshold_scan_{args.model}{suffix}.csv"
    atomic_to_csv(df, scan_csv)

    summary = {
        "grid_min": thr_min,
        "grid_max": thr_max,
        "step": step,
        "n": int(len(grid)),
        "prevalence": float(np.mean(y)),
        "youden_thr": rec["youden_thr"],
        "f1_thr": rec["f1_thr"],
        "sens_targets": rec["sens_targets"],
        "spec_targets": rec["spec_targets"],
    }
    scan_json = outputs_dir / f"threshold_scan_{args.model}{suffix}_summary.json"
    scan_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # plots (with non-overlapping vertical labels)
    title_base = f"{args.model} | {args.method}  (prev={np.mean(y):.3f})"
    curves = {
        "sensitivity": df["sensitivity"].values,
        "specificity": df["specificity"].values,
        "precision":   df["precision"].values,
        "accuracy":    df["accuracy"].values,
        "brier":       df["brier"].values,
    }
    fig1 = outputs_dir / f"thr_metrics_{args.model}{suffix}.png"
    plot_metrics_with_labels(grid, curves, rec, fig1, f"Threshold metrics – {title_base}")

    fig2 = outputs_dir / f"thr_fscore_{args.model}{suffix}.png"
    plot_fscore_with_labels(grid, df["f1"].values, rec, fig2, f"F1.0 across thresholds – {args.model} | {args.method}")

    # console
    print("[ok] 阈值扫描完成：")
    print(f"  - grid: [{thr_min:.3f}, {thr_max:.3f}] step={step:.3f} (n={len(grid)})  prevalence={np.mean(y):.4f}")
    print(f"  - 推荐（Youden）: thr={rec['youden_thr']:.3f}")
    print(f"  - 推荐（F1-max）: thr={rec['f1_thr']:.3f}")
    for s, t in rec["sens_targets"]:
        if t is not None:
            print(f"  - 推荐（sens≥{s:.2f}）: thr={t:.3f}")
    for s, t in rec["spec_targets"]:
        if t is not None:
            print(f"  - 推荐（spec≥{s:.2f}）: thr={t:.3f}")
    print(f"[save] 扫描表: {scan_csv}")
    print(f"[save] 摘要  : {scan_json}")
    print(f"[save] 曲线1 : {fig1}")
    print(f"[save] 曲线2 : {fig2}")

if __name__ == "__main__":
    main()




# python -m src.threshold_scan --config conf/config.yaml --model random_forest --method isotonic --sens_targets 0.9
