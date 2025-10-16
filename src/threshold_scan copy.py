# src/threshold_scan.py
# -*- coding: utf-8 -*-
"""
Threshold scanning with MI-ensemble probabilities.

Changes in this refactor:
- Import shared utilities from data_utils (atomic writes / heartbeat / style / metrics)
- Keep previous behavior and outputs unchanged
- Vertical annotations placed BELOW x-axis to avoid overlaps

Outputs:
  - outputs/threshold_scan_<model>[ _<method>].csv
  - outputs/threshold_scan_<model>[ _<method>]_summary.json
  - outputs/thr_metrics_<model>[ _<method>].png
  - outputs/thr_fscore_<model>[ _<method>].png
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

# tqdm (optional fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---- import common tools from data_utils ----
from .data_utils import (
    load_yaml, ensure_dir,
    read_json, write_json,
    Heartbeat, apply_mpl_style,
    threshold_metrics, atomic_to_csv, atomic_savefig
)


# ---------------- helpers (non-generic, keep local) ----------------
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
    """Return P(y=1)."""
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
      X_test_df (with full columns), y_test (1d), prob_ensemble (1d), used_features (list)
    """
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


# ---------------- scanning & recommendations ----------------
def compute_scan_table(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    recs = []
    for t in thresholds:
        m = threshold_metrics(y_true, prob, float(t))
        m["thr"] = float(t)
        recs.append(m)
    df = pd.DataFrame(recs)
    # consistent order
    cols = ["thr", "sensitivity", "specificity", "precision", "npv", "accuracy", "f1", "brier", "tp", "fp", "tn", "fn"]
    return df[cols]


def youden_recommend(df: pd.DataFrame) -> Dict[str, float]:
    # maximize sens+spec-1 ; tie-breaker: larger sens
    tmp = df.copy()
    tmp["youden"] = tmp["sensitivity"] + tmp["specificity"] - 1.0
    ix = tmp.sort_values(["youden", "sensitivity", "specificity"], ascending=[False, False, False]).index[0]
    r = tmp.loc[ix]
    return {"thr": float(r["thr"]), "sens": float(r["sensitivity"]), "spec": float(r["specificity"])}


def fbeta_recommend(df: pd.DataFrame, beta: float = 1.0) -> Dict[str, float]:
    b2 = float(beta) ** 2
    prec = df["precision"].to_numpy(dtype=float)
    rec  = df["sensitivity"].to_numpy(dtype=float)
    denom = (b2 * prec + rec)
    fbeta = (1 + b2) * prec * rec / np.where(denom > 0, denom, np.inf)
    i = int(np.nanargmax(fbeta))
    return {"thr": float(df.iloc[i]["thr"]), "fbeta": float(fbeta[i]), "prec": float(prec[i]), "recall": float(rec[i])}


def target_sens_spec(df: pd.DataFrame, sens_targets: List[float], spec_targets: List[float]) -> Dict[str, Dict[str, float]]:
    rec: Dict[str, Dict[str, float]] = {}
    if sens_targets:
        # pick lowest threshold achieving sens >= target
        for s in sens_targets:
            ok = df[df["sensitivity"] >= s]
            if len(ok) == 0:
                thr = float(df["thr"].min())
                row = df.sort_values("thr", ascending=True).iloc[0]
            else:
                row = ok.sort_values("thr", ascending=True).iloc[0]
                thr = float(row["thr"])
            rec[f"sens>={s:.3f}"] = {"thr": thr, "sens": float(row["sensitivity"]), "spec": float(row["specificity"])}
    if spec_targets:
        # pick highest threshold achieving spec >= target
        for s in spec_targets:
            ok = df[df["specificity"] >= s]
            if len(ok) == 0:
                thr = float(df["thr"].max())
                row = df.sort_values("thr", ascending=False).iloc[0]
            else:
                row = ok.sort_values("thr", ascending=True).iloc[0]
                thr = float(row["thr"])
            rec[f"spec>={s:.3f}"] = {"thr": thr, "sens": float(row["sensitivity"]), "spec": float(row["specificity"])}
    return rec


# ---------------- plotting ----------------
def _vline_with_label_below(ax, x: float, label: str, color: str = "tab:red", ymin_pad: float = 0.12):
    """
    Draw a vertical dashed line at x, and put text BELOW the x-axis area vertically.
    ymin_pad: fraction of axis height to extend. We will expand ylim to negative.
    """
    # draw vline
    ax.axvline(x, ls="--", lw=1, color=color, alpha=0.9)
    # current limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # extend bottom margin for labels
    if ymin > -ymin_pad:
        ax.set_ylim(bottom=-ymin_pad)
        ymin = -ymin_pad
    # place vertical text just below 0 level
    ax.text(
        x, ymin + 0.01, label,
        rotation=90, va="bottom", ha="center", fontsize=8,
        color=color, clip_on=False
    )


def plot_metrics(df: pd.DataFrame, out_png: Path, title: str, annotations: List[Tuple[float, str, str]]):
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df["thr"], df["sensitivity"], label="Sensitivity", lw=2)
    ax.plot(df["thr"], df["specificity"], label="Specificity", lw=2)
    ax.plot(df["thr"], df["precision"],   label="Precision",   lw=1.5, alpha=0.9)
    ax.plot(df["thr"], df["accuracy"],    label="Accuracy",    lw=1.5, alpha=0.9)
    ax.plot(df["thr"], df["brier"],       label="Brier",       lw=1.2, alpha=0.9)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)

    # annotations below axis
    for x, lab, col in annotations:
        _vline_with_label_below(ax, float(x), lab, color=col)

    ax.legend(loc="best", ncol=3)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png)
    plt.close(fig)


def plot_fscore(df: pd.DataFrame, out_png: Path, title: str, beta: float, annotations: List[Tuple[float, str, str]]):
    apply_mpl_style()
    b2 = float(beta) ** 2
    prec = df["precision"].to_numpy(dtype=float)
    rec  = df["sensitivity"].to_numpy(dtype=float)
    denom = (b2 * prec + rec)
    fbeta = (1 + b2) * prec * rec / np.where(denom > 0, denom, np.inf)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["thr"], fbeta, label=f"F{beta:.1f}", lw=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel(f"F{beta:.1f}")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)

    # annotations below axis
    for x, lab, col in annotations:
        _vline_with_label_below(ax, float(x), lab, color=col)

    ax.legend(loc="best")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png)
    plt.close(fig)


# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Threshold scan with MI-ensemble probabilities.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--thr_min", type=float, default=0.01)
    ap.add_argument("--thr_max", type=float, default=0.99)
    ap.add_argument("--thr_step", type=float, default=0.001)
    ap.add_argument("--fbeta", type=float, default=1.0)
    ap.add_argument("--sens_targets", type=str, default="", help="comma separated, e.g. 0.8,0.9")
    ap.add_argument("--spec_targets", type=str, default="", help="comma separated, e.g. 0.8,0.9")
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

    # probabilities across M
    X_ref_df, y_te, prob_ens, feat_used = average_probs_across_m(
        model_name=args.model, method=args.method, m_paths=m_paths,
        selected_for_pred=selected, heartbeat_sec=float(args.heartbeat)
    )

    prev = float(np.mean(y_te))
    thr_min = max(1e-6, float(args.thr_min))
    thr_max = min(1 - 1e-6, float(args.thr_max))
    step = float(args.thr_step)
    thresholds = np.arange(thr_min, thr_max + 1e-12, step, dtype=float)

    # scan
    df = compute_scan_table(y_te, prob_ens, thresholds)

    # recommendations
    rec_youden = youden_recommend(df)
    rec_fbeta  = fbeta_recommend(df, beta=float(args.fbeta))
    sens_targets = [float(s) for s in args.sens_targets.split(",") if s.strip() != ""]
    spec_targets = [float(s) for s in args.spec_targets.split(",") if s.strip() != ""]
    rec_targets  = target_sens_spec(df, sens_targets=sens_targets, spec_targets=spec_targets)

    # save CSV + summary JSON
    suffix = "" if args.method == "raw" else f"_{args.method}"
    scan_csv = outputs_dir / f"threshold_scan_{args.model}{suffix}.csv"
    summary_json = outputs_dir / f"threshold_scan_{args.model}{suffix}_summary.json"

    atomic_to_csv(df, scan_csv)
    write_json({
        "grid": {"min": thr_min, "max": thr_max, "step": step, "n": len(df)},
        "prevalence": prev,
        "youden": rec_youden,
        f"f{args.fbeta:.1f}_max": rec_fbeta,
        "targets": rec_targets
    }, summary_json)

    # plots with vertical labels under x-axis
    annos: List[Tuple[float, str, str]] = []
    annos.append((rec_youden["thr"], f"Youden={rec_youden['thr']:.3f}", "tab:red"))
    annos.append((rec_fbeta["thr"],  f"F{args.fbeta:.1f}={rec_fbeta['thr']:.3f}", "tab:purple"))
    # targets
    for k, v in rec_targets.items():
        annos.append((v["thr"], f"{k}\n@{v['thr']:.3f}", "tab:green"))

    metrics_png = outputs_dir / f"thr_metrics_{args.model}{suffix}.png"
    fscore_png  = outputs_dir / f"thr_fscore_{args.model}{suffix}.png"

    plot_metrics(
        df, metrics_png,
        title=f"Threshold metrics – {args.model}{'' if args.method=='raw' else ' | '+args.method}  (prev={prev:.3f})",
        annotations=annos
    )
    plot_fscore(
        df, fscore_png,
        title=f"F{args.fbeta:.1f} across thresholds – {args.model}{'' if args.method=='raw' else ' | '+args.method}",
        beta=float(args.fbeta),
        annotations=annos
    )

    # console
    print("[ok] 阈值扫描完成：")
    print(f"  - grid: [{thr_min:.3f}, {thr_max:.3f}] step={step:.3f} (n={len(df)})  prevalence={prev:.4f}")
    print(f"  - 推荐（Youden）: thr={rec_youden['thr']:.3f}, sens={rec_youden['sens']:.3f}, spec={rec_youden['spec']:.3f}")
    print(f"  - 推荐（F{args.fbeta:.1f}-max）: thr={rec_fbeta['thr']:.3f}, fbeta={rec_fbeta['fbeta']:.3f}, "
          f"prec={rec_fbeta['prec']:.3f}, recall={rec_fbeta['recall']:.3f}")
    for k, v in rec_targets.items():
        print(f"  - 推荐（{k}）: thr={v['thr']:.3f}, sens={v['sens']:.3f}, spec={v['spec']:.3f}")
    print(f"[save] 扫描表: {scan_csv}")
    print(f"[save] 摘要  : {summary_json}")
    print(f"[save] 曲线1 : {metrics_png}")
    print(f"[save] 曲线2 : {fscore_png}")


if __name__ == "__main__":
    main()



# python -m src.threshold_scan --config conf/config.yaml --model random_forest --method isotonic --sens_targets 0.9
