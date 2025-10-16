# src/subgroup_eval.py
# -*- coding: utf-8 -*-
"""
Subgroup evaluation with MI-ensemble probabilities.

Refactor highlights:
- Import shared utilities from data_utils (atomic writes / heartbeat / style / metrics)
- Keep previous behavior and outputs unchanged
- Auto-NaN for single-class subgroups; configurable minimal pos/neg thresholds
- Quiet mode to silence sklearn UndefinedMetricWarning
- Strict feature alignment to selected_features.json
- Supports raw | isotonic | sigmoid calibrated models (per-m), then avg probs across M

Outputs:
  - outputs/subgroup_metrics_<model>[ _<method>].csv
  - outputs/figures/subgroup_forest_<model>[ _<method>].png
"""

from __future__ import annotations

import argparse
import json
import warnings
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

from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# ---- import common tools from data_utils ----
from .data_utils import (
    load_yaml, ensure_dir,
    read_json, write_json,
    Heartbeat, apply_mpl_style,
    threshold_metrics, atomic_to_csv, atomic_savefig
)

# ---------------- helpers (local, non-generic) ----------------
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

def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))

# ---------------- subgroup definition ----------------
def default_subgroups(X: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Build boolean masks for subgroups based on config lists and existing columns.
    - gender (if present): male / female / other
    - age bins: <50, 50-64, 65-79, >=80  (if 'age' exists)
    - comorbidities from cfg['features']['comorbidities'] (binary 0/1 columns)
    """
    masks: Dict[str, np.ndarray] = {}
    cols = set(X.columns.tolist())

    # gender
    if "gender" in cols:
        # assume original gender is 0/1 or 'M'/'F' or one-hot columns
        g = X["gender"]
        if pd.api.types.is_numeric_dtype(g):
            masks["Gender: Male"] = (g == 1).to_numpy()
            masks["Gender: Female"] = (g == 0).to_numpy()
        else:
            gs = g.astype(str).str.lower()
            masks["Gender: Male"] = (gs.str.startswith("m")).to_numpy()
            masks["Gender: Female"] = (gs.str.startswith("f")).to_numpy()

    # age bins
    if "age" in cols:
        a = X["age"].astype(float)
        masks["Age <50"]       = (a < 50).to_numpy()
        masks["Age 50-64"]     = ((a >= 50) & (a <= 64)).to_numpy()
        masks["Age 65-79"]     = ((a >= 65) & (a <= 79)).to_numpy()
        masks["Age ≥80"]       = (a >= 80).to_numpy()

    # comorbidities
    comorb = cfg.get("features", {}).get("comorbidities", []) or []
    for c in comorb:
        if c in cols:
            masks[f"Comorb: {c}"] = (X[c] == 1).to_numpy()

    return masks

# ---------------- plotting ----------------
def plot_forest_auc(df: pd.DataFrame, out_png: Path, title: str):
    """
    Expect df columns: subgroup, n, prevalence, auc, auc_lo, auc_hi
    """
    apply_mpl_style()
    plot_df = df.copy().reset_index(drop=True)
    # order by AUC (nanlast)
    plot_df["auc_plot"] = plot_df["auc"].fillna(-1.0)
    plot_df = plot_df.sort_values("auc_plot", ascending=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(7.0, 0.44 * (len(plot_df) + 6)))

    # CIs
    for i, r in plot_df.iterrows():
        auc = r["auc"]
        lo  = r.get("auc_lo", np.nan)
        hi  = r.get("auc_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y=i, xmin=lo, xmax=hi, color="tab:blue", lw=2, alpha=0.8)
        if np.isfinite(auc):
            ax.plot([auc], [i], "o", color="tab:blue")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['subgroup']}  (N={int(r['n'])}, prev={r['prevalence']:.2f})" for _, r in plot_df.iterrows()], fontsize=9)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    atomic_savefig(fig, out_png)
    plt.close(fig)

# ---------------- bootstrap CI (AUC) ----------------
def bootstrap_auc_ci(y_true: np.ndarray, prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    n = len(y)
    aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]; pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.nanquantile(aucs, 0.025))
    hi = float(np.nanquantile(aucs, 0.975))
    return (lo, hi)

# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Subgroup evaluation with MI-ensemble probabilities.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    ap.add_argument("--thr", type=float, default=float("nan"), help="Decision threshold; if NaN, auto from threshold_scan summary (Youden) else 0.5 fallback.")
    ap.add_argument("--min_pos", type=int, default=5, help="Minimal positives to compute subgroup metrics; otherwise NaN.")
    ap.add_argument("--min_neg", type=int, default=5, help="Minimal negatives to compute subgroup metrics; otherwise NaN.")
    ap.add_argument("--n_boot",  type=int, default=1000, help="Bootstrap iterations for AUC CI.")
    ap.add_argument("--heartbeat", type=float, default=5.0)
    ap.add_argument("--quiet", action="store_true", help="Silence UndefinedMetricWarning etc.")
    args = ap.parse_args(argv)

    if args.quiet:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    cfg = load_yaml(Path(args.config))
    # outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    # ✅ 新增/改成这两行（若文件中已有 figures_dir，可保留；关键是加 tables_dir 并指向 outputs/tables）
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables",  "outputs/tables"));  ensure_dir(tables_dir)
    python
    复制代码

    # figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures")); ensure_dir(figures_dir)
    # tables_dir  = Path(cfg.get("output", {}).get("tables", "outputs/tables")); ensure_dir(tables_dir)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))

    # MI artifacts
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))
    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    selected = read_selected_features(artifacts_dir)

    # assemble probs across M
    with Heartbeat(prefix=f"[hb] subgroup ({args.model})", interval=float(args.heartbeat)):
        X_ref_df, y_te, prob_ens, feat_used = average_probs_across_m(
            model_name=args.model, method=args.method, m_paths=m_paths,
            selected_for_pred=selected, heartbeat_sec=float(args.heartbeat)
        )

    prev = float(np.mean(y_te))

    # choose threshold
    suffix = "" if args.method == "raw" else f"_{args.method}"
    thr = args.thr
    if np.isnan(thr):
        summ_path = outputs_dir / f"threshold_scan_{args.model}{suffix}_summary.json"
        summ = read_json(summ_path)
        if summ and "youden" in summ and "thr" in summ["youden"]:
            thr = float(summ["youden"]["thr"])
        else:
            thr = 0.5
    print(f"[info] 使用阈值 thr={thr:.3f} 进行亚组评估")

    # build subgroups
    sg_masks = default_subgroups(X_ref_df, cfg)

    # evaluate overall row first
    rows: List[Dict[str, Any]] = []
    auc_overall = safe_auc(y_te, prob_ens)
    # overall sens/spec/precision/f1/acc/brier at thr
    met_overall = threshold_metrics(y_te, prob_ens, thr)
    rows.append({
        "subgroup": "Overall",
        "n": int(len(y_te)),
        "prevalence": prev,
        "auc": auc_overall,
        "auc_lo": np.nan,
        "auc_hi": np.nan,
        **{k: met_overall[k] for k in ["sensitivity", "specificity", "precision", "npv", "accuracy", "f1", "brier", "tp", "fp", "tn", "fn"]},
    })

    # per-subgroup evaluation
    for name, mask in tqdm(sg_masks.items(), desc="Subgroups", unit="sg"):
        idx = mask.astype(bool)
        n = int(idx.sum())
        if n == 0:
            # empty subgroup
            rows.append({
                "subgroup": name, "n": 0, "prevalence": np.nan,
                "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "sensitivity": np.nan, "specificity": np.nan, "precision": np.nan,
                "npv": np.nan, "accuracy": np.nan, "f1": np.nan, "brier": np.nan,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0
            })
            continue

        y_sg = y_te[idx]
        p_sg = prob_ens[idx]
        pos = int((y_sg == 1).sum())
        neg = n - pos
        prev_sg = float(pos / n) if n > 0 else np.nan

        # compute metrics if both classes present and pass minimal counts
        if pos >= int(args.min_pos) and neg >= int(args.min_neg) and np.unique(y_sg).size == 2:
            auc = safe_auc(y_sg, p_sg)
            lo, hi = bootstrap_auc_ci(y_sg, p_sg, n_boot=int(args.n_boot), seed=42)
            m = threshold_metrics(y_sg, p_sg, thr)
            row = {
                "subgroup": name, "n": n, "prevalence": prev_sg,
                "auc": auc, "auc_lo": lo, "auc_hi": hi,
                **{k: m[k] for k in ["sensitivity", "specificity", "precision", "npv", "accuracy", "f1", "brier", "tp", "fp", "tn", "fn"]},
            }
        else:
            # single-class / too small → NaN
            row = {
                "subgroup": name, "n": n, "prevalence": prev_sg,
                "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "sensitivity": np.nan, "specificity": np.nan, "precision": np.nan,
                "npv": np.nan, "accuracy": np.nan, "f1": np.nan, "brier": np.nan,
                "tp": np.nan, "fp": np.nan, "tn": np.nan, "fn": np.nan
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    # save table
    # out_csv = outputs_dir / f"subgroup_metrics_{args.model}{suffix}.csv"
    # atomic_to_csv(df, out_csv)
    out_csv = tables_dir / f"subgroup_metrics_{args.model}{suffix}.csv"
    atomic_to_csv(sub_df, out_csv)

    # forest plot (AUC)
    out_png = Path(cfg.get("output", {}).get("figures", "outputs/figures")) / f"subgroup_forest_{args.model}{suffix}.png"
    plot_forest_auc(df[["subgroup","n","prevalence","auc","auc_lo","auc_hi"]], out_png,
                    title=f"Subgroup AUC – {args.model}{'' if args.method=='raw' else ' | '+args.method} (thr={thr:.3f})")

    fig_path = figures_dir / f"subgroup_forest_{args.model}{suffix}.png"
    atomic_savefig(fig, fig_path)

    # console
    print("[ok] Subgroup 评估完成：")
    print(f"  - 输出表: {out_csv}")
    print(f"  - 森林图: {out_png}")

if __name__ == "__main__":
    main()


# python -m src.subgroup_eval --config conf/config.yaml --model random_forest --method isotonic
# # 如需更严格或更宽松门槛：
# python -m src.subgroup_eval --config conf/config.yaml --model random_forest --method isotonic --min_pos 3 --min_neg 3 --min_n 15
