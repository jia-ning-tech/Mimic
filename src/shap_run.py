# src/shap_run.py
# -*- coding: utf-8 -*-
"""
SHAP explanation on test set (best-m by AUC), aligned with selected features.

HARDENED:
- Always extract a *fitted* tree estimator (raw preferred; else unwrap calibrated)
- Provide background for interventional TreeExplainer
- Normalize SHAP outputs to 2D consistently; flatten importance to 1D
- Handle (n, n_features, 2) SHAP outputs (pick class=1 along the last axis)
- Unified utils from data_utils (atomic writes, heartbeat, style)

Outputs:
  - outputs/shap/shap_importance_<model>.csv
  - outputs/shap/shap_importance_bar_<model>.png
  - outputs/shap/shap_summary_<model>.png
  - (optional) outputs/shap/<model>_model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ---- common utils ----
from .data_utils import (
    load_yaml, ensure_dir,
    read_json, write_json,
    Heartbeat, apply_mpl_style,
    atomic_to_csv, atomic_savefig,
)

# ---------------- helpers (project-specific) ----------------
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

def choose_m_index_by_auc(
    m_paths: List[Path], model_paths: List[Path], selected_feats: Optional[List[str]]
) -> int:
    """Choose m (1-based) with highest AUC on test set using given model_paths (method can be raw/isotonic/sigmoid)."""
    aucs: List[float] = []
    for i, (mi_p, mdl_p) in enumerate(zip(m_paths, model_paths), start=1):
        art = joblib.load(mi_p)
        feat_order = list(art["feature_order"])
        X_te_df = pd.DataFrame(art["X_test"], columns=feat_order)
        y_te = np.asarray(art["y_test"]).astype(int)

        if selected_feats:
            cols = [c for c in selected_feats if c in X_te_df.columns]
            if len(cols) != len(selected_feats):
                miss = [c for c in selected_feats if c not in X_te_df.columns]
                raise RuntimeError(f"selected_features.json 中的列缺失于测试矩阵：{miss}")
            X = X_te_df[cols].values.astype("float32")
        else:
            X = X_te_df.values.astype("float32")

        if not mdl_p.exists():
            raise FileNotFoundError(f"未找到模型文件：{mdl_p}；请先训练/校准。")
        est = joblib.load(mdl_p)
        prob = get_proba(est, X)
        if np.unique(y_te).size < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_te, prob)))
    idx = int(np.nanargmax(np.array(aucs)))
    return idx + 1

# ---------------- fitted tree retrieval ----------------
def _pipeline_tail(est):
    """If est is a Pipeline, return its last step estimator; else return est."""
    if isinstance(est, Pipeline):
        return est.steps[-1][1]
    return est

def is_fitted_tree(est) -> bool:
    """Heuristic to tell if a (tree/forest/gbdt) estimator is fitted."""
    if hasattr(est, "estimators_"):
        try:
            _ = len(est.estimators_)
            return True
        except Exception:
            return False
    if hasattr(est, "n_features_in_"):
        return True
    return False

def unwrap_base_estimator(est):
    """
    Try to unwrap to a base estimator (may or may not be fitted).
    Covers: Pipeline / CalibratedClassifier / common attributes.
    """
    est = _pipeline_tail(est)
    base = getattr(est, "base_estimator", None) or getattr(est, "estimator", None)
    if base is not None:
        return _pipeline_tail(base)
    cc_list = getattr(est, "calibrated_classifiers_", None)
    if isinstance(cc_list, (list, tuple)) and len(cc_list) > 0:
        cc0 = cc_list[0]
        base = getattr(cc0, "base_estimator", None) or getattr(cc0, "estimator", None)
        if base is not None:
            return _pipeline_tail(base)
    return est

def get_fitted_tree_estimator(
    model_name: str, m_index: int, method: str
):
    """
    Return a *fitted* tree estimator to feed TreeExplainer.
    Priority:
      1) RAW model if exists and fitted
      2) From calibrated model (method) via calibrated_classifiers_[0].base_estimator
      3) From other wrappers (CalibratedClassifier / Pipeline)
    """
    # 1) RAW
    raw_path = list_model_path(model_name, "raw", m_index)
    if raw_path.exists():
        est_raw = joblib.load(raw_path)
        est_raw = unwrap_base_estimator(est_raw)
        if is_fitted_tree(est_raw):
            return est_raw, raw_path, "raw-fitted"
    # 2) calibrated (method)
    if method != "raw":
        cali_path = list_model_path(model_name, method, m_index)
        if cali_path.exists():
            est_cali = joblib.load(cali_path)
            cc_list = getattr(est_cali, "calibrated_classifiers_", None)
            if isinstance(cc_list, (list, tuple)) and len(cc_list) > 0:
                cc0 = cc_list[0]
                base = getattr(cc0, "base_estimator", None) or getattr(cc0, "estimator", None)
                base = _pipeline_tail(base) if base is not None else None
                if base is not None and is_fitted_tree(base):
                    return base, cali_path, "isotonic-unwrapped" if method == "isotonic" else "sigmoid-unwrapped"
            base2 = unwrap_base_estimator(est_cali)
            if is_fitted_tree(base2):
                return base2, cali_path, f"{method}-unwrapped-generic"
    # 3) try other calibrated variants
    for mtd in ("isotonic", "sigmoid"):
        cali_path = list_model_path(model_name, mtd, m_index)
        if cali_path.exists():
            est_cali = joblib.load(cali_path)
            cc_list = getattr(est_cali, "calibrated_classifiers_", None)
            if isinstance(cc_list, (list, tuple)) and len(cc_list) > 0:
                cc0 = cc_list[0]
                base = getattr(cc0, "base_estimator", None) or getattr(cc0, "estimator", None)
                base = _pipeline_tail(base) if base is not None else None
                if base is not None and is_fitted_tree(base):
                    return base, cali_path, f"{mtd}-unwrapped"
            base2 = unwrap_base_estimator(est_cali)
            if is_fitted_tree(base2):
                return base2, cali_path, f"{mtd}-unwrapped-generic"
    raise RuntimeError(
        "未能找到已拟合的树模型用于 SHAP。请先确保已训练/校准并存在 outputs/models/*_mXX.joblib。"
    )

# ---------------- SHAP normalization helpers ----------------
def _normalize_shap_array(v: Any) -> np.ndarray:
    """
    Ensure SHAP output is a 2D array (n_samples, n_features).
    Handles:
      - list-of-arrays -> pick class 1 if exists
      - (n, n_features, 1) -> squeeze
      - (n, n_features, 2) -> take last axis index 1 (positive class)
      - (2, n, n_features) or (1, n, n_features) -> pick class then squeeze
    """
    if isinstance(v, list):
        if len(v) == 2:
            v = v[1]
        else:
            v = v[0]
    arr = np.asarray(v)

    # If 3D with class on last axis (n, k, 1/2)
    if arr.ndim == 3 and arr.shape[-1] in (1, 2):
        arr = arr[:, :, 1] if arr.shape[-1] == 2 else np.squeeze(arr, axis=-1)

    # squeeze remaining singletons
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 2):
        # class-major layout (c, n, k)
        arr = np.squeeze(arr[1] if arr.shape[0] == 2 else arr[0])

    if arr.ndim != 2:
        raise RuntimeError(f"SHAP 返回形状异常: {arr.shape}, 期望二维 (n, n_features)")
    return arr

def batched_shap_values(explainer, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Compute SHAP values in batches to reduce memory footprint and normalize to 2D."""
    n = X.shape[0]
    vals: List[np.ndarray] = []
    for i in tqdm(range(0, n, batch_size), desc="SHAP", unit="row"):
        xb = X[i : min(i + batch_size, n)]
        vb = explainer.shap_values(xb)
        vb2 = _normalize_shap_array(vb)
        vals.append(vb2)
    return np.vstack(vals)

# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run SHAP explanation on test set.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"],
                    help="用于选择AUC最佳 m 的模型 & 尝试解包校准以获得已拟合底模")
    ap.add_argument("--m_only", type=int, default=0, help="若>0，则仅解释指定的 m（1-based），跳过自动选择。")
    ap.add_argument("--max_n", type=int, default=0, help="仅取测试集前 max_n 条做 SHAP（0=全量）。")
    ap.add_argument("--batch_size", type=int, default=256, help="SHAP 分批大小，控制内存峰值。")
    ap.add_argument("--heartbeat", type=float, default=5.0)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    shap_dir = outputs_dir / "shap"; ensure_dir(shap_dir)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))
    mi_dir = Path(cfg.get("missing_data", {}).get("mice", {}).get("mice_output_dir", "outputs/mi_runs"))

    mi_index = read_mi_index(mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    if not m_paths:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    selected = read_selected_features(artifacts_dir)

    # 1) 选择用于解释的 m：默认根据 args.method 的 AUC 最优；也可 --m_only 指定
    if args.m_only and 1 <= int(args.m_only) <= len(m_paths):
        m_idx = int(args.m_only)
        print(f"[info] 按指定 m={m_idx} 做解释（method={args.method}）")
    else:
        model_paths = [list_model_path(args.model, args.method, i + 1) for i in range(len(m_paths))]
        m_idx = choose_m_index_by_auc(m_paths, model_paths, selected)
        print(f"[info] 选择 AUC 最好的 m 进行解释（如需指定，请用 --m_only k）")
        print(f"[info] 解释使用 m={m_idx} 的模型（method=raw 优先，若无则解包 {args.method}）")

    # 2) 读取 m_idx 的测试集，并对齐 selected_features
    art = joblib.load(m_paths[m_idx - 1])
    feat_order = list(art["feature_order"])
    X_te_df = pd.DataFrame(art["X_test"], columns=feat_order)
    y_te = np.asarray(art["y_test"]).astype(int)

    if selected:
        cols = [c for c in selected if c in X_te_df.columns]
        if len(cols) != len(selected):
            miss = [c for c in selected if c not in X_te_df.columns]
            raise RuntimeError(f"selected_features.json 中的列缺失于测试矩阵：{miss}")
        X = X_te_df[cols].values.astype("float32")
        feat_names = cols
    else:
        X = X_te_df.values.astype("float32")
        feat_names = feat_order

    n_total = X.shape[0]
    if args.max_n and int(args.max_n) > 0:
        n_use = min(int(args.max_n), n_total)
    else:
        n_use = n_total
    X = X[:n_use]
    y_te = y_te[:n_use]
    print(f"[info] 测试集用于SHAP的样本: {n_use} / {n_total}，特征数: {len(feat_names)}")

    # 3) 获取“已拟合”的树模型
    with Heartbeat(prefix=f"[hb] fetch fitted tree", interval=float(args.heartbeat)):
        base_for_shap, used_path, how = get_fitted_tree_estimator(args.model, m_idx, args.method)
    print(f"[info] SHAP 使用底模：{used_path.name} [{how}]")

    # 4) 提供 background，构建 explainer
    rng = np.random.default_rng(42)
    bg_n = min(200, X.shape[0])
    background = X[rng.choice(X.shape[0], size=bg_n, replace=False)] if bg_n > 0 else None

    with Heartbeat(prefix=f"[hb] shap explainer", interval=float(args.heartbeat)):
        explainer = shap.TreeExplainer(
            base_for_shap,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )

    # 5) 计算 SHAP（分批），并汇总重要度（确保为 1D）
    shap_vals = batched_shap_values(explainer, X, batch_size=int(args.batch_size))  # (n, k)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_abs = np.asarray(mean_abs).reshape(-1).astype(float)
    if len(mean_abs) != len(feat_names):
        raise RuntimeError(f"SHAP 重要度长度与特征不一致: {len(mean_abs)} vs {len(feat_names)}")

    imp_df = (
        pd.DataFrame({"feature": list(feat_names), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp_csv = shap_dir / f"shap_importance_{args.model}.csv"
    atomic_to_csv(imp_df, imp_csv)

    # 6) 绘图（bar + swarm）
    apply_mpl_style()

    # bar
    fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
    show_df = imp_df.head(min(20, len(imp_df))).iloc[::-1]  # top-N reverse for horizontal bar
    ax1.barh(show_df["feature"], show_df["mean_abs_shap"], lw=0)
    ax1.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax1.set_title(f"SHAP importance – {args.model}")
    plt.tight_layout()
    bar_png = shap_dir / f"shap_importance_bar_{args.model}.png"
    atomic_savefig(fig1, bar_png)
    plt.close(fig1)

    # swarm
    fig2 = plt.figure(figsize=(6.4, 4.8))
    try:
        shap.summary_plot(
            shap_vals, X, feature_names=feat_names, show=False, plot_type="dot", max_display=20
        )
        plt.title(f"SHAP summary – {args.model}")
        plt.tight_layout()
        swarm_png = shap_dir / f"shap_summary_{args.model}.png"
        atomic_savefig(plt.gcf(), swarm_png)
    finally:
        plt.close(fig2)

    # 可选保存底层模型（便于追踪）
    mdl_out = shap_dir / f"{args.model}_model.joblib"
    try:
        joblib.dump(base_for_shap, mdl_out)
    except Exception:
        pass

    # 控制台输出
    print("[ok] SHAP 完成：")
    print(f"  - explained m = {m_idx} [{how}]")
    print(f"  - importance CSV: {imp_csv}")
    print(f"  - bar:   {bar_png}")
    print(f"  - swarm: {swarm_png}")

if __name__ == "__main__":
    main()



# # 用未校准模型，默认选 AUC 最好的 m，最多取 1000 个样本、批大小 256
# python -m src.shap_run --config conf/config.yaml --model random_forest --max_n 1000 --batch_size 256

# # 用各 m 的 isotonic 校准模型，强制用 m=2，并画某个测试样本的 waterfall
# python -m src.shap_run --config conf/config.yaml --model random_forest --method isotonic --m_only 2 --waterfall_index 316
