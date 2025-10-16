# src/nested_cv.py
# -*- coding: utf-8 -*-
"""
Nested cross-validation with Optuna for ICU lymphoma ML.

- Use imputed+scaled data from MI artifacts (default m=1, configurable via --m_index)
- Outer StratifiedKFold for evaluation, inner Optuna for hyperparameter tuning
- Primary metric follows config.models.primary_metric (default: roc_auc)
- Heartbeat + tqdm, robust I/O
- Saves per-fold CSV, summary JSON, and an AUC boxplot figure

Outputs:
  CSV   : outputs/nested_cv/nested_cv_<model>.csv
  JSON  : outputs/nested_cv/nested_cv_<model>_summary.json
  FIG   : outputs/figures/nested_cv_auc_<model>.png
"""









# src/nested_cv.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Nested cross-validation with Optuna inner search.

- 外层 StratifiedKFold；内层 Optuna 超参搜索（主指标=ROC AUC）
- 读入已选特征（若存在 outputs/artifacts/selected_features.json）
- Pipeline：
  * 若 use_ros=True 且安装了 imbalanced-learn：使用 imblearn.pipeline.Pipeline，
    流程为 SimpleImputer → StandardScaler → RandomOverSampler → Estimator
  * 否则：使用 sklearn.pipeline.Pipeline（不含过采样）
- 类别不平衡：支持 config.imbalance.resampling_in_cv = "random_over_sampler"
- 类权重：支持 config.imbalance.use_class_weight / positive_class_weight
- tqdm 进度 + 心跳；稳健写盘
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline as SkPipeline  # 默认 pipeline

# tqdm 进度条
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# 可选：随机过采样 + imblearn 管道
_HAVE_IMB = True
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    _HAVE_IMB = False

# 项目内工具
from .data_utils import (
    load_yaml,
    ensure_dir,
    atomic_to_csv,
    Heartbeat,
)

# 统一模型工厂与搜索空间
from .models import make_model, suggest_params, apply_params


# -------------------- 小工具 --------------------
def load_processed_matrix(cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """读取 preprocess 产物（features_processed.parquet），并返回 X 与 y."""
    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data_processed"))
    mat_path = processed_dir / "features_processed.parquet"
    if not mat_path.exists():
        raise FileNotFoundError(f"未找到预处理矩阵：{mat_path}。请先运行 `python -m src.preprocess --config conf/config.yaml`。")
    df = pd.read_parquet(mat_path)
    y_col = cfg.get("data", {}).get("outcome_col", "mor_hospital")
    if y_col not in df.columns:
        raise KeyError(f"结局列 {y_col} 不在矩阵中。现有列：{list(df.columns)[:8]} ...")
    y = df[y_col].astype(int)
    X = df.drop(columns=[y_col])
    return X, y


def load_selected_features(cfg: dict) -> Optional[List[str]]:
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))
    sel_path = artifacts_dir / "selected_features.json"
    if not sel_path.exists():
        return None
    try:
        obj = json.loads(sel_path.read_text(encoding="utf-8"))
        feats = obj.get("selected_features")
        if isinstance(feats, list) and feats:
            return feats
    except Exception:
        pass
    return None


def class_weight_from_cfg(cfg: dict) -> Optional[dict | str]:
    imb = cfg.get("imbalance", {}) or {}
    use_cw = bool(imb.get("use_class_weight", True))
    if not use_cw:
        return None
    pcw = imb.get("positive_class_weight", None)
    if pcw is None:
        return "balanced"  # 兜底
    try:
        w1 = float(pcw)
        return {0: 1.0, 1: w1}
    except Exception:
        return "balanced"


def resampling_flag_from_cfg(cfg: dict) -> bool:
    imb = cfg.get("imbalance", {}) or {}
    return imb.get("resampling_in_cv", "random_over_sampler") == "random_over_sampler"


def pick_features(X: pd.DataFrame, selected: Optional[List[str]]) -> pd.DataFrame:
    if not selected:
        return X
    cols = [c for c in selected if c in X.columns]
    if len(cols) != len(selected):
        miss = [c for c in selected if c not in X.columns]
        raise RuntimeError(f"selected_features.json 中的列缺失于矩阵：{miss}")
    return X[cols]


def predict_proba_binary(clf: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    y = clf.predict(X)
    return y.astype(float)


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thr = np.linspace(0.01, 0.99, 99)
    youden = []
    for t in thr:
        pred = (prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        tn = np.sum((y_true == 0) & (pred == 0))
        fp = np.sum((y_true == 0) & (pred == 1))
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        if np.isnan(sens) or np.isnan(spec):
            youden.append(-np.inf)
        else:
            youden.append(sens + spec - 1.0)
    idx = int(np.nanargmax(np.asarray(youden)))
    return float(thr[idx])


# ---- 关键兜底：统一清洗 RandomForest 的 max_features=auto ----
def sanitize_rf_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """把 'auto' 替换为 'sqrt'，兼容不同键名前缀。返回新 dict，不改原对象。"""
    clean = dict(params)
    keys = []
    if "max_features" in clean:
        keys.append("max_features")
    if "clf__max_features" in clean:
        keys.append("clf__max_features")
    for k in keys:
        if clean.get(k) == "auto":
            clean[k] = "sqrt"
    return clean


# -------------------- 构建 Pipeline（支持 imblearn） --------------------
def build_pipeline(
    name: str,
    class_weight: Optional[dict | str],
    n_jobs: int,
    use_ros: bool,
):
    """
    - use_ros=True 且 _HAVE_IMB：使用 ImbPipeline，并包含 RandomOverSampler
    - 否则：使用 SkPipeline，不包含过采样
    """
    steps: List[Tuple[str, Any]] = []
    steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("scale", StandardScaler()))

    if use_ros and _HAVE_IMB:
        steps.append(("ros", RandomOverSampler(random_state=42)))
        pipe_cls = ImbPipeline  # imblearn 的 Pipeline 允许采样器出现在中间步骤
    elif use_ros and not _HAVE_IMB:
        warnings.warn("[nested_cv] 未安装 imbalanced-learn，无法使用 RandomOverSampler；将跳过过采样。")
        pipe_cls = SkPipeline
    else:
        pipe_cls = SkPipeline

    est = make_model(name, random_state=42, class_weight=class_weight, n_jobs=n_jobs)
    steps.append(("clf", est))
    return pipe_cls(steps)


# -------------------- Optuna 内层目标 --------------------
def inner_objective(trial, X_tr: np.ndarray, y_tr: np.ndarray, name: str, class_weight, n_jobs: int, use_ros: bool):
    import optuna  # 延迟导入

    params = suggest_params(trial, name)
    # 内层兜底：auto -> sqrt
    if name == "random_forest":
        params = sanitize_rf_params(params)

    pipe = build_pipeline(name, class_weight=class_weight, n_jobs=n_jobs, use_ros=use_ros)
    pipe = apply_params(pipe, {"clf__" + k: v for k, v in params.items()})

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs: List[float] = []
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        Xt, Xv = X_tr[tr_idx], X_tr[va_idx]
        yt, yv = y_tr[tr_idx], y_tr[va_idx]
        pipe.fit(Xt, yt)
        prob = predict_proba_binary(pipe, Xv)
        if np.unique(yv).size < 2:
            continue  # 单类折跳过
        aucs.append(roc_auc_score(yv, prob))
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


# -------------------- 主流程 --------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Nested cross-validation with Optuna inner search.")
    ap.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    ap.add_argument("--model", "-m", type=str, required=True)
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--heartbeat", type=float, default=5.0)
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    out_dir = Path("outputs") / "nested_cv"
    ensure_dir(out_dir)

    # 数据/特征
    X_df, y = load_processed_matrix(cfg)
    selected = load_selected_features(cfg)
    X_df = pick_features(X_df, selected)
    feat_names = list(X_df.columns)

    X = X_df.values.astype("float32")
    y = np.asarray(y).astype(int)

    # 类别权重 & 过采样开关
    cw = class_weight_from_cfg(cfg)
    use_ros = resampling_flag_from_cfg(cfg)

    # 外层 CV
    skf_outer = StratifiedKFold(n_splits=int(args.outer_folds), shuffle=True, random_state=int(args.seed))

    # 记录
    folds_rows: List[Dict[str, Any]] = []

    # 进度 + 心跳
    with Heartbeat(prefix="[hb] nested-cv", interval=float(args.heartbeat)):
        for k, (tr_idx, te_idx) in enumerate(
            tqdm(skf_outer.split(X, y), total=int(args.outer_folds), desc="Outer", unit="fold"), start=1
        ):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # —— 内层 Optuna 搜索 —— #
            try:
                import optuna
            except Exception as e:
                raise RuntimeError("需要 optuna 以运行嵌套CV内层搜索，请在环境中安装 optuna。") from e

            sampler = optuna.samplers.TPESampler(seed=int(args.seed))
            study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"inner_{args.model}_fold{k}")

            study.optimize(
                lambda t: inner_objective(t, X_tr, y_tr, args.model, cw, int(args.n_jobs), use_ros),
                n_trials=int(args.inner_trials),
                show_progress_bar=False,
            )
            best_params = study.best_params if hasattr(study, "best_params") else {}
            # —— 外层也兜底：auto -> sqrt（既支持原键名，也支持 clf__ 前缀）——
            if args.model == "random_forest":
                best_params = sanitize_rf_params(best_params)
                best_params = sanitize_rf_params({("clf__" + k): v for k, v in best_params.items()})  # 统一加前缀

            # —— 外层训练评估 —— #
            pipe = build_pipeline(args.model, class_weight=cw, n_jobs=int(args.n_jobs), use_ros=use_ros)

            # 如果上面已经加了 clf__ 前缀，就直接 apply；否则补前缀
            if any(k.startswith("clf__") for k in best_params.keys()):
                pipe = apply_params(pipe, best_params)
            else:
                pipe = apply_params(pipe, {"clf__" + k2: v2 for k2, v2 in best_params.items()})

            pipe.fit(X_tr, y_tr)
            prob_te = predict_proba_binary(pipe, X_te)

            # 主指标
            auc = roc_auc_score(y_te, prob_te) if np.unique(y_te).size == 2 else np.nan
            ap = average_precision_score(y_te, prob_te) if np.unique(y_te).size == 2 else np.nan
            brier = brier_score_loss(y_te, prob_te)

            # 阈值：Youden 基于外层训练集
            prob_tr = predict_proba_binary(pipe, X_tr)
            thr = youden_threshold(y_tr, prob_tr)

            pred05 = (prob_te >= 0.5).astype(int)
            predY = (prob_te >= thr).astype(int)

            row = {
                "fold": k,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "prevalence_test": float(np.mean(y_te)),
                "auc": float(auc) if not np.isnan(auc) else np.nan,
                "ap": float(ap) if not np.isnan(ap) else np.nan,
                "brier": float(brier),
                "thr_youden": float(thr),
                # at 0.5
                "acc@0.5": float(accuracy_score(y_te, pred05)),
                "f1@0.5": float(f1_score(y_te, pred05, zero_division=0)),
                "prec@0.5": float(precision_score(y_te, pred05, zero_division=0)),
                "recall@0.5": float(recall_score(y_te, pred05, zero_division=0)),
                # at youden
                "acc@Y": float(accuracy_score(y_te, predY)),
                "f1@Y": float(f1_score(y_te, predY, zero_division=0)),
                "prec@Y": float(precision_score(y_te, predY, zero_division=0)),
                "recall@Y": float(recall_score(y_te, predY, zero_division=0)),
            }
            folds_rows.append(row)

    # 写盘
    df_folds = pd.DataFrame(folds_rows)
    csv_path = out_dir / f"metrics_per_fold_{args.model}.csv"
    atomic_to_csv(df_folds, csv_path)

    summary = {
        "model": args.model,
        "outer_folds": int(args.outer_folds),
        "inner_trials": int(args.inner_trials),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "features": list(feat_names),
        "use_ros": bool(use_ros),
        "class_weight": cw,
        "metrics_mean": {
            "auc": float(np.nanmean(df_folds["auc"])) if "auc" in df_folds else np.nan,
            "ap": float(np.nanmean(df_folds["ap"])) if "ap" in df_folds else np.nan,
            "brier": float(np.nanmean(df_folds["brier"])) if "brier" in df_folds else np.nan,
            "acc@0.5": float(np.nanmean(df_folds["acc@0.5"])) if "acc@0.5" in df_folds else np.nan,
            "f1@0.5": float(np.nanmean(df_folds["f1@0.5"])) if "f1@0.5" in df_folds else np.nan,
            "prec@0.5": float(np.nanmean(df_folds["prec@0.5"])) if "prec@0.5" in df_folds else np.nan,
            "recall@0.5": float(np.nanmean(df_folds["recall@0.5"])) if "recall@0.5" in df_folds else np.nan,
            "acc@Y": float(np.nanmean(df_folds["acc@Y"])) if "acc@Y" in df_folds else np.nan,
            "f1@Y": float(np.nanmean(df_folds["f1@Y"])) if "f1@Y" in df_folds else np.nan,
            "prec@Y": float(np.nanmean(df_folds["prec@Y"])) if "prec@Y" in df_folds else np.nan,
            "recall@Y": float(np.nanmean(df_folds["recall@Y"])) if "recall@Y" in df_folds else np.nan,
        },
        "metrics_std": {
            "auc": float(np.nanstd(df_folds["auc"])) if "auc" in df_folds else np.nan,
            "ap": float(np.nanstd(df_folds["ap"])) if "ap" in df_folds else np.nan,
            "brier": float(np.nanstd(df_folds["brier"])) if "brier" in df_folds else np.nan,
            "acc@0.5": float(np.nanstd(df_folds["acc@0.5"])) if "acc@0.5" in df_folds else np.nan,
            "f1@0.5": float(np.nanstd(df_folds["f1@0.5"])) if "f1@0.5" in df_folds else np.nan,
            "prec@0.5": float(np.nanstd(df_folds["prec@0.5"])) if "prec@0.5" in df_folds else np.nan,
            "recall@0.5": float(np.nanstd(df_folds["recall@0.5"])) if "recall@0.5" in df_folds else np.nan,
            "acc@Y": float(np.nanstd(df_folds["acc@Y"])) if "acc@Y" in df_folds else np.nan,
            "f1@Y": float(np.nanstd(df_folds["f1@Y"])) if "f1@Y" in df_folds else np.nan,
            "prec@Y": float(np.nanstd(df_folds["prec@Y"])) if "prec@Y" in df_folds else np.nan,
            "recall@Y": float(np.nanstd(df_folds["recall@Y"])) if "recall@Y" in df_folds else np.nan,
        },
    }
    (out_dir / f"summary_{args.model}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[ok] Nested CV 完成：")
    print(f"  - 折内 trials = {args.inner_trials}, 折外 K = {args.outer_folds}")
    print(f"  - 每折结果: {csv_path}")
    print(f"  - 汇总: {out_dir / f'summary_{args.model}.json'}")


if __name__ == "__main__":
    main()



# python -m src.nested_cv --config conf/config.yaml --model random_forest --outer_folds 5 --inner_trials 30