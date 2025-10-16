# -*- coding: utf-8 -*-
"""
LASSO feature selection with:
- Candidate restriction (default: continuous only)
- High-correlation pruning (|r| >= corr_threshold, keep one)
- Wide C grid (logspace 1e-4..1e2, 60 points)
- 1-SE rule: among Cs with mean AUC >= (best_mean - se_best), pick the sparsest
- Stability across multiple imputations (frequency across m)
- Optional target_k enforcement (default k=8)

Usage:
    python -m src.feature_select --config conf/config.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# plotting (optional)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# I/O helpers
# -----------------------
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


def read_feature_schema(artifacts_dir: Path) -> Dict:
    schema_path = artifacts_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到特征架构：{schema_path}；请先运行 `python -m src.preprocess`。")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# Correlation helpers
# -----------------------
def point_biserial_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation between continuous x and binary y (0/1)."""
    try:
        r, _ = pearsonr(x, y)
        return float(abs(r))
    except Exception:
        return 0.0


def prune_high_correlation(df_train: pd.DataFrame,
                           y_train: np.ndarray,
                           candidates: List[str],
                           corr_threshold: float,
                           priority: List[str] | None = None) -> List[str]:
    """
    Prune features with |r| >= corr_threshold using:
    1) keep the one with larger |corr(x, y)|
    2) tie-breaker by presence in priority list (paper-preferred set)
    3) tie-breaker by alphabetical order (stable)
    """
    if len(candidates) <= 1:
        return candidates[:]

    sub = df_train[candidates].copy()
    corr = sub.corr().values
    cols = list(sub.columns)

    # Precompute |corr(x, y)|
    y = y_train.astype(float)
    xy_abs = {c: point_biserial_abs_corr(sub[c].values, y) for c in cols}

    keep = set(cols)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if cols[i] not in keep or cols[j] not in keep:
                continue
            if np.isnan(corr[i, j]):
                continue
            if abs(corr[i, j]) >= corr_threshold:
                a, b = cols[i], cols[j]
                ay, by = xy_abs.get(a, 0.0), xy_abs.get(b, 0.0)
                if abs(ay - by) > 1e-12:
                    loser = b if ay > by else a
                else:
                    if priority:
                        a_prio = a in priority
                        b_prio = b in priority
                        if a_prio and not b_prio:
                            loser = b
                        elif b_prio and not a_prio:
                            loser = a
                        else:
                            loser = max(a, b)
                    else:
                        loser = max(a, b)
                keep.discard(loser)

    pruned = [c for c in cols if c in keep]
    return pruned


# -----------------------
# L1 logistic with CV (wide C grid)
# -----------------------
def fit_l1_logit_cv_with_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_jobs: int,
    C_grid: np.ndarray,
    class_weight: Dict[int, float] | None,
    seed: int,
) -> LogisticRegressionCV:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        cv=cv,
        Cs=C_grid,
        max_iter=4000,
        n_jobs=n_jobs,
        class_weight=class_weight,
        refit=True,
        random_state=seed,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)
    return model


def _get_scores_array(model: LogisticRegressionCV) -> np.ndarray:
    """
    Return scores array shape (n_folds, n_Cs) robustly for binary case.
    """
    # keys of scores_/coefs_paths_ are class labels; pick the positive class (index 1)
    cls_pos = model.classes_[1]
    scores = model.scores_[cls_pos]  # (n_folds, n_Cs)
    return np.asarray(scores)


def _get_coefs_paths_array(model: LogisticRegressionCV) -> np.ndarray:
    """
    Return coef paths array shape (n_folds, n_Cs, n_features) robustly for binary case.
    """
    cls_pos = model.classes_[1]
    paths = model.coefs_paths_[cls_pos]  # (n_folds, n_Cs, n_features)
    return np.asarray(paths)


def _get_Cs_vector(model: LogisticRegressionCV) -> np.ndarray:
    """
    Return 1D Cs vector of length n_Cs robustly across sklearn versions.
    """
    Cs_attr = np.asarray(model.Cs_)
    if Cs_attr.ndim == 1:
        return Cs_attr
    # Multi-class shape (n_classes, n_Cs) — take positive class row
    cls_pos_idx = 1  # index within classes_, safe because binary => [neg, pos]
    return np.asarray(Cs_attr[cls_pos_idx])


def pick_C_by_1se_rule(model: LogisticRegressionCV) -> Tuple[float, int]:
    """
    Use model.scores_ and model.coefs_paths_ to implement 1-SE rule.
    Returns:
        chosen_C, chosen_index
    """
    scores = _get_scores_array(model)          # (n_folds, n_Cs)
    means = scores.mean(axis=0)                # (n_Cs,)
    stds = scores.std(axis=0, ddof=1)
    n_folds = scores.shape[0]
    ses = stds / np.sqrt(max(1, n_folds))

    best_idx = int(np.argmax(means))
    best_mean = float(means[best_idx])
    best_se = float(ses[best_idx])

    mask_1se = means >= (best_mean - best_se + 1e-12)
    cand_idx = np.where(mask_1se)[0]
    if len(cand_idx) == 0:
        Cs_vec = _get_Cs_vector(model)
        return float(Cs_vec[best_idx]), int(best_idx)

    coefs_paths = _get_coefs_paths_array(model)  # (n_folds, n_Cs, n_features)
    nnz_median = []
    for j in cand_idx:
        nnz_per_fold = (np.abs(coefs_paths[:, j, :]) > 1e-8).sum(axis=1)
        nnz_median.append(np.median(nnz_per_fold))
    nnz_median = np.array(nnz_median)

    # Sort by nnz asc, then mean desc
    order = np.lexsort((-means[cand_idx], nnz_median))
    chosen_idx = int(cand_idx[order[0]])

    Cs_vec = _get_Cs_vector(model)
    return float(Cs_vec[chosen_idx]), int(chosen_idx)


# -----------------------
# Optional: correlation heatmap
# -----------------------
def plot_correlation_heatmap(df_train: pd.DataFrame, cols: List[str], out_path: Path) -> None:
    try:
        if len(cols) < 2:
            return
        corr = df_train[cols].corr().values
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(cols)), max(5, 0.5 * len(cols))))
        im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] 相关矩阵绘制失败：{e}")


# -----------------------
# Paths
# -----------------------
@dataclass
class Paths:
    artifacts_dir: Path
    mi_dir: Path
    out_dir: Path
    selected_json: Path
    selected_csv: Path
    corr_png: Path


def build_paths(cfg: Dict) -> Paths:
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    mi_dir = Path(cfg["missing_data"]["mice"]["mice_output_dir"])
    out_dir = Path("outputs/feature_selection")
    ensure_dir(artifacts_dir)
    ensure_dir(mi_dir)
    ensure_dir(out_dir)
    selected_json = artifacts_dir / "selected_features.json"
    selected_csv = out_dir / "selected_features.csv"
    corr_png = out_dir / "corr_selected.png"
    return Paths(artifacts_dir, mi_dir, out_dir, selected_json, selected_csv, corr_png)


# -----------------------
# Main
# -----------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="LASSO selection with 1-SE rule & correlation pruning.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    paths = build_paths(cfg)

    # ---- configs
    seed = int(cfg["project"].get("seed", 42))
    n_jobs = int(cfg["project"].get("n_jobs", -1))

    lasso_cfg = cfg["selection"]["lasso"]
    cv_folds = int(lasso_cfg.get("cv", 5))
    lock_across_m = bool(lasso_cfg.get("lock_features_across_m", True))
    freq_threshold = float(lasso_cfg.get("freq_threshold", 0.5))
    restrict_to = lasso_cfg.get("restrict_to", "continuous").lower()  # "continuous" | "all"
    target_k = int(lasso_cfg.get("target_k", 8))
    corr_threshold = float(lasso_cfg.get("corr_threshold", 0.90))

    # class_weight
    class_weight = None
    if cfg["imbalance"].get("use_class_weight", True):
        pos_w = float(cfg["imbalance"].get("positive_class_weight", 1.0))
        class_weight = {0: 1.0, 1: pos_w}

    # schema + MI index
    schema = read_feature_schema(paths.artifacts_dir)
    mi_index = read_mi_index(paths.mi_dir)
    m_paths = [Path(p["path"]) for p in mi_index["paths"]]
    all_features: List[str] = mi_index["feature_order"]

    # Candidate space
    if restrict_to == "continuous":
        base_candidates = [c for c in schema.get("continuous", []) if c in all_features]
        space_info = "continuous"
    else:
        base_candidates = [c for c in all_features]
        space_info = "all"

    if len(base_candidates) < 2:
        raise RuntimeError(f"候选特征数({len(base_candidates)})太少；请检查 schema 或将 restrict_to=all。")

    paper_priority = ["bun", "platelets", "pt", "heart_rate", "sbp", "aptt", "spo2", "bicarbonate"]
    C_grid = np.logspace(-4, 2, 60)

    print(f"[info] LASSO 选择：M={len(m_paths)}, cv={cv_folds}, restrict_to={space_info}, "
          f"corr_threshold={corr_threshold}, 1-SE启用, target_k={target_k}")

    # Frequency accumulation
    freq_counter = {f: 0 for f in base_candidates}
    coef_abs_sum = {f: 0.0 for f in base_candidates}
    first_train_df = None

    for i, p in enumerate(m_paths, start=1):
        art = joblib.load(p)
        df_tr_full = pd.DataFrame(art["X_train"], columns=art["feature_order"])
        y_tr = np.asarray(art["y_train"]).astype(int)

        if i == 1:
            first_train_df = df_tr_full.copy()

        # correlation pruning on this m
        pruned = prune_high_correlation(
            df_train=df_tr_full,
            y_train=y_tr,
            candidates=base_candidates,
            corr_threshold=corr_threshold,
            priority=paper_priority
        )
        if len(pruned) < 2:
            pruned = base_candidates[:]
        X_tr = df_tr_full[pruned].copy()

        # remove zero-variance columns
        var = X_tr.var(axis=0)
        keep_cols = var[var > 0].index.tolist()
        if len(keep_cols) < 2:
            raise RuntimeError("训练数据有效特征少于2列，无法进行 LASSO。")
        X_use = X_tr[keep_cols].values

        # Fit and score
        model = fit_l1_logit_cv_with_scores(
            X=X_use, y=y_tr, n_splits=cv_folds, n_jobs=n_jobs,
            C_grid=C_grid, class_weight=class_weight, seed=seed
        )

        # 1-SE pick
        chosen_C, chosen_idx = pick_C_by_1se_rule(model)

        # Refit at chosen C for exact sparsity
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=chosen_C, max_iter=4000,
            class_weight=class_weight, fit_intercept=True, random_state=seed
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_use, y_tr)

        coef = clf.coef_.ravel()
        selected_in_this_m = [keep_cols[j] for j, nz in enumerate(np.abs(coef) > 1e-8) if nz]

        for f in selected_in_this_m:
            freq_counter[f] = freq_counter.get(f, 0) + 1
            coef_abs_sum[f] = coef_abs_sum.get(f, 0.0) + float(abs(coef[keep_cols.index(f)]))

        print(f"[m={i}] 相关裁剪后候选={len(pruned)}，1-SE 选 C={chosen_C:.4g}，入选={len(selected_in_this_m)}")

    # Aggregate across M
    M = len(m_paths)
    rows = []
    for f in base_candidates:
        cnt = int(freq_counter.get(f, 0))
        freq = float(cnt / M)
        mean_abs_coef = float(coef_abs_sum.get(f, 0.0) / cnt) if cnt > 0 else 0.0
        rows.append({"feature": f, "count": cnt, "freq": freq, "mean_abs_coef": mean_abs_coef})

    freq_df = pd.DataFrame(rows).sort_values(["freq", "mean_abs_coef"], ascending=[False, False])

    # Primary lock by threshold
    if lock_across_m:
        selected = [r["feature"] for r in freq_df.to_dict(orient="records") if r["freq"] >= freq_threshold]
    else:
        selected = []

    # Enforce target_k if configured
    ranked = freq_df["feature"].tolist()
    if target_k and target_k > 0:
        if len(selected) > target_k:
            selected_final = freq_df.head(target_k)["feature"].tolist()
        elif len(selected) < target_k:
            remain = [f for f in ranked if f not in selected]
            selected_final = selected + remain[: max(0, target_k - len(selected))]
        else:
            selected_final = selected
    else:
        selected_final = selected if selected else ranked

    # Outputs
    out_dir = paths.out_dir
    ensure_dir(out_dir)
    freq_df.to_csv(paths.selected_csv, index=False)
    print(f"[ok] 频次表已导出: {paths.selected_csv}")

    selected_obj = {
        "selected_features": selected_final,
        "target_k": target_k,
        "restrict_to": restrict_to,
        "freq_threshold": freq_threshold if lock_across_m else None,
        "locked_across_m": bool(lock_across_m),
        "M": M,
        "cv_folds": cv_folds,
        "class_weight": {k: float(v) for k, v in ({} if class_weight is None else class_weight).items()},
        "candidate_space_size": len(base_candidates),
        "corr_threshold": corr_threshold,
        "all_features_ranked": freq_df.to_dict(orient="records"),
    }
    with open(paths.selected_json, "w", encoding="utf-8") as f:
        json.dump(selected_obj, f, ensure_ascii=False, indent=2)
    print(f"[ok] 已写入最终特征清单: {paths.selected_json}")
    print(f"[stat] 最终入选特征（{len(selected_final)}个）: {selected_final}")

    # Corr heatmap on m=1 (post-selection)
    if first_train_df is not None and len(selected_final) >= 2:
        plot_correlation_heatmap(first_train_df, selected_final, paths.corr_png)
        print(f"[ok] 已导出相关矩阵图: {paths.corr_png}")

    print("[done] LASSO 特征选择完成。")


if __name__ == "__main__":
    main()



# python -m src.feature_select --config conf/config.yaml
