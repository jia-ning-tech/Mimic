# src/optuna_search.py
# -*- coding: utf-8 -*-
"""
Optuna-based hyperparameter search for ICU lymphoma ML (CPU-friendly).

- Uses MI artifacts (defaults to m=1) as training data
- Respects selected_features.json if present
- Stratified K-Fold CV with ROC AUC as the optimization target
- Handles class imbalance via class_weight or sample_weight
- Optional RandomOverSampler in CV folds (if imblearn installed)
- Saves best params to outputs/best_params_<model>.json

Usage:
    python -m src.optuna_search --config conf/config.yaml --model random_forest --trials 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models import make_model

# Optional: imblearn for oversampling
try:
    from imblearn.over_sampling import RandomOverSampler
    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False


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


# -----------------------
# Utilities
# -----------------------
def get_primary_metric(cfg: Dict) -> str:
    return str(cfg.get("models", {}).get("primary_metric", "roc_auc")).lower()


def class_weight_to_sample_weight(y: np.ndarray, cw: Dict[int, float]) -> np.ndarray:
    w0 = float(cw.get(0, 1.0))
    w1 = float(cw.get(1, 1.0))
    sw = np.where(y == 1, w1, w0).astype("float32")
    return sw


def get_class_weight(cfg: Dict) -> Optional[Dict[int, float]]:
    imb = cfg.get("imbalance", {}) or {}
    if imb.get("use_class_weight", True):
        return {0: 1.0, 1: float(imb.get("positive_class_weight", 1.0))}
    return None


def get_proba(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        # scale to [0,1] via logistic-ish transform for AUC comparability
        # but ROC AUC is invariant to monotonic transforms; we can return raw scores
        return s
    # fallback
    preds = est.predict(X)
    return preds.astype(float)


def maybe_resample(X: np.ndarray, y: np.ndarray, method: str, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    if method == "random_over_sampler":
        if _HAS_IMBLEARN:
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res
        else:
            print("[warn] imblearn 未安装，无法进行 RandomOverSampler，将跳过过采样。")
            return X, y
    return X, y


# -----------------------
# Search spaces (CPU-friendly)
# -----------------------
def suggest_params(trial: optuna.Trial, model_name: str) -> Dict:
    m = model_name.lower()
    if m in ("logistic", "logreg"):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e+2, log=True),
            "max_iter": 2000,
        }
    if m in ("ridge", "ridge_classifier"):
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e+2, log=True),
        }
    if m in ("svm", "svc"):
        return {
            "C": trial.suggest_float("C", 1e-2, 1e+2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf"]),  # rbf only (CPU)
        }
    if m in ("knn",):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 31, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if m in ("decision_tree", "dt"):
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if m in ("random_forest", "rf"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if m in ("extra_trees", "et"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False]),  # ET 常用 False
        }
    if m in ("gbdt", "gradient_boosting", "gradient_boosting_classifier"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if m in ("xgboost", "xgb", "xgb_classifier"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        }
    if m in ("lightgbm", "lgbm", "lgbm_classifier"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=4),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if m in ("catboost", "catboost_classifier", "cat"):
        return {
            "iterations": trial.suggest_int("iterations", 400, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if m in ("adaboost",):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        }
    if m in ("mlp", "mlp_classifier", "neural_net"):
        return {
            "hidden_layer_sizes": tuple([trial.suggest_int("h1", 32, 128, step=32),
                                         trial.suggest_int("h2", 16, 64, step=16)]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 300,
        }
    # default empty
    return {}


# -----------------------
# Objective
# -----------------------
class Objective:
    def __init__(self, cfg: Dict, X: np.ndarray, y: np.ndarray, model_name: str):
        self.cfg = cfg
        self.X = X
        self.y = y
        self.model_name = model_name.lower()
        opt_cfg = cfg.get("models", {}).get("optimization", {}) or {}
        self.cv_folds = int(opt_cfg.get("cv_folds", 5))
        self.random_state = int(cfg.get("project", {}).get("seed", 42))
        self.resampling_in_cv = str(cfg.get("imbalance", {}).get("resampling_in_cv", "none")).lower()
        self.primary = get_primary_metric(cfg)
        self.cw = get_class_weight(cfg)  # for sample_weight fallback

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(trial, self.model_name)
        est = make_model(self.model_name, self.cfg, custom_params=params)

        # If DummyClassifier returned (missing optional libs), abort trial
        if est.__class__.__name__ == "DummyClassifier":
            trial.set_user_attr("invalid_reason", "missing_optional_library")
            return 0.0

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        aucs: List[float] = []

        for tr_idx, va_idx in skf.split(self.X, self.y):
            X_tr, X_va = self.X[tr_idx], self.X[va_idx]
            y_tr, y_va = self.y[tr_idx], self.y[va_idx]

            # optional resampling ONLY on training fold
            X_fit, y_fit = maybe_resample(
                X_tr, y_tr, method=self.resampling_in_cv, random_state=self.random_state
            )

            fit_kwargs = {}
            # For models without class_weight support, use sample_weight
            if self.cw is not None:
                # known to ignore/unsupported: GradientBoosting (sklearn), (XGB handled via scale_pos_weight typically but we use sample_weight), AdaBoost partly.
                if hasattr(est, "class_weight") or "class_weight" in est.get_params().keys():
                    # already set via make_model
                    pass
                else:
                    fit_kwargs["sample_weight"] = class_weight_to_sample_weight(y_fit, self.cw)

            est.fit(X_fit, y_fit, **fit_kwargs)
            proba_va = get_proba(est, X_va)
            # roc_auc_score can accept scores (decision_function) directly
            auc = roc_auc_score(y_va, proba_va)
            aucs.append(float(auc))

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        trial.set_user_attr("cv_auc_mean", mean_auc)
        trial.set_user_attr("params_used", params)
        return mean_auc


# -----------------------
# Main
# -----------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml", help="Path to config.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name (e.g., random_forest).")
    parser.add_argument("--trials", type=int, default=None, help="Override trial numbers.")
    parser.add_argument("--mindex", type=int, default=1, help="Use which MI run as training (default: 1).")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))

    # Load MI artifacts (use mindex-th run)
    mi_dir = Path(cfg["missing_data"]["mice"]["mice_output_dir"])
    mi_index = read_mi_index(mi_dir)
    paths_list = mi_index.get("paths", [])
    if not paths_list:
        raise RuntimeError("MI 索引为空；请先运行 `python -m src.multiple_imputation`。")

    mindex = max(1, int(args.mindex))
    if mindex > len(paths_list):
        mindex = 1  # fallback to first

    art_path = Path(paths_list[mindex - 1]["path"])
    art = joblib.load(art_path)
    X_train_df = pd.DataFrame(art["X_train"], columns=art["feature_order"])
    y_train = np.asarray(art["y_train"]).astype(int)

    # Restrict to selected features if exists
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    selected = read_selected_features(artifacts_dir)
    if selected:
        # 保证顺序一致且存在
        selected = [c for c in selected if c in X_train_df.columns]
        if not selected:
            raise RuntimeError("selected_features.json 中的列在训练矩阵中不存在。")
        X_train_df = X_train_df[selected]

    X_train = X_train_df.values.astype("float32")

    # Optuna setup
    study_dir = Path(cfg["output"]["logs"])
    ensure_dir(study_dir)
    study_name = f"optuna_{args.model}"
    storage = None  # 使用内存存储；如需持久化可改 sqlite:///...

    direction = cfg.get("models", {}).get("optimization", {}).get("direction", "maximize")
    n_trials = int(args.trials if args.trials is not None else cfg.get("models", {}).get("optimization", {}).get("n_trials", 50))

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_trials // 5))
    sampler = optuna.samplers.TPESampler(seed=int(cfg["project"].get("seed", 42)))

    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, load_if_exists=False,
                                sampler=sampler, pruner=pruner)

    obj = Objective(cfg=cfg, X=X_train, y=y_train, model_name=args.model)
    print(f"[info] 开始调参：model={args.model}, trials={n_trials}, folds={obj.cv_folds}, "
          f"resampling_in_cv={cfg.get('imbalance',{}).get('resampling_in_cv','none')}")

    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    best_score = float(best.value)

    # Save best params
    out_json = Path("outputs") / f"best_params_{args.model}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "best_params": best_params,
                "best_cv_auc": best_score,
                "trials": n_trials,
                "cv_folds": obj.cv_folds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[ok] 最优参数已保存：{out_json}")
    print(f"[stat] best_cv_auc={best_score:.4f}")
    print(f"[stat] best_params={best_params}")


if __name__ == "__main__":
    main()
