# src/models.py
# -*- coding: utf-8 -*-
"""
Model registry & factory for ICU lymphoma ML project.

- Unified model factory: get_model(name, *, n_jobs, class_weight, random_state, params)
- CPU-friendly defaults; probability outputs enabled when meaningful
- Graceful fallback when optional libs (xgboost/lightgbm/catboost) are missing
- Filters unsupported kwargs per estimator to avoid TypeError
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import warnings

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier

# 可选依赖
try:
    import xgboost as xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    import lightgbm as lgb  # type: ignore
    _HAVE_LGB = True
except Exception:
    _HAVE_LGB = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    _HAVE_CAT = True
except Exception:
    _HAVE_CAT = False

# ====== 基础能力 ======
def get_model() -> List[str]:
    names = [
        "logistic",
        "ridge",
        "lda",
        "svm",
        "knn",
        "gaussian_nb",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gbdt",
        "xgboost",
        "lightgbm",
        "catboost",
        "adaboost",
        "mlp",
    ]
    return names

def is_tree_ensemble(name: str) -> bool:
    return name in {"random_forest", "extra_trees", "gbdt", "xgboost", "lightgbm", "catboost", "adaboost"}

def default_eval_metric(name: str) -> str:
    # 仅供 optuna_search 参考
    if name in {"xgboost", "lightgbm"}:
        return "auc"
    return "roc_auc"

# ====== 模型工厂 ======
def make_model(
    name: str,
    random_state: int = 42,
    class_weight: Optional[Dict[int, float] | str] = None,
    n_jobs: Optional[int] = None,
) -> Any:
    name = name.lower()

    # 通用 kw
    tree_common = {}
    if n_jobs is not None:
        # sklearn 树系支持 n_jobs
        tree_common["n_jobs"] = n_jobs

    # —— 线性/核类 ——
    if name == "logistic":
        # 使用 saga 支持 L1/L2；默认用 class_weight
        return LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=5000,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs if n_jobs is not None else None,
        )
    if name == "ridge":
        return RidgeClassifier(
            random_state=random_state
        )
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        # 概率用于曲线/阈值
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            n_jobs=n_jobs if n_jobs is not None else None,
        )
    if name == "gaussian_nb":
        return GaussianNB()

    # —— 树系 ——
    if name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=random_state,
            class_weight=class_weight,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            class_weight=class_weight,
            random_state=random_state,
            **tree_common,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            bootstrap=False,
            class_weight=class_weight,
            random_state=random_state,
            **tree_common,
        )
    if name == "gbdt":
        # sklearn GBDT 不支持 class_weight；用 sample_weight 外部适配
        return GradientBoostingClassifier(
            random_state=random_state,
        )
    if name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
        )

    # —— Boosting 家族（可选依赖）——
    if name == "xgboost":
        if not _HAVE_XGB:
            warnings.warn("[models] 未安装 xgboost，回退到 GradientBoostingClassifier。")
            return GradientBoostingClassifier(random_state=random_state)
        # 使用 xgb.sklearn API；注意：class_weight 需外部加权 sample_weight
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs if n_jobs is not None else 0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            verbosity=0,
        )

    if name == "lightgbm":
        if not _HAVE_LGB:
            warnings.warn("[models] 未安装 lightgbm，回退到 ExtraTreesClassifier。")
            return ExtraTreesClassifier(
                n_estimators=400,
                random_state=random_state,
                **tree_common,
            )
        # class_weight 可直接传 dict 或 "balanced"
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs if n_jobs is not None else -1,
            class_weight=class_weight,
            objective="binary",
            metric="auc",
        )

    if name == "catboost":
        if not _HAVE_CAT:
            warnings.warn("[models] 未安装 catboost，回退到 RandomForestClassifier。")
            return RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight=class_weight,
                **tree_common,
            )
        # CatBoost 原生支持 class_weights
        return CatBoostClassifier(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="AUC",
            loss_function="Logloss",
            thread_count=n_jobs if n_jobs is not None else -1,
            verbose=False,
            class_weights=class_weight if isinstance(class_weight, dict) else None,
        )

    # —— MLP ——
    if name == "mlp":
        # MLP 不支持 class_weight；外部 sample_weight 处理
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=random_state,
        )

    raise ValueError(f"未知模型名称: {name}. 支持: {get_supported_models()}")

# ====== Optuna 参数建议 ======
def suggest_params(trial, name: str) -> Dict[str, Any]:
    """Return a parameter dict for `apply_params`."""
    name = name.lower()
    p: Dict[str, Any] = {}

    if name == "logistic":
        p["C"] = trial.suggest_float("C", 1e-3, 100.0, log=True)
        p["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        return p

    if name == "ridge":
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        return p

    if name == "lda":
        p["solver"] = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
        if p["solver"] != "svd":
            p["shrinkage"] = trial.suggest_float("shrinkage", 0.0, 1.0)
        return p

    if name == "svm":
        p["C"] = trial.suggest_float("C", 1e-2, 100.0, log=True)
        p["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        return p

    if name == "knn":
        p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 51, step=2)
        p["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        p["p"] = trial.suggest_int("p", 1, 2)
        return p

    if name == "decision_tree":
        p["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        return p

    if name == "random_forest":
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=50)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
        p["max_features"] = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])
        p["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        return p

    if name == "extra_trees":
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=50)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
        p["max_features"] = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])
        return p

    if name == "gbdt":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 6)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
        return p

    if name == "adaboost":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
        return p

    if name == "mlp":
        p["hidden_layer_sizes"] = trial.suggest_categorical(
            "hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)]
        )
        p["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        p["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        return p

    if name == "xgboost":
        if not _HAVE_XGB:
            return {}
        p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1200, step=100)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 8)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return p

    if name == "lightgbm":
        if not _HAVE_LGB:
            return {}
        p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1500, step=100)
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 63, step=2)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return p

    if name == "catboost":
        if not _HAVE_CAT:
            return {}
        p["iterations"] = trial.suggest_int("iterations", 400, 1500, step=100)
        p["depth"] = trial.suggest_int("depth", 4, 10)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
        return p

    raise ValueError(f"未知模型名称: {name}")

# ====== 将参数应用到模型 ======
def apply_params(est, params: Dict[str, Any]):
    """A thin wrapper to set_params safely."""
    if not params:
        return est
    try:
        est.set_params(**params)
    except ValueError:
        # 某些参数名在不同版本不兼容，尽量降级处理
        safe = {}
        for k, v in params.items():
            if k in est.get_params():
                safe[k] = v
        if safe:
            est.set_params(**safe)
    return est

