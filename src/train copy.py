# src/train.py
from __future__ import annotations
import argparse, os, yaml
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from .preprocess import (clip_outliers, mice_fit_transform, mice_transform,
                         scale_fit_transform, scale_transform)
from .models import make_model

def load_cfg(p="conf/config.yaml"):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def auto_type_split(df: pd.DataFrame, outcome: str):
    y = df[outcome].astype(int)
    X = df.drop(columns=[outcome])
    # 简单规则：数值型为连续，非数值型当作类别
    cont = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat  = [c for c in X.columns if c not in cont]
    return X, y, cont, cat

def main(config_path: str):
    cfg = load_cfg(config_path)
    proc_dir = Path(cfg["data"].get("processed_dir","data_proc"))
    parquet_path = proc_dir / "s1_data.parquet"
    assert parquet_path.exists(), f"请先运行: python -m src.ingest 生成 {parquet_path}"

    df = pd.read_parquet(parquet_path)

    outcome = cfg["data"]["outcome_col"]
    if outcome not in df.columns:
        raise ValueError(f"结局列 {outcome} 未在数据中发现，请检查 conf/config.yaml 或先运行 src.ingest 自动猜测。")

    X, y, cont_cols, cat_cols = auto_type_split(df, outcome)

    # 这里只使用论文关键连续特征（你可在 config 中精准指定）
    wanted = set(cfg["features"]["continuous"] + cfg["features"].get("categorical",[]))
    keep = [c for c in X.columns if c in wanted]
    if not keep:
        print("[WARN] config.features.* 与数据列未对齐，暂以所有数值列为特征（仅用于跑通）。")
        keep = cont_cols
    X = X[keep].copy()

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], train_size=cfg["split"]["train_size"],
        random_state=cfg["split"]["random_state"], stratify=y
    )

    # 仅在训练集内做清洗/插补/标准化
    X_train = clip_outliers(X_train, *cfg["preprocess"]["outlier_clip_quantiles"], cols=keep)
    imp, X_train = mice_fit_transform(
        X_train, random_state=cfg["project"]["seed"], max_iter=cfg["missing_data"]["mice"]["max_iter"]
    )
    scaler, X_train = scale_fit_transform(X_train, scaler_kind=cfg["preprocess"]["scaler"])

    # 测试集 transform
    X_test  = clip_outliers(X_test, *cfg["preprocess"]["outlier_clip_quantiles"], cols=keep)
    X_test  = mice_transform(imp, X_test)
    X_test  = scale_transform(scaler, X_test)

    # 类不平衡：在训练集内过采样（或使用 class_weight）
    if cfg["imbalance"]["resampling_in_cv"] == "random_over_sampler":
        ros = RandomOverSampler(random_state=cfg["project"]["seed"])
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # 先跑一个基线模型（CatBoost/Logistic均可；此处以 CatBoost 为例）
    cls_w = cfg["imbalance"]["positive_class_weight"] if cfg["imbalance"]["use_class_weight"] else None
    # model = make_model("catboost", class_weight=cls_w, random_state=cfg["project"]["seed"])
    for name in ["logistic","random_forest","xgboost","lightgbm","catboost"]:
        try:
            m = make_model(name, class_weight=cls_w, random_state=cfg["project"]["seed"])
            m.fit(X_train, y_train)
            proba = m.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, proba)
            print(f"[TEST] {name}: AUC={auc:.4f}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    print(f"[TEST] ROC-AUC = {auc:.4f}")

    # 输出曲线
    reports = Path(cfg["project"]["results_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"CatBoost (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.savefig(reports / "roc_test.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 校准
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    bs = brier_score_loss(y_test, proba)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration (Brier={bs:.3f})")
    plt.savefig(reports / "calibration_test.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 保存结果表
    out_tbl = pd.DataFrame({"metric":["roc_auc","brier"], "value":[auc, bs]})
    out_tbl.to_csv(reports / "metrics_test.csv", index=False)
    print(f"[OK] Results saved to: {reports}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    args = parser.parse_args()
    main(args.config)
