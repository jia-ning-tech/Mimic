# src/data_utils.py
# -*- coding: utf-8 -*-
"""
Common utilities for ICU lymphoma ML project.

- Robust I/O (atomic CSV/JSON/fig writes)
- YAML/JSON/Joblib helpers
- Global seeding and reproducibility
- Fixed stratified split index save/load
- Timer & Heartbeat helpers
- Metrics helpers: safe AUC, threshold metrics, PR/ROC points
- Matplotlib style for publication-quality figures
"""

from __future__ import annotations

import os
import io
import json
import math
import time
import shutil
import random
import warnings
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import joblib

# ====== Optional: plotting defaults ======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ====== Warning controls (optional) ======
def silence_sklearn_warnings() -> None:
    try:
        from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass


# ====== Path helpers ======
def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(data: bytes, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)


def atomic_write_text(text: str, out_path: Path, encoding: str = "utf-8") -> None:
    atomic_write_bytes(text.encode(encoding), out_path)


def atomic_to_csv(df: pd.DataFrame, out_path: Path, **to_csv_kwargs) -> None:
    ensure_dir(out_path.parent)
    default = dict(index=False)
    default.update(to_csv_kwargs or {})
    with io.StringIO() as buf:
        df.to_csv(buf, **default)
        atomic_write_text(buf.getvalue(), out_path)


def atomic_savefig(fig: plt.Figure, out_path: Path, dpi: int = 200, bbox_inches: str = "tight") -> None:
    ensure_dir(out_path.parent)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=out_path.suffix, delete=False) as tmp:
        fig.savefig(tmp.name, dpi=dpi, bbox_inches=bbox_inches)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(out_path)


# ====== YAML/JSON/Joblib ======
def load_yaml(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path | str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=indent)
    atomic_write_text(s, Path(path))


def dump_joblib(obj: Any, path: Path | str, compress: int = 3) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp, compress=compress)
    Path(tmp).replace(path)


def load_joblib(path: Path | str) -> Any:
    return joblib.load(path)


# ====== Reproducibility ======
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass
    # sklearn uses numpy RNG; no extra setup needed


# ====== Timer & Heartbeat ======
@contextmanager
def Timer(msg: str = "", verbose: bool = True):
    t0 = time.time()
    try:
        yield
    finally:
        if verbose:
            print(f"{msg} 用时 {time.time() - t0:.2f}s", flush=True)


class Heartbeat:
    def __init__(self, prefix: str, interval: float = 5.0):
        self.prefix = prefix
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._t0: Optional[float] = None

    def _runner(self):
        while not self._stop.wait(self.interval):
            if self._t0 is None:
                continue
            elapsed = time.time() - self._t0
            print(f"{self.prefix} ... {elapsed:.1f}s elapsed", flush=True)

    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._runner, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.1)


# ====== Split index save/load ======
def save_split_index(train_idx: np.ndarray, test_idx: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    obj = {
        "train_idx": np.asarray(train_idx, dtype=int).tolist(),
        "test_idx": np.asarray(test_idx, dtype=int).tolist(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(obj, out_path)


def load_split_index(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = read_json(path)
    if not obj or "train_idx" not in obj or "test_idx" not in obj:
        raise ValueError(f"分割索引文件格式错误：{path}")
    tr = np.asarray(obj["train_idx"], dtype=int)
    te = np.asarray(obj["test_idx"], dtype=int)
    return tr, te


# ====== Metrics helpers ======
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC；若 y_true 单类则返回 NaN。"""
    try:
        from sklearn.metrics import roc_auc_score
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def pr_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return p, r


def roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve
    f, t, _ = roc_curve(y_true, y_prob)
    return f, t


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    pred = (p >= float(thr)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def _div(a, b): return a / b if b > 0 else 0.0

    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    npv  = _div(tn, tn + fn)
    acc  = _div(tp + tn, tp + fp + tn + fn)
    f1   = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    br   = brier_score(y, p)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv,
        "accuracy": acc, "f1": f1, "brier": br
    }


# ====== Matplotlib styles ======
def apply_mpl_style(small: bool = False) -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 10 if small else 11,
        "axes.titlesize": 11 if small else 12,
        "axes.labelsize": 10 if small else 11,
        "xtick.labelsize": 9 if small else 10,
        "ytick.labelsize": 9 if small else 10,
    })


# ====== Artifact helpers ======
def find_artifact(root: Path, pattern: str) -> Optional[Path]:
    """在 root 下按模式查找第一个匹配文件；找不到返回 None。"""
    root = Path(root)
    for p in root.rglob(pattern):
        return p
    return None


# ====== Pretty printing ======
def dict_head(d: Dict[str, Any], k: int = 5) -> str:
    keys = list(d.keys())[:k]
    return "{" + ", ".join(f"{k}:{repr(d[k])}" for k in keys) + ("..." if len(d) > k else "") + "}"


# ====== Simple cache key ======
def cache_key_from_array(X: np.ndarray) -> str:
    """基于形状与前几个数生成一个轻量 cache key（调试用）。"""
    h = hash((X.shape, float(np.nanmean(X[:5].ravel()))))
    return hex(h & 0xFFFFFFFF)


# ====== CSV append with unique key ======
def append_or_replace_row(csv_path: Path, row: Dict[str, Any], key_cols: List[str]) -> None:
    """
    在 CSV 中按 key_cols 去重后追加/替换一行。
    """
    ensure_dir(csv_path.parent)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        mask = np.ones(len(df), dtype=bool)
        for k in key_cols:
            mask &= (df[k].astype(str).values == str(row[k]))
        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], axis=0, ignore_index=True)
    else:
        df = pd.DataFrame([row])
    atomic_to_csv(df, csv_path)


# ====== End of file ======
