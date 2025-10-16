# src/report_export.py
# -*- coding: utf-8 -*-
"""
Export a one-stop Markdown report that aggregates:
- Metrics (AUC/AP/Acc/F1/Precision/Recall/Brier), both summary & per-model
- ROC / PR / Calibration figures
- Threshold scan results & recommendations (Youden / F-beta / Sens/Spec targets)
- DCA curves (NB & sNB)
- SHAP (top-k table + bar & beeswarm)
- Data/ingest summary (from artifacts & logs if present)
- Config snapshot

Outputs:
- outputs/summary_report.md
- outputs/tables/summary_metrics_<model>[_<method>].csv
- outputs/tables/summary_shap_top20_<model>[_<method>].csv
"""

from __future__ import annotations

import argparse
import json
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import yaml


# ---------- utils ----------
def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


# ---------- Heartbeat ----------
class Heartbeat:
    def __init__(self, prefix: str, interval: float = 5.0):
        self.prefix = prefix
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._t = None
        self._t0 = None

    def _runner(self):
        while not self._stop.wait(self.interval):
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


# ---------- core ----------
def safe_rel(path: Path) -> str:
    # For markdown display relative to project root
    return str(path.as_posix())


def section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


def kv(k: str, v) -> str:
    return f"- **{k}**: {v}"


def md_table(df: Optional[pd.DataFrame], max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    df_show = df.head(max_rows)
    return df_show.to_markdown(index=False) + ("\n" if len(df_show) <= max_rows else f"\n\n_... and {len(df)-max_rows} more rows_\n")


def summarize_metrics(outputs_dir: Path, model: str, method: str) -> Dict[str, Optional[pd.DataFrame | Dict]]:
    suffix = "" if method == "raw" else f"_{method}"
    files = {
        "metrics_model": outputs_dir / f"metrics_test_{model}{suffix}.csv",
        "metrics_all": outputs_dir / "metrics_test.csv",
        "auc_table": outputs_dir / "model_auc_test.csv",
        "thr_scan": outputs_dir / f"threshold_scan_{model}{suffix}.csv",
        "thr_summary": outputs_dir / f"threshold_scan_{model}{suffix}_summary.json",
    }
    out: Dict[str, Optional[pd.DataFrame | Dict]] = {}
    for k, v in files.items():
        if v.suffix == ".csv":
            out[k] = read_csv(v)
        else:
            out[k] = read_json(v)
    return out


def summarize_dca(outputs_dir: Path, model: str, method: str) -> Dict[str, Optional[Path]]:
    suffix = "" if method == "raw" else f"_{method}"
    return {
        "csv": outputs_dir / f"dca_{model}{suffix}.csv",
        "nb_png": outputs_dir / f"dca_nb_{model}{suffix}.png",
        "snb_png": outputs_dir / f"dca_snb_{model}{suffix}.png",
    }


def summarize_figures(outputs_dir: Path, model: str, method: str) -> Dict[str, Path]:
    suffix = "" if method == "raw" else f"_{method}"
    return {
        "roc": outputs_dir / f"roc_test_{model}{suffix}.png",
        "pr": outputs_dir / f"pr_test_{model}{suffix}.png",
        "calib": outputs_dir / f"calibration_test_{model}{suffix}.png",
    }


def summarize_shap(outputs_dir: Path, model: str, method: str) -> Dict[str, Optional[Path]]:
    suffix = "" if method == "raw" else f"_{method}"
    shap_dir = outputs_dir / "shap"
    return {
        "imp_csv": shap_dir / f"shap_importance_{model}{suffix}.csv",
        "bar_png": shap_dir / f"shap_importance_bar_{model}{suffix}.png",
        "swarm_png": shap_dir / f"shap_summary_{model}{suffix}.png",
    }


def summarize_ingest(artifacts_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "split_idx": artifacts_dir / "split_idx.joblib",
        "feature_schema": artifacts_dir / "feature_schema.json",
        "columns_overview": Path("data_processed") / "columns_overview.csv",
        "missing_overview": Path("data_processed") / "missingness_overview.csv",
    }


def load_first_row(df: Optional[pd.DataFrame], default=None):
    if df is None or df.empty:
        return default
    return df.iloc[0].to_dict()


def build_report(cfg: Dict, model: str, method: str, hb_sec: float = 5.0) -> str:
    outputs_dir = Path("outputs")
    figures_dir = Path(cfg.get("output", {}).get("figures", "outputs/figures"))
    tables_dir  = Path(cfg.get("output", {}).get("tables",  "outputs/tables"))
    ensure_dir(outputs_dir); ensure_dir(figures_dir); ensure_dir(tables_dir)

    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "outputs/artifacts"))
    ensure_dir(artifacts_dir)

    # 1) Metrics & thresholds
    with Heartbeat(prefix="[hb] collect-metrics", interval=hb_sec):
        met = summarize_metrics(outputs_dir, model, method)
        df_metrics_model = met["metrics_model"]  # type: ignore
        df_metrics_all   = met["metrics_all"]    # type: ignore
        df_auc_table     = met["auc_table"]      # type: ignore
        df_thr_scan      = met["thr_scan"]       # type: ignore
        js_thr_summary   = met["thr_summary"]    # type: ignore

    # Extract primary metrics row
    _ = load_first_row(df_metrics_model, default={})  # currently unused; kept for extension

    # 2) Figures
    figs = summarize_figures(outputs_dir, model, method)
    # 3) DCA
    dca = summarize_dca(outputs_dir, model, method)
    df_dca = read_csv(dca["csv"]) if dca["csv"] and file_exists(dca["csv"]) else None
    # 4) SHAP
    shap_files = summarize_shap(outputs_dir, model, method)
    df_shap_imp = read_csv(shap_files["imp_csv"]) if shap_files["imp_csv"] and file_exists(shap_files["imp_csv"]) else None
    if df_shap_imp is not None and "mean_abs_shap" in df_shap_imp.columns:
        # ensure columns: feature, mean_abs_shap
        if df_shap_imp.columns[0] != "feature":
            df_shap_imp = df_shap_imp.rename(columns={df_shap_imp.columns[0]: "feature"})
        df_shap_imp = df_shap_imp[["feature", "mean_abs_shap"]]
        df_shap_top = df_shap_imp.sort_values("mean_abs_shap", ascending=False).head(20).reset_index(drop=True)
        df_shap_top.to_csv(tables_dir / f"summary_shap_top20_{model}{'' if method=='raw' else '_'+method}.csv", index=False)
    else:
        df_shap_top = None

    # 5) Ingest & data overview
    ing = summarize_ingest(artifacts_dir)
    df_columns_overview = read_csv(ing["columns_overview"]) if ing["columns_overview"] else None
    df_missing_overview = read_csv(ing["missing_overview"]) if ing["missing_overview"] else None

    # 6) Persist metrics summary table（若有）
    if isinstance(df_metrics_model, pd.DataFrame) and not df_metrics_model.empty:
        df_metrics_model.to_csv(tables_dir / f"summary_metrics_{model}{'' if method=='raw' else '_'+method}.csv", index=False)

    # 7) Compose Markdown
    md = []
    md.append("# ICU Lymphoma ML – Summary Report\n")
    md.append(f"_Model_: **{model}**    |    _Method_: **{method}**\n")
    md.append(f"_Generated at_: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    # ---- Data overview
    md.append(section("Data overview"))
    if df_columns_overview is not None:
        md.append("**Columns overview (head):**\n")
        md.append(md_table(df_columns_overview, max_rows=15))
    else:
        md.append("- Columns overview: _not generated_\n")

    if df_missing_overview is not None:
        md.append("\n**Missingness (top 15):**\n")
        df_miss_top = df_missing_overview.sort_values("pct_missing", ascending=False).head(15)
        md.append(md_table(df_miss_top, max_rows=15))
    else:
        md.append("- Missingness overview: _not generated_\n")

    # ---- Test metrics
    md.append(section("Test metrics"))
    if isinstance(df_metrics_model, pd.DataFrame) and not df_metrics_model.empty:
        md.append(md_table(df_metrics_model))
    else:
        md.append("_metrics not found_\n")

    # ---- ROC / PR / Calibration
    md.append(section("ROC / PR / Calibration", level=2))
    roc_png = figs["roc"]; pr_png = figs["pr"]; calib_png = figs["calib"]
    if file_exists(roc_png):
        md.append(f"**ROC (test)**  \n![]({safe_rel(roc_png)})\n")
    else:
        md.append("- ROC: _not generated_\n")
    if file_exists(pr_png):
        md.append(f"**PR (test)**  \n![]({safe_rel(pr_png)})\n")
    else:
        md.append("- PR: _not generated_\n")
    if file_exists(calib_png):
        md.append(f"**Calibration (test)**  \n![]({safe_rel(calib_png)})\n")
    else:
        md.append("- Calibration: _not generated_\n")

    # ---- Threshold scan
    md.append(section("Threshold scan", level=2))
    if isinstance(df_thr_scan, pd.DataFrame) and not df_thr_scan.empty:
        md.append("**Scan (head):**\n")
        md.append(md_table(df_thr_scan, max_rows=20))
    else:
        md.append("_scan not found_\n")

    if isinstance(js_thr_summary, dict) and js_thr_summary:
        md.append("\n**Recommendations**\n")
        try:
            youden = js_thr_summary.get("recommendations", {}).get("youden", {})
            fbeta  = js_thr_summary.get("recommendations", {}).get("fbeta", {})
            sens0  = js_thr_summary.get("recommendations", {}).get("sens_targets", {})
            spec0  = js_thr_summary.get("recommendations", {}).get("spec_targets", {})
            if youden:
                md.append(kv("Youden", f"thr={youden.get('thr', float('nan')):.3f}, sens={youden.get('sensitivity', 0):.3f}, spec={youden.get('specificity', 0):.3f}"))
            if fbeta:
                beta_val = fbeta.get("beta", 1.0)
                md.append(kv(f"F{beta_val}-max", f"thr={fbeta.get('thr', float('nan')):.3f}, fbeta={fbeta.get('fbeta', 0):.3f}, prec={fbeta.get('precision', 0):.3f}, recall={fbeta.get('sensitivity', 0):.3f}"))
            if sens0:
                k = sorted(list(sens0.keys()), key=lambda x: float(x))[0]
                s = sens0[k]
                md.append(kv(f"sens≥{k}", f"thr={s.get('thr', float('nan')):.3f}, sens={s.get('sensitivity', 0):.3f}, spec={s.get('specificity', 0):.3f}"))
            if spec0:
                k = sorted(list(spec0.keys()), key=lambda x: float(x))[0]
                s = spec0[k]
                md.append(kv(f"spec≥{k}", f"thr={s.get('thr', float('nan')):.3f}, sens={s.get('sensitivity', 0):.3f}, spec={s.get('specificity', 0):.3f}"))
            md.append("")
        except Exception:
            md.append("- (recommendations parse failed)\n")
    else:
        md.append("- Recommendations: _not generated_\n")

    # ---- DCA
    md.append(section("Decision Curve Analysis (DCA)", level=2))
    if df_dca is not None and not df_dca.empty:
        prev = df_dca["prevalence"].iloc[0] if "prevalence" in df_dca.columns else None
        if prev is not None:
            md.append(kv("Prevalence", f"{prev:.4f}"))
        md.append("\n")
    nb_png  = dca["nb_png"]; snb_png = dca["snb_png"]
    if nb_png and file_exists(nb_png):
        md.append(f"**Net Benefit**  \n![]({safe_rel(nb_png)})\n")
    else:
        md.append("- Net Benefit: _not generated_\n")
    if snb_png and file_exists(snb_png):
        md.append(f"**Standardized Net Benefit**  \n![]({safe_rel(snb_png)})\n")
    else:
        md.append("- sNB: _not generated_\n")

    # ---- SHAP
    md.append(section("Explainability (SHAP)", level=2))
    if df_shap_top is not None and not df_shap_top.empty:
        md.append("**Top-20 by mean(|SHAP|)**\n")
        md.append(md_table(df_shap_top, max_rows=20))
    else:
        md.append("- SHAP importance: _not generated_\n")
    if shap_files["bar_png"] and file_exists(shap_files["bar_png"]):
        md.append(f"\n**Importance (bar)**  \n![]({safe_rel(shap_files['bar_png'])})\n")
    else:
        md.append("- SHAP bar: _not generated_\n")
    if shap_files["swarm_png"] and file_exists(shap_files["swarm_png"]):
        md.append(f"\n**Summary (beeswarm)**  \n![]({safe_rel(shap_files['swarm_png'])})\n")
    else:
        md.append("- SHAP beeswarm: _not generated_\n")

    # ---- AUC table
    md.append(section("Per-m AUC (before/after)", level=2))
    if isinstance(df_auc_table, pd.DataFrame) and not df_auc_table.empty:
        this_model_name = f"{model}{'' if method=='raw' else '_'+method}"
        sub = df_auc_table[df_auc_table["model"] == this_model_name]
        md.append(md_table(sub if not sub.empty else df_auc_table, max_rows=30))
    else:
        md.append("_AUC table not found_\n")

    # ---- Config snapshot
    md.append(section("Config snapshot", level=2))
    cfg_str = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    md.append("```yaml\n" + cfg_str + "```\n")

    return "\n".join(md)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export Markdown summary report.")
    parser.add_argument("--config", "-c", type=str, default="conf/config.yaml")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--method", type=str, default="raw", choices=["raw", "isotonic", "sigmoid"])
    parser.add_argument("--heartbeat", type=float, default=5.0)
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    outputs_dir = Path("outputs"); ensure_dir(outputs_dir)
    tables_dir  = Path(cfg.get("output", {}).get("tables",  "outputs/tables")); ensure_dir(tables_dir)

    with Heartbeat(prefix="[hb] build-report", interval=float(args.heartbeat)):
        md = build_report(cfg, args.model, args.method, hb_sec=float(args.heartbeat))

    report_path = outputs_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    # 构建要打印的两个可选表路径，避免嵌套 f-string 语法问题
    suffix = "" if args.method == "raw" else f"_{args.method}"
    metrics_csv_path = tables_dir / f"summary_metrics_{args.model}{suffix}.csv"
    shap_top20_csv_path = tables_dir / f"summary_shap_top20_{args.model}{suffix}.csv"

    print("[ok] 报告导出完成：")
    print(f"  - Markdown: {report_path}")
    print(f"  - (若存在) 指标汇总: {metrics_csv_path}")
    print(f"  - (若存在) SHAP Top-20: {shap_top20_csv_path}")


if __name__ == "__main__":
    main()



# python -m src.report_export --config conf/config.yaml --model random_forest
# # 或
# python -m src.report_export --config conf/config.yaml --model random_forest --method isotonic
