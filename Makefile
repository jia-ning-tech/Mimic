# ============================
# ICU Lymphoma ML – Makefile
# ============================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# -------- 基本参数（可在命令行覆盖）---------
PY       ?= python -u
CFG      ?= conf/config.yaml
MODEL    ?= random_forest
METHOD   ?= raw           # raw | isotonic | sigmoid
TRIALS   ?= 30            # optuna 试验数 / nested CV 内层试验数
KFOLDS   ?= 5             # 校准/嵌套CV的 K 折数（外层由 outer_folds 指定）
M        ?= 20            # 多重插补次数（生成 MI 工件）
THR      ?= 0.5           # 亚组评估/阈值扫描默认阈值
FAST     ?= 0             # 1/true 开启快速模式（中位数插补代替 MICE，仅少数脚本支持）
MAX_N    ?= 1000          # SHAP 采样上限
BATCH    ?= 256           # SHAP 批大小
HEARTBEAT?= 5.0           # 心跳秒数（多数脚本支持）
OUTER    ?= 5             # 嵌套CV外层折数
N_BOOT   ?= 1000          # evaluate 的 bootstrap CI 次数
PLOT_CI  ?= 0             # evaluate 图中是否绘制 CI 阴影带（1=是）

# 阈值扫描附加参数
SENS_TARGETS ?=           # 例：0.8,0.9
SPEC_TARGETS ?=           # 例：0.8,0.9
THR_MIN      ?= 0.01
THR_MAX      ?= 0.99
THR_STEP     ?= 0.01

# 15个模型清单（和 config.yaml 对齐）
MODELS15 := logistic ridge lda svm knn gaussian_nb decision_tree random_forest extra_trees gbdt xgboost lightgbm catboost adaboost mlp

# FAST 标志
FAST_FLAG = $(if $(filter $(FAST),1 true yes),--fast,)

# PLOT_CI 标志
PLOT_CI_FLAG = $(if $(filter $(PLOT_CI),1 true yes),--plot_ci,)

# -------- 目标列表 --------
.PHONY: help setup kernel overview ingest preprocess mi feature-select optuna train mi-train evaluate shap calibrate dca threshold-scan subgroup nested report freeze conda-lock clean compare-all tree readme-tree

help:
	@echo "Usage: make <target> [VAR=...]"
	@echo
	@echo "核心流程："
	@echo "  setup             - conda 按 env/conda.yml 创建环境"
	@echo "  kernel            - 将当前环境注册为 Jupyter 内核"
	@echo "  overview          - 执行 notebooks/01_data_overview.ipynb"
	@echo "  ingest            - 数据摄取/标准化（src.ingest）"
	@echo "  preprocess        - 预处理（src.preprocess）"
	@echo "  mi                - 生成多重插补工件 m=M（src.multiple_imputation）"
	@echo "  feature-select    - LASSO 特征选择（src.feature_select）"
	@echo "  optuna            - 调参（src.optuna_search），可传 MODEL/TRIALS"
	@echo "  train             - 训练并评估（src.train），可传 MODEL/METHOD"
	@echo "  calibrate         - 概率校准（src.calibrate），K折=KFOLDS"
	@echo "  evaluate          - 汇总评估与图（src.evaluate），支持 bootstrap CI"
	@echo "  shap              - 计算 SHAP（src.shap_run），可传 MODEL/MAX_N/BATCH"
	@echo "  dca               - 决策曲线分析（src.dca）"
	@echo "  threshold-scan    - 阈值扫描（src.threshold_scan）可传目标敏感度/特异度"
	@echo "  subgroup          - 亚组评估（src.subgroup_eval）"
	@echo "  nested            - 嵌套交叉验证（src.nested_cv）"
	@echo "  report            - 汇总报告导出（src.report_export）"
	@echo "  compare-all       - 逐一跑满 15 个模型（train→calibrate→evaluate）"
	@echo
	@echo "环境管理："
	@echo "  freeze            - 导出 pip 依赖冻结文件"
	@echo "  conda-lock        - 导出 conda 环境锁定文件"
	@echo
	@echo "文档/目录树："
	@echo "  tree              - 生成项目目录树到 project_tree.txt（DEPTH=3 可配）"
	@echo "  readme-tree       - 将目录树嵌入 README.md 的占位符"
	@echo
	@echo "变量示例："
	@echo "  make train MODEL=random_forest METHOD=raw"
	@echo "  make calibrate MODEL=random_forest KFOLDS=5"
	@echo "  make evaluate MODEL=random_forest METHOD=isotonic N_BOOT=1000 PLOT_CI=1"
	@echo "  make threshold-scan MODEL=random_forest METHOD=isotonic SENS_TARGETS=0.9 SPEC_TARGETS=0.8"
	@echo "  make nested MODEL=random_forest OUTER=5 TRIALS=30"

# —— 环境与内核 ——
setup:
	conda env create -f env/conda.yml || true

kernel:
	python -m ipykernel install --user --name icu-lymphoma-ml --display-name "icu-lymphoma-ml (py310)"

# —— 数据与概览 ——
overview:
	jupyter nbconvert --to notebook --execute notebooks/01_data_overview.ipynb

ingest:
	$(PY) -m src.ingest --config $(CFG)

preprocess:
	$(PY) -m src.preprocess --config $(CFG)

# —— 多重插补与特征选择 ——
mi:
	$(PY) -m src.multiple_imputation --config $(CFG) --m $(M)

feature-select:
	$(PY) -m src.feature_select --config $(CFG)

# —— 训练与调参 ——
train:
	# METHOD=raw|isotonic|sigmoid；train.py 内置对 METHOD 的校准/集成逻辑
	$(PY) -m src.train --config $(CFG) --model $(MODEL) $(if $(filter $(METHOD),raw),,--method $(METHOD)) --kfolds $(KFOLDS)

mi-train:
	# 逐 m 训练并聚合（新脚本），更贴近 Rubin 口径；推荐在论文复现时使用
	$(PY) -m src.train_mi --config $(CFG) --model $(MODEL) --method $(METHOD) --kfolds $(KFOLDS) --heartbeat $(HEARTBEAT)

optuna:
	$(PY) -m src.optuna_search --model $(MODEL) --config $(CFG) --trials $(TRIALS)

nested:
	$(PY) -m src.nested_cv --config $(CFG) --model $(MODEL) --outer_folds $(OUTER) --inner_trials $(TRIALS) --heartbeat $(HEARTBEAT)

# —— 解释性 ——
shap:
	$(PY) -m src.shap_run --model $(MODEL) --config $(CFG) --max_n $(MAX_N) --batch_size $(BATCH)

# —— 概率校准与 DCA/评估 ——
calibrate:
	$(PY) -m src.calibrate --config $(CFG) --model $(MODEL) --method $(METHOD) --kfolds $(KFOLDS)

evaluate:
	$(PY) -m src.evaluate --config $(CFG) --model $(MODEL) --method $(METHOD) --n_boot $(N_BOOT) $(PLOT_CI_FLAG)

dca:
	$(PY) -m src.dca --config $(CFG) --model $(MODEL) --method $(METHOD) --thr_min $(THR_MIN) --thr_max $(THR_MAX) --thr_step $(THR_STEP)

# —— 阈值扫描与亚组 ——
threshold-scan:
	$(PY) -m src.threshold_scan --config $(CFG) --model $(MODEL) --method $(METHOD) \
		$(if $(SENS_TARGETS),--sens_targets $(SENS_TARGETS),) \
		$(if $(SPEC_TARGETS),--spec_targets $(SPEC_TARGETS),)

subgroup:
	$(PY) -m src.subgroup_eval --config $(CFG) --model $(MODEL) --method $(METHOD)

# —— 报告与环境锁定 ——
report:
	$(PY) -m src.report_export --config $(CFG) --model $(MODEL) --method $(METHOD)

freeze:
	pip freeze > env/requirements_freeze.txt

conda-lock:
	conda env export > env/conda_lock.yml

# —— 批量对比：跑满 15 个模型（train→calibrate→evaluate） ——
compare-all:
	@set -euo pipefail; \
	for m in $(MODELS15); do \
	  echo "===== [$$m] train ====="; \
	  $(PY) -m src.train --config $(CFG) --model $$m $(if $(filter $(METHOD),raw),,--method $(METHOD)) --kfolds $(KFOLDS); \
	  echo "===== [$$m] calibrate ====="; \
	  $(PY) -m src.calibrate --config $(CFG) --model $$m --method $(METHOD) --kfolds $(KFOLDS); \
	  echo "===== [$$m] evaluate ====="; \
	  $(PY) -m src.evaluate --config $(CFG) --model $$m --method $(METHOD) --n_boot $(N_BOOT); \
	done
	@echo "[ok] compare-all 完成，汇总请查看 outputs/metrics_test*.csv / model_auc_test.csv"

# —— 清理（谨慎使用） ——
clean:
	rm -rf outputs/* \
	       data_processed/*.parquet data_processed/*.csv data_processed/*.json || true

# ========= 动态目录树生成 =========
.PHONY: tree readme-tree _ensure_tree _tree_cmd
DEPTH ?= 3

_tree_cmd = if command -v tree >/dev/null 2>&1; then \
               tree -L $(DEPTH) --dirsfirst; \
            else \
               echo "[info] 'tree' 未安装，回退到 'find'。可用 'sudo apt-get install tree' 安装。"; \
               find . -maxdepth $(DEPTH) -print | sed 's|^\./||'; \
            fi

tree:
	@echo "[gen] 扫描深度: $(DEPTH)"
	@$(call _tree_cmd) > project_tree.txt
	@echo "[ok] 已写入 project_tree.txt"

readme-tree: tree
	@echo "[patch] 将目录树嵌入 README.md 中的占位符 ..."
	@awk 'BEGIN {print "```text"} {print} END {print "```"}' project_tree.txt > .tree_block.md
	@if grep -q "<!-- BEGIN:PROJECT_TREE -->" README.md && grep -q "<!-- END:PROJECT_TREE -->" README.md; then \
		sed -n '1,/<!-- BEGIN:PROJECT_TREE -->/p' README.md > README.tmp; \
		echo '<!-- BEGIN:PROJECT_TREE -->' >> README.tmp; \
		cat .tree_block.md >> README.tmp; \
		echo '<!-- END:PROJECT_TREE -->' >> README.tmp; \
		sed -n '/<!-- END:PROJECT_TREE -->/,$$p' README.md | tail -n +2 >> README.tmp; \
		mv README.tmp README.md; \
		echo "[ok] README.md 已更新目录树"; \
	else \
		echo "[warn] 未检测到占位符，追加到 README.md 尾部"; \
		echo '\n## 📦 项目文件结构（自动生成）\n' >> README.md; \
		echo '<!-- BEGIN:PROJECT_TREE -->' >> README.md; \
		cat .tree_block.md >> README.md; \
		echo '<!-- END:PROJECT_TREE -->' >> README.md; \
	fi
	@rm -f .tree_block.md
