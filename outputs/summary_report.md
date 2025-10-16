# ICU Lymphoma ML – Summary Report

_Model_: **random_forest**    |    _Method_: **raw**

_Generated at_: 2025-10-13 06:33:22 UTC


## Data overview

**Columns overview (head):**

| column      | dtype   |   n_unique |   n_missing |   pct_missing |   example |
|:------------|:--------|-----------:|------------:|--------------:|----------:|
| aptt        | float64 |        330 |         176 |      11.0622  |     25.6  |
| inr         | float64 |         39 |         158 |       9.93086 |      1.5  |
| pt          | float64 |        204 |         158 |       9.93086 |     16.7  |
| calcium     | float64 |         64 |          68 |       4.27404 |     10.1  |
| temperature | float64 |        107 |          35 |       2.19987 |     36.72 |
| glucose     | float64 |        215 |          20 |       1.25707 |    124    |
| wbc         | float64 |        282 |          19 |       1.19422 |     23.7  |
| sodium      | float64 |         43 |          18 |       1.13136 |    135    |
| aniongap    | float64 |         27 |          17 |       1.06851 |     12    |
| bicarbonate | float64 |         38 |          17 |       1.06851 |     20    |
| bun         | float64 |        106 |          17 |       1.06851 |     35    |
| chloride    | float64 |         48 |          17 |       1.06851 |     99    |
| platelets   | float64 |        413 |          17 |       1.06851 |    288    |
| potassium   | float64 |         42 |          17 |       1.06851 |      4.7  |
| creatinine  | float64 |         67 |          16 |       1.00566 |      2.3  |


**Missingness (top 15):**

| column      | dtype   |   n_missing |   pct_missing |
|:------------|:--------|------------:|--------------:|
| aptt        | float64 |         176 |      11.0622  |
| pt          | float64 |         158 |       9.93086 |
| inr         | float64 |         158 |       9.93086 |
| calcium     | float64 |          68 |       4.27404 |
| temperature | float64 |          35 |       2.19987 |
| glucose     | float64 |          20 |       1.25707 |
| wbc         | float64 |          19 |       1.19422 |
| sodium      | float64 |          18 |       1.13136 |
| platelets   | float64 |          17 |       1.06851 |
| bicarbonate | float64 |          17 |       1.06851 |
| bun         | float64 |          17 |       1.06851 |
| aniongap    | float64 |          17 |       1.06851 |
| potassium   | float64 |          17 |       1.06851 |
| chloride    | float64 |          17 |       1.06851 |
| creatinine  | float64 |          16 |       1.00566 |


## Test metrics

| model         |   M |   roc_auc |   accuracy |       f1 |   precision |   recall |   brier |   avg_precision |   thr |
|:--------------|----:|----------:|-----------:|---------:|------------:|---------:|--------:|----------------:|------:|
| random_forest |   3 |  0.750511 |   0.797071 | 0.312057 |    0.578947 | 0.213592 | 0.14716 |      nan        | nan   |
| random_forest |   3 |  0.750511 |   0.797071 | 0.312057 |    0.578947 | 0.213592 | 0.14716 |        0.439239 |   0.5 |


## ROC / PR / Calibration

**ROC (test)**  



![](outputs/roc_test_random_forest.png)

**PR (test)**  
![](outputs/pr_test_random_forest.png)

**Calibration (test)**  
![](outputs/calibration_test_random_forest.png)


## Threshold scan

_scan not found_

- Recommendations: _not generated_


## Decision Curve Analysis (DCA)

- **Prevalence**: 0.2155


**Net Benefit**  
![](outputs/dca_nb_random_forest.png)

**Standardized Net Benefit**  
![](outputs/dca_snb_random_forest.png)


## Explainability (SHAP)

**Top-20 by mean(|SHAP|)**

| feature     |   mean_abs_shap |
|:------------|----------------:|
| pt          |       0.0765969 |
| bun         |       0.0752722 |
| platelets   |       0.0543798 |
| heart_rate  |       0.0500208 |
| sbp         |       0.0454873 |
| bicarbonate |       0.0373531 |
| spo2        |       0.0352402 |
| hemoglobin  |       0.0250608 |


**Importance (bar)**  
![](outputs/shap/shap_importance_bar_random_forest.png)


**Summary (beeswarm)**  
![](outputs/shap/shap_summary_random_forest.png)


## Per-m AUC (before/after)

| model         |      auc |   m_index |   auc_before |   auc_after |
|:--------------|---------:|----------:|-------------:|------------:|
| random_forest | 0.787275 |       nan |          nan |         nan |
| random_forest | 0.751469 |         1 |          nan |         nan |
| random_forest | 0.751288 |         2 |          nan |         nan |
| random_forest | 0.746667 |         3 |          nan |         nan |
| random_forest | 0.750511 |         0 |          nan |         nan |
| random_forest | 0.751469 |         1 |          nan |         nan |
| random_forest | 0.751288 |         2 |          nan |         nan |
| random_forest | 0.746667 |         3 |          nan |         nan |
| random_forest | 0.750511 |         0 |          nan |         nan |


## Config snapshot

```yaml
project:
  seed: 42
  results_dir: outputs
  cache_dir: data_processed
  artifacts_dir: outputs/artifacts
  n_jobs: -1
  deterministic: true
data:
  raw_dir: data_raw
  processed_dir: data_processed
  input_format: excel
  input_files:
  - data.xlsx
  id_col: null
  outcome_col: mor_hospital
  positive_label: 1
cohort:
  include:
  - adult_only: true
  - first_icu_stay: true
  - lymphoma: true
  exclude:
  - stay_lt_24h: true
features:
  continuous:
  - age
  - heart_rate
  - sbp
  - spo2
  - platelets
  - bicarbonate
  - bun
  - pt
  - aptt
  - temperature
  - hematocrit
  - hemoglobin
  - wbc
  - aniongap
  - calcium
  - chloride
  - creatinine
  - glucose
  - sodium
  - potassium
  - inr
  categorical:
  - gender
  comorbidities:
  - myocardial_infarct
  - heart_failure
  - peripheral_vascular
  - dementia
  - cerebrovascular_disease
  - chronic_pulmonary_disease
  - rheumatic_disease
  - peptic_ulcer_disease
  - mild_liver_disease
  - diabetes
  - paraplegia
  - renal_disease
  - malignant_cancer
  - severe_liver_disease
  - metastatic_solid_tumor
  - aids
missing_data:
  strategy: mice
  mice:
    m: 20
    max_iter: 15
    estimator: random_forest
    sample_posterior: false
    fit_scope: train_only
    rubin_pooling: true
    random_state: 42
    mice_output_dir: outputs/mi_runs
split:
  method: stratified_holdout
  train_size: 0.7
  test_size: 0.3
  shuffle: true
  random_state: 42
  index_cache: outputs/artifacts/split_idx.joblib
preprocess:
  scale_continuous: true
  scaler: standard
  outlier_clip_quantiles:
  - 0.001
  - 0.999
  categorical_encoding: onehot_drop_first
  impute_before_scaling: true
imbalance:
  use_class_weight: true
  positive_class_weight: 3.65
  resampling_in_cv: random_over_sampler
  resampling_scope: fold_train_only
selection:
  lasso:
    enabled: true
    cv: 5
    lock_features_across_m: true
    freq_threshold: 0.5
    save_path: outputs/artifacts/selected_features.json
models:
  compare:
  - logistic
  - ridge
  - lda
  - svm
  - knn
  - gaussian_nb
  - decision_tree
  - random_forest
  - extra_trees
  - gbdt
  - xgboost
  - lightgbm
  - catboost
  - adaboost
  - mlp
  optimization:
    library: optuna
    n_trials: 100
    cv_folds: 5
    direction: maximize
  primary_metric: roc_auc
  catboost_defaults:
    iterations: 1000
    learning_rate: 0.03
    depth: 6
    loss_function: Logloss
    eval_metric: AUC
    thread_count: -1
    random_seed: 42
    od_type: IncToDec
    od_wait: 50
    verbose: false
evaluation:
  metrics:
  - roc_auc
  - pr_auc
  - f1
  - accuracy
  - precision
  - recall
  - brier
  calibration: true
  calibration_method:
  - none
  - isotonic
  thresholds:
  - 0.2
  - 0.3
  - 0.4
  - 0.5
  youden: true
  curves:
    roc: true
    pr: true
    calibration_curves: true
  report_ci: false
explainability:
  shap:
    enabled: true
    summary: true
    dependence: true
    waterfall: true
    sample_size: 4000
    save_dir: outputs/shap
output:
  figures: outputs/figures/
  tables: outputs/tables/
  logs: outputs/logs/
```
