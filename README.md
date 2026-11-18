# Tanzania Tourism Cost Classification

Predicting how much a tourist spends in Tanzania, as a **6-class classification** problem built on the Zindi *Tanzania Tourism Classification Challenge* dataset. The project follows a notebook-first workflow (EDA → preprocessing → baseline → feature engineering → …) and moves finished logic into a reusable `src/` pipeline, with every experiment tracked in MLflow.

---

## Problem

Given a tourist's trip profile (origin country, age group, travel companions, purpose, main activity, package contents, nights spent, etc.), predict the **expenditure range** `cost_category`:

`Lower Cost` · `Low Cost` · `Normal Cost` · `High Cost` · `Higher Cost` · `Highest Cost`

| | Train | Test |
|---|---|---|
| Rows × columns | 18,506 × 21 | 6,169 × 20 |

**Class distribution (train):** Normal 5,471 · Higher 4,865 · High 3,678 · Lower 2,567 · Low 1,566 · Highest 359 — clearly imbalanced, so macro F1 is tracked next to accuracy and class-balanced sample weights are used.

---

## Results so far

| Experiment | Accuracy | Macro F1 |
|---|---|---|
| XGBoost baseline (holdout) | 44.32% | 0.40 |
| XGBoost baseline (6-fold stratified CV) | 44.41% ± 0.35% | — |

Holdout and CV agree closely, so the baseline is stable but weak on minority classes — the reference point for the experiments below.

---

## Project status

Tracked on the [GitHub project board](https://github.com/users/kareem0088/projects/2), organised in five phases:

| Phase | Issue | Task | Status |
|---|---|---|---|
| **1. System & Data Engineering** | #1 | 01 – Project architecture & modular directory structure | ✅ Done |
| | #14 | 02 – `src/` folder and pipeline initiation | ✅ Done |
| **2. Statistical EDA & Feature Lab** | #16 | 03 – Exploratory data analysis & target distribution | ✅ Done |
| | #9 | 04 – Data cleaning, missing values, outlier treatment | ✅ Done |
| | #10 | 06 – Modular feature engineering & transformation pipeline | ✅ Done |
| **3. Modeling & Experimentation** | #4 | 05 – Train & evaluate initial baseline classifier | ✅ Done |
| | #6 | 07 – MLflow experiment tracking & model registry | 🔲 Open |
| | #21 | 08 – Generative synthetic data & training-set balancing | 🔲 Open |
| | #8 | 09 – Exploratory research & algorithm benchmarking | 🔲 Open |
| **4. Validation & Explainability (XAI)** | #5 | 10 – Cross-validation & SHAP explanations | 🔲 Open |
| **5. MLOps, Testing & Serving** | #15 | 11 – Synthetic feature generator & Kafka streaming producer | 🔲 Open |

Notebooks `05`–`08` are placeholders for the open modeling tasks.

---

## What has been done

**EDA** — missing values, duplicates, numeric and categorical profiling, target analysis, correlations, and outliers.

**Preprocessing** — schema checks between train and test, standardised country labels, corrected `main_activity` labels, validated ID uniqueness, no infinite values, consistent dtypes (`src/preprocessing.py`).

**Baseline** — XGBoost (`multi:softprob`, `max_depth=5`, `max_leaves=15`, `learning_rate=0.01`), stratified holdout + 6-fold CV, class-balanced sample weights.

**Feature engineering** (`src/features/builder.py`) — `total nights`, `total_adults`, `night_ratio`, `adults_per_night`, `male_to_female_ratio`, and a `continent` feature mapped from country. Categorical encoding lives in `src/features/encoding.py` and is applied **after** the split to avoid leakage.

**Experiment tracking** — MLflow with a local SQLite backend (`mlflow.db`).

### Next steps
1. **Experiment tracking (#6)** — finish the MLflow tracking and model registry on top of the existing SQLite setup.
2. **Synthetic data (#21)** — generate minority-class samples on training folds only (Copula / CTGAN), then re-evaluate.
3. **Benchmarking (#8)** — compare algorithms and test whether the engineered shares/ratios add value beyond the raw counts; fit encoders inside each CV fold and report macro F1 across folds.
4. **XAI (#5)** — cross-validation and SHAP explanations.
5. **Serving (#15)** — synthetic feature generator and a Kafka streaming producer.

---

## Repository structure

```text
tanzania-tourism-classification/
├── .gitignore
├── README.md
├── requirements.txt
├── mlflow.db                              # MLflow tracking store (SQLite)
├── my profile.code-profile                # VS Code profile
│
├── data/
│   ├── README.md                          # Data source & license notes
│   ├── VariableDefinitions.csv            # Column definitions
│   ├── raw/
│   │   ├── Train.csv                      # Original competition training set
│   │   └── Test.csv                       # Original competition test set
│   ├── features/
│   │   ├── train_data_FE.parquet          # Train set with engineered features
│   │   └── test_data_FE.parquet           # Test set with engineered features
│   └── processed/                         # Created by src/config.py (cleaned data, submissions)
│
├── notebooks/
│   ├── 01_statistical_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_synthetic_data_generation.ipynb 
│   ├── 06_hyperparameter_tuning.ipynb    
│   ├── 07_model_evaluation.ipynb          
│   └── 08_model_experimentation.ipynb    
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # Paths, column groups, seeds, model params, MLflow setup
│   ├── preprocessing.py                   # Data_Preprocessor: loading, validation, cleaning
│   ├── utils.py                           # EDA helpers (outliers, missing values)
│   ├── train.py                           # (planned)
│   ├── evaluate.py                        # (planned)
│   ├── predict.py                         # (planned)
│   └── features/
│       ├── builder.py                     # Feature engineering
│       └── encoding.py                    # Encoders (fit on train only)
│
└── models/
    └── baseline-model.json                # Saved XGBoost baseline
```

---

## Getting started

```bash
git clone https://github.com/kareem0088/tanzania-tourism-classification.git
cd tanzania-tourism-classification

# initialise folders and the MLflow SQLite store
python -m src.config

# build engineered features
python -m src.features.builder

# browse tracked experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

Then open the notebooks in order, starting from `notebooks/01_statistical_eda.ipynb`.

---

## Data & license

The dataset comes from the Zindi *Tanzania Tourism Classification Challenge* and is licensed **CC-BY-SA 4.0** for non-commercial, research and educational use. Only the official competition data and public open-source packages are used — no AutoML tools or external data. See [`data/README.md`](data/README.md).


## Copyright and Usage

Copyright (c) 2026 Project Owner. All rights reserved.

This repository and its source code are provided for portfolio
and educational review purposes only.

No permission is granted to copy, reproduce, modify, distribute,
or use any part of this project's source code without prior
written permission from the copyright owner.

