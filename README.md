# MEGAFS:  Multi-stage Explainable Gene-Aware Feature Selection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/)

**MEGAFS** a multi-stage explainable feature selection framework that combines multivariate filtering with evolutionary optimization.

The library employs a robust **two-stage architecture**:
1.  **Search Space Pruning:** It utilizes the **mRMR** (Minimum Redundancy Maximum Relevance) algorithm to drastically reduce the initial feature space, filtering out redundant variables.
2.  **Fine-tuning Optimization:** It initializes a **Genetic Algorithm (GA)** on the reduced set to select an optimal, stable, and minimal feature subset.

This approach is validated through a case study on HIV-1 seroconversion susceptibility, benchmarking against a wide spectrum of state-of-the-art extraction and selection techniques.

---

## 🚀 Key Features

* **Hybrid Efficiency:** Combines the speed of filter methods (mRMR) with the precision of wrapper methods (Genetic Algorithms).
* **Scikit-learn Compatible:** Works natively with any sklearn classifier (RandomForest, SVM, XGBoost, etc.).
* **Automated Workflow:** Handles data loading, validation, scaling, feature selection, and cross-validation in a single function call.
* **Stability:** Performs multiple independent runs to ensure the selected features are robust and not artifacts of randomness.
* **Excel Integration:** Directly accepts `.xlsx` datasets and generates Excel reports.

---

## 🛠 Installation

To use MEGAFS, you need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn mrmr-selection sklearn-genetic xgboost openpyxl

```

*(Note: Once the package is published to PyPI, you will be able to install it simply via `pip install MEGAFS`)*

---

## 📊 Data Format

The input dataset **must** be an Excel file (`.xlsx`) structured as follows:

1. **Rows:** Represent individual samples (patients, observations).
2. **Columns:** Represent features (genes, biomarkers, variables).
3. **Target:** A specific column named **`class`** containing the target labels (0/1, integers, etc.).

| Gene_1 | Gene_2 | ... | Gene_N | class |
| --- | --- | --- | --- | --- |
| 0.54 | 1.23 | ... | 0.99 | 0 |
| 0.12 | 0.45 | ... | 0.88 | 1 |

---

## 📖 Usage / Quick Start

The main entry point is the `megafs` function. Here is a complete example of how to run an experiment:

```python
from MEGAFS import megafs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Define the path to your dataset
dataset_path = 'data/my_biomarkers.xlsx'

# 2. (Optional) Define a custom dictionary of classifiers
# If you pass None, MEGAFS uses a default set of robust classifiers (SVC, GBC, etc.)
my_classifiers = {
    'RFC': RandomForestClassifier(n_estimators=100, random_state=42),
    'GBC': GradientBoostingClassifier(random_state=42)
}

# 3. Run the MEGAFS pipeline
megafs(
    dataset=dataset_path,
    model_list=my_classifiers,
    metric='roc_auc',       # Metric to optimize
    n_splits=5,             # 5-fold Cross-Validation
    n_runs=3,               # 3 Independent runs per model for stability
    output_file='Experiment_Results'
)

```

---

## ⚙️ API Parameters

The `megafs` function accepts the following arguments:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | `str` | **Required** | Path to the `.xlsx` file containing the data. |
| `model_list` | `dict` | `None` | Dictionary `{name: model}`. If `None`, uses a default set of sklearn classifiers. |
| `factor_maxFpI` | `float` | `1` | Scaling factor for *max features per individual* (GA parameter). |
| `factor_nFiSE` | `float` | `2` | Scaling factor for *initial population size* (GA parameter). |
| `percent_feat_total` | `float` | `1` | Percentage of total features to keep after the mRMR pruning stage (range 0.0 to 1.0]. |
| `metric` | `str` | `'roc_auc'` | Scikit-learn metric name to optimize (e.g., `'accuracy'`, `'f1'`, `'roc_auc'`). |
| `n_splits` | `int` | `5` | Number of folds for Stratified Cross-Validation (Must be between 2-10). |
| `n_runs` | `int` | `3` | Number of independent GA runs (1-10) for stability analysis. |
| `output_file` | `str` | `'Results_MEGAFS'` | Base name for the generated output text and Excel files. |

---

## 📂 Output Files

The execution generates two files in your working directory based on the `output_file` name:

### 1. Log File (`.txt`)

Contains detailed execution logs:

* Global parameters used.
* **Per-run details:** Max features allowed, number of selected features, Train/Test scores.
* **Selected Features:** A list of the specific feature names selected in each run.
* **Genetic Evolution:** A list showing the fitness score evolution across generations.

### 2. Summary File (`.xlsx`)

A spreadsheet summarizing the results:

* **Mean Metric:** Average performance (e.g., ROC-AUC) across all runs.
* **Mean Features:** Average number of features selected.
* Aggregated rows for every model tested.

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

```

```