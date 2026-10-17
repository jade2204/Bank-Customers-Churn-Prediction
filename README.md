# Bank Customer Churn Prediction: Random Forest From Scratch vs scikit-learn

## Project Overview

This project predicts whether a bank customer will leave the bank (churn) using a **Random Forest** classifier. It implements the algorithm in two ways and compares them on identical data:

1. **From scratch**: a Decision Tree and Random Forest built with NumPy only (`scratch_model.py`).
2. **Library**: scikit-learn's `RandomForestClassifier` (`sklearn_model.py`).

The goal is to understand how Random Forest works internally (impurity, information gain, bootstrap sampling, feature subsampling, majority voting) and to check how close a hand-written version gets to a production library.

## Dataset

The dataset is loaded from `data/bank_churn.csv`. Each row is a bank customer.

| Column | Type | Description |
|---|---|---|
| `CreditScore` | Numerical | Customer's credit score |
| `Geography` | Categorical | Country of residence |
| `Gender` | Categorical | Customer's gender |
| `Age` | Numerical | Customer's age |
| `Tenure` | Numerical | Years as a bank customer |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of bank products used |
| `HasCrCard` | Binary | Has a credit card (0/1) |
| `IsActiveMember` | Binary | Is an active member (0/1) |
| `EstimatedSalary` | Numerical | Estimated salary |
| `Exited` | **Target** | 1 = churned, 0 = stayed |

`RowNumber`, `CustomerId`, and `Surname` are dropped because they are identifiers, not predictive features.

## Project Structure

```
Bank Customers Churn Prediction/
├── data/
│   └── bank_churn.csv       # dataset (place it here)
├── src/
│   ├── data.py              # loading, preprocessing, train/test split, shared settings
│   ├── scratch_model.py     # Decision Tree + Random Forest implemented with NumPy
│   ├── sklearn_model.py     # scikit-learn Random Forest
│   └── compare.py           # runs both models and prints a comparison table
├── requirements.txt
└── README.md
```

## Methodology

### Shared setup (`data.py`)

Both models use exactly the same data and settings so the comparison is fair:

- Drop identifier columns and rows with a missing target.
- Categorical columns (`Geography`, `Gender`): missing values become `"Unknown"`, then encoded as integer codes.
- Numerical columns: converted to numbers, missing values filled with the median.
- 80/20 train/test split, **stratified** on `Exited` because churn data is imbalanced, with `random_state=42`.
- Same hyperparameters: 10 trees, max depth 5, entropy criterion, `sqrt` features per split, 80% bootstrap sample per tree.

### From-scratch implementation (`scratch_model.py`)

- **Impurity:** entropy or Gini.
- **Information gain:** parent impurity minus the weighted impurity of the two child nodes.
- **Best split:** tries candidate thresholds on a random subset of features and keeps the split with the highest gain. For speed, continuous features use at most 32 quantile-based thresholds (`n_thresholds`).
- **Stopping rules:** pure node, maximum depth reached, too few samples, or no split with positive gain.
- **Random Forest:** each tree is trained on a bootstrap sample, and predictions are combined by majority vote.

### Library implementation (`sklearn_model.py`)

`RandomForestClassifier` configured with the same hyperparameters. It also prints feature importances.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

Run from the project root:

```bash
# Compare both models side by side
python src/compare.py

# Or run each model on its own
python src/scratch_model.py
python src/sklearn_model.py
```

`compare.py` prints accuracy, precision, recall, and F1 for the churn class, runtime, and a majority-class baseline.

## Results

Fill in after running `python src/compare.py`:

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | Time (s) |
|---|---|---|---|---|---|
| From scratch | | | | | |
| scikit-learn | | | | | |

Baseline accuracy (always predict "stayed"): _fill in_

### Notes on interpreting the results

- **Accuracy alone is misleading.** Roughly 80% of customers typically do not churn, so a model that always predicts "stayed" already scores about 80%. Compare each model against the baseline and look at recall and F1 for the churn class.
- **Small differences are expected.** The two versions use different random number generators and split search strategies (quantile thresholds vs exhaustive search), so results will be close but not identical.
- **Speed:** scikit-learn is much faster because it is implemented in optimized Cython, while the from-scratch version is written in pure Python and NumPy for clarity.

## Possible Improvements

- Handle class imbalance (class weights, oversampling, or threshold tuning) to improve churn recall.
- Add hyperparameter tuning (`GridSearchCV`) for the scikit-learn model.
- Use cross-validation instead of a single train/test split for steadier estimates.
- Add feature importance to the from-scratch forest for a direct comparison.

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn

## Conclusion

This project shows that a Random Forest built from first principles can approach the performance of scikit-learn's implementation on the same data, while making the inner workings of the algorithm visible. The library version remains the better choice for speed and robustness.