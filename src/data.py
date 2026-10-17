#Load and preprocess data
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank_churn.csv"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"
CATEGORICAL = ["Geography", "Gender"]

# Same parameters for both models to compare fairly
RANDOM_STATE = 42
N_TREES = 10
MAX_DEPTH = 5
SUBSAMPLE_RATIO = 0.8


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[TARGET])

    # Categorical: missing -> "Unknown", then encode to numbers
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype("category").cat.codes

    # Numerical: transform to numerical, if missing -> median
    num_cols = [c for c in df.columns if c not in CATEGORICAL + [TARGET]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def get_splits(test_size=0.2):
    df = load_data()
    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int)
    # stratify because churn is often imbalanced (~20% Exited=1)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )