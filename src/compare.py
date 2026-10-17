#Run both of the models and compare their performance
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import scratch_model
import sklearn_model


def evaluate(name, runner):
    start = time.perf_counter()
    y_test, y_pred, _ = runner()
    elapsed = time.perf_counter() - start
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision(churn)": precision_score(y_test, y_pred, zero_division=0),
        "recall(churn)": recall_score(y_test, y_pred, zero_division=0),
        "f1(churn)": f1_score(y_test, y_pred, zero_division=0),
        "time (s)": elapsed,
    }, y_test


if __name__ == "__main__":
    scratch_res, y_test = evaluate("scratch", scratch_model.run)
    sklearn_res, _ = evaluate("sklearn", sklearn_model.run)

    table = pd.DataFrame([scratch_res, sklearn_res]).set_index("model").round(4)
    print(table.to_string())

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nBaseline (always predict major class): {baseline:.4f}")