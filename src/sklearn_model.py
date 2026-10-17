import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data import MAX_DEPTH, N_TREES, RANDOM_STATE, SUBSAMPLE_RATIO, get_splits


def run():
    X_train, X_test, y_train, y_test = get_splits()
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        criterion="entropy",
        max_features="sqrt",
        bootstrap=True,
        max_samples=SUBSAMPLE_RATIO,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf.feature_names_ = list(X_train.columns)
    return y_test.to_numpy(), y_pred, rf


if __name__ == "__main__":
    y_test, y_pred, rf = run()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    importances = pd.Series(rf.feature_importances_, index=rf.feature_names_)
    print("Feature Importances:")
    print(importances.sort_values(ascending=False))