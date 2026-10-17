import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from data import MAX_DEPTH, N_TREES, RANDOM_STATE, SUBSAMPLE_RATIO, get_splits


class SimpleDecisionTree:
    def __init__(self, max_depth=None, criterion="entropy", max_features=None,
                 min_samples_split=2, n_thresholds=32, rng=None):
        self.max_depth = max_depth
        self.criterion = criterion          # 'entropy' or 'gini'
        self.max_features = max_features    # None | 'sqrt' | int
        self.min_samples_split = min_samples_split
        self.n_thresholds = n_thresholds    
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tree = None

    # Impurity and gain calculations
    def _impurity(self, y):
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        return -np.sum(probs * np.log2(probs))

    def _gain(self, X, y, feature_idx, threshold, parent_imp):
        left = X[:, feature_idx] <= threshold
        n_left = left.sum()
        n_right = len(y) - n_left
        if n_left == 0 or n_right == 0:      
            return -1.0
        child_imp = (n_left * self._impurity(y[left])
                     + n_right * self._impurity(y[~left])) / len(y)  # weighted impurity
        return parent_imp - child_imp         # high gain=> good split

    # Candidate thresholds and feature selection
    def _candidate_thresholds(self, col):
        values = np.unique(col)
        if len(values) <= 1:
            return []
        if len(values) > self.n_thresholds:   # feature liên tục: dùng quantile
            qs = np.linspace(0, 1, self.n_thresholds + 2)[1:-1]
            return np.unique(np.quantile(col, qs))
        return (values[:-1] + values[1:]) / 2  # trung điểm giữa các giá trị liền kề

    def _n_features_to_try(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def _best_split(self, X, y):
        n_features = X.shape[1]
        feats = self.rng.choice(n_features, self._n_features_to_try(n_features), replace=False)
        parent_imp = self._impurity(y)

        best_gain, best_feature, best_threshold = 0.0, None, None  # require gain > 0
        for f in feats:
            for t in self._candidate_thresholds(X[:, f]):
                gain = self._gain(X, y, f, t, parent_imp)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, f, t
        return best_feature, best_threshold

    # Build and predict
    def _build(self, X, y, depth):
        majority = int(np.bincount(y).argmax())
        if (len(np.unique(y)) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return majority

        feature, threshold = self._best_split(X, y)
        if feature is None:                   
            return majority

        left = X[:, feature] <= threshold
        # Node = (feature, threshold, left_subtree, right_subtree); leaf = int
        return (feature, threshold,
                self._build(X[left], y[left], depth + 1),
                self._build(X[~left], y[~left], depth + 1))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        self.tree = self._build(X, y, depth=0)
        return self

    def _predict_single(self, x, node):
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if x[feature] <= threshold else right
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, subsample_ratio=0.8,
                 max_features="sqrt", criterion="entropy", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample_ratio = subsample_ratio
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.trees = []                       # reset to fit
        for _ in range(self.n_trees):
            # Bootstrap: sample with replacement
            idxs = rng.choice(n_samples, int(n_samples * self.subsample_ratio), replace=True)
            tree = SimpleDecisionTree(
                max_depth=self.max_depth, criterion=self.criterion,
                max_features=self.max_features, rng=rng,
            )
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        return np.array([np.bincount(col).argmax() for col in preds.T])  # majority vote


def run():
    X_train, X_test, y_train, y_test = get_splits()
    rf = RandomForest(n_trees=N_TREES, max_depth=MAX_DEPTH,
                      subsample_ratio=SUBSAMPLE_RATIO, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    return y_test.to_numpy(), rf.predict(X_test), rf


if __name__ == "__main__":
    y_test, y_pred, _ = run()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))