import numpy as np
from decisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=100, max_features=None, **tree_params):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.estimators = []

        # Store additional parameters for tree initialization
        self.tree_params = tree_params

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_features = self.max_features if self.max_features else int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[idx]
            y_bootstrap = y[idx]

            # Randomly select features
            feature_indices = np.random.choice(n_features, max_features, replace=False)

            # Train decision tree
            tree = DecisionTree(**self.tree_params)
            tree.fit(X_bootstrap[:, feature_indices], y_bootstrap)

            # Store decision tree and feature indices as a tuple
            self.estimators.append((tree, feature_indices))

    def predict(self, X):
        predictions = []
        for tree, feature_indices in self.estimators:
            predictions.append(tree.predict(X[:, feature_indices]))

        mean_predictions = np.mean(predictions, axis=0)
        pred = np.where(mean_predictions >= 0.2, 1, 0).astype(int)
        return pred