import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < self.min_samples_split:
            return {'class': np.argmax(np.bincount(y)), 'depth': depth}

        # Find best split
        best_split = self._find_best_split(X, y)

        # Check if split is found
        if best_split is None:
            return {'class': np.argmax(np.bincount(y)), 'depth': depth}

        feature_idx, threshold = best_split

        # Split data
        left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
        right_idxs = np.where(X[:, feature_idx] > threshold)[0]

        # Check if split is meaningful
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return {'class': np.argmax(np.bincount(y)), 'depth': depth}

        # Grow left and right subtrees
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature_idx': feature_idx,
                'threshold': threshold,
                'left': left_tree,
                'right': right_tree,
                'depth': depth}

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                right_idxs = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue

                gini = self._gini_impurity(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)

        return best_split

    def _gini_impurity(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        p_left = len(left_y) / n
        p_right = len(right_y) / n

        gini_left = 1 - sum((np.bincount(left_y) / len(left_y))**2)
        gini_right = 1 - sum((np.bincount(right_y) / len(right_y))**2)

        gini = (p_left * gini_left) + (p_right * gini_right)

        return gini

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_tree(sample, self.tree))
        return predictions

    def _predict_tree(self, sample, tree):
        if 'class' in tree:
            return tree['class']

        feature_idx = tree['feature_idx']
        threshold = tree['threshold']

        if sample[feature_idx] <= threshold:
            return self._predict_tree(sample, tree['left'])
        else:
            return self._predict_tree(sample, tree['right'])
        
    def score(self, X, y):
        # Calculate and return a performance metric based on the model's predictions and the true labels
        predictions = self.predict(X)
        # Here, you can use any suitable metric, such as accuracy_score, precision_score, etc.
        # For example, if you want to use accuracy score:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self