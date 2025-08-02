import numpy as np

class DecisionTreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        
    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def information_gain(self, y_parent, y_left, y_right):
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        
        gain = self.gini_impurity(y_parent) - \
               (weight_left * self.gini_impurity(y_left) + 
                weight_right * self.gini_impurity(y_right))
        return gain
    
    def best_split(self, X, y):

        best_gain = -1
        best_feature, best_threshold = None, None
        
        # Randomly select features to consider (key Random Forest feature)
        n_features = X.shape[1]
        if self.max_features and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        # Create leaf node if stopping criteria met
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            self.gini_impurity(y) == 0):
            
            leaf = DecisionTreeNode()
            leaf.value = np.bincount(y).argmax()  # Most common class
            return leaf
        
        # Find best split
        best_feature, best_threshold = self.best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf = DecisionTreeNode()
            leaf.value = np.bincount(y).argmax()
            return leaf
        
        # Create internal node
        node = DecisionTreeNode()
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
        
    def predict_sample(self, sample, node):
        """Predict single sample by traversing tree"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if sample[node.feature_idx] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(sample, self.root) for sample in X])
    
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=10):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
        
    def get_max_features(self, n_features):
        """Calculate number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:  # 'auto' or None - use all features
            return n_features
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.max_features_to_use = self.get_max_features(self.n_features)
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Create tree with feature subsampling capability
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                max_features=self.max_features_to_use
            )
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                 axis=0, arr=predictions)



if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load a small dataset (Iris, but let's use only 2 classes for simplicity)
    iris = load_iris()
    X = iris.data[iris.target != 2]
    y = iris.target[iris.target != 2]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test your Random Forest
    rf = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
