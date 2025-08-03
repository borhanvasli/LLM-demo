import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def minkowski_distance(self, x1, x2, p=3):
        return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)
    
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self.minkowski_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def fit(self, X, y):
        """Store training data (lazy learning)"""
        self.X_train = X
        self.y_train = y
        
    def predict_single(self, x):
        """Predict single sample"""
        # Calculate distances to all training points
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_single(x) for x in X])

class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights  # 'uniform' or 'distance'
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def fit(self, X, y):
        """Store training data (lazy learning)"""
        self.X_train = X
        self.y_train = y
        
    def predict_single(self, x):
        """Predict single sample"""
        # Calculate distances to all training points
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = [self.y_train[i] for i in k_indices]
        k_distances = [distances[i] for i in k_indices]
        
        if self.weights == 'uniform':
            # Simple average
            return np.mean(k_nearest_values)
        elif self.weights == 'distance':
            # Weighted average (inverse distance weighting)
            if min(k_distances) == 0:  # Exact match
                return self.y_train[k_indices[0]]
            weights = [1/d for d in k_distances]
            weighted_sum = sum(w * v for w, v in zip(weights, k_nearest_values))
            return weighted_sum / sum(weights)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_single(x) for x in X])