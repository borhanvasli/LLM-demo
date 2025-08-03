import numpy as np
from collections import defaultdict
import math

class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}  # {class: {feature_idx: {'mean': x, 'var': y}}}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate class priors P(y)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
            
        # Calculate feature statistics for each class
        for cls in self.classes:
            class_samples = X[y == cls]
            self.feature_stats[cls] = {}
            
            for feature_idx in range(n_features):
                feature_values = class_samples[:, feature_idx]
                self.feature_stats[cls][feature_idx] = {
                    'mean': np.mean(feature_values),
                    'var': np.var(feature_values) + 1e-9  # Add small value to avoid division by zero
                }
    
    def gaussian_probability(self, x, mean, var):
        """Calculate Gaussian probability density"""
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    def predict_proba_single(self, x):
        """Calculate probabilities for single sample"""
        posteriors = {}
        
        for cls in self.classes:
            # Start with class prior P(y)
            posterior = self.class_priors[cls]
            
            # Multiply by feature likelihoods P(xi|y)
            for feature_idx, feature_value in enumerate(x):
                mean = self.feature_stats[cls][feature_idx]['mean']
                var = self.feature_stats[cls][feature_idx]['var']
                likelihood = self.gaussian_probability(feature_value, mean, var)
                posterior *= likelihood
                
            posteriors[cls] = posterior
            
        # Normalize probabilities
        total = sum(posteriors.values())
        if total > 0:
            for cls in posteriors:
                posteriors[cls] /= total
                
        return posteriors
    
    def predict_single(self, x):
        """Predict single sample"""
        posteriors = self.predict_proba_single(x)
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_single(x) for x in X])
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        probabilities = []
        for x in X:
            probs = self.predict_proba_single(x)
            prob_array = [probs.get(cls, 0) for cls in self.classes]
            probabilities.append(prob_array)
        return np.array(probabilities)

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}
        self.feature_probs = {}  # {class: {feature_idx: probability}}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_features = n_features
        
        # Calculate class priors P(y)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
            
        # Calculate feature probabilities for each class
        for cls in self.classes:
            class_samples = X[y == cls]
            self.feature_probs[cls] = {}
            
            # Total count of all features for this class
            total_count = np.sum(class_samples) + self.alpha * n_features
            
            for feature_idx in range(n_features):
                # Count of this specific feature for this class
                feature_count = np.sum(class_samples[:, feature_idx]) + self.alpha
                self.feature_probs[cls][feature_idx] = feature_count / total_count
    
    def predict_proba_single(self, x):
        """Calculate probabilities for single sample"""
        posteriors = {}
        
        for cls in self.classes:
            # Start with log of class prior to avoid underflow
            log_posterior = np.log(self.class_priors[cls])
            
            # Add log likelihoods
            for feature_idx, count in enumerate(x):
                if count > 0:  # Only consider non-zero features
                    prob = self.feature_probs[cls][feature_idx]
                    log_posterior += count * np.log(prob)
                    
            posteriors[cls] = log_posterior
            
        # Convert back from log space and normalize
        max_log_prob = max(posteriors.values())
        for cls in posteriors:
            posteriors[cls] = np.exp(posteriors[cls] - max_log_prob)
            
        total = sum(posteriors.values())
        if total > 0:
            for cls in posteriors:
                posteriors[cls] /= total
                
        return posteriors
    
    def predict_single(self, x):
        """Predict single sample"""
        posteriors = self.predict_proba_single(x)
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_single(x) for x in X])

class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.feature_probs = {}  # {class: {feature_idx: P(feature=1|class)}}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate class priors
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
            
        # Calculate feature probabilities
        for cls in self.classes:
            class_samples = X[y == cls]
            n_class_samples = len(class_samples)
            self.feature_probs[cls] = {}
            
            for feature_idx in range(n_features):
                # Probability that feature=1 given class (with smoothing)
                positive_count = np.sum(class_samples[:, feature_idx]) + self.alpha
                total_count = n_class_samples + 2 * self.alpha
                self.feature_probs[cls][feature_idx] = positive_count / total_count
    
    def predict_single(self, x):
        """Predict single sample"""
        posteriors = {}
        
        for cls in self.classes:
            log_posterior = np.log(self.class_priors[cls])
            
            for feature_idx, feature_value in enumerate(x):
                prob_1 = self.feature_probs[cls][feature_idx]
                prob_0 = 1 - prob_1
                
                if feature_value == 1:
                    log_posterior += np.log(prob_1)
                else:
                    log_posterior += np.log(prob_0)
                    
            posteriors[cls] = log_posterior
            
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
