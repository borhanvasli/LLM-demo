import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iters=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
    def initialize_parameters(self, X):
        """Initialize GMM parameters"""
        if self.random_state:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize mixing coefficients (equal weights)
        self.pi = np.ones(self.n_components) / self.n_components
        
        # Initialize means randomly from data points
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()
        
        # Initialize covariances as identity matrices scaled by data variance
        data_var = np.var(X, axis=0)
        self.covariances = []
        for k in range(self.n_components):
            cov = np.eye(n_features) * np.mean(data_var)
            self.covariances.append(cov)
        self.covariances = np.array(self.covariances)
        
        # Initialize responsibilities
        self.responsibilities = np.zeros((n_samples, self.n_components))
        
    def multivariate_gaussian(self, X, mean, cov):
        """Calculate multivariate Gaussian probability density"""
        n_features = len(mean)
        
        # Add small value to diagonal for numerical stability
        cov_reg = cov + np.eye(n_features) * 1e-6
        
        # Calculate probability density
        diff = X - mean
        inv_cov = np.linalg.inv(cov_reg)
        
        # Vectorized computation
        exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov_reg))
        
        return norm_const * np.exp(exp_term)
    
    def e_step(self, X):
        """Expectation step: calculate responsibilities"""
        n_samples = X.shape[0]
        
        # Calculate weighted probabilities for each component
        weighted_probs = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            prob = self.multivariate_gaussian(X, self.means[k], self.covariances[k])
            weighted_probs[:, k] = self.pi[k] * prob
        
        # Calculate responsibilities (normalize)
        total_prob = np.sum(weighted_probs, axis=1, keepdims=True)
        total_prob[total_prob == 0] = 1e-8  # Avoid division by zero
        
        self.responsibilities = weighted_probs / total_prob
        
        return np.sum(np.log(np.sum(weighted_probs, axis=1)))  # Log-likelihood
    
    def m_step(self, X):
        """Maximization step: update parameters"""
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        Nk = np.sum(self.responsibilities, axis=0)
        
        # Update mixing coefficients
        self.pi = Nk / n_samples
        
        # Update means
        for k in range(self.n_components):
            if Nk[k] > 0:
                self.means[k] = np.sum(self.responsibilities[:, k:k+1] * X, axis=0) / Nk[k]
        
        # Update covariances
        for k in range(self.n_components):
            if Nk[k] > 0:
                diff = X - self.means[k]
                weighted_diff = self.responsibilities[:, k:k+1] * diff
                self.covariances[k] = (weighted_diff.T @ diff) / Nk[k]
                
                # Add regularization for numerical stability
                self.covariances[k] += np.eye(n_features) * 1e-6
    
    def fit(self, X):
        """Fit GMM using EM algorithm"""
        self.initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        self.log_likelihoods = []
        
        for iteration in range(self.max_iters):
            # E-step
            log_likelihood = self.e_step(X)
            self.log_likelihoods.append(log_likelihood)
            
            # M-step
            self.m_step(X)
            
            # Check for convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict_proba(self, X):
        """Return cluster probabilities for each point"""
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            prob = self.multivariate_gaussian(X, self.means[k], self.covariances[k])
            probs[:, k] = self.pi[k] * prob
        
        # Normalize
        total_prob = np.sum(probs, axis=1, keepdims=True)
        total_prob[total_prob == 0] = 1e-8
        
        return probs / total_prob
    
    def predict(self, X):
        """Return most likely cluster for each point"""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def sample(self, n_samples=1):
        """Generate samples from the fitted GMM"""
        if not hasattr(self, 'means'):
            raise ValueError("Model must be fitted before sampling")
        
        # Choose components based on mixing coefficients
        components = np.random.choice(self.n_components, n_samples, p=self.pi)
        
        samples = []
        for i in range(n_samples):
            component = components[i]
            sample = np.random.multivariate_normal(
                self.means[component], 
                self.covariances[component]
            )
            samples.append(sample)
            
        return np.array(samples)
    
    def score(self, X):
        """Calculate log-likelihood of data"""
        n_samples = X.shape[0]
        log_prob = 0
        
        for i in range(n_samples):
            prob = 0
            for k in range(self.n_components):
                component_prob = self.multivariate_gaussian(
                    X[i:i+1], self.means[k], self.covariances[k]
                )[0]
                prob += self.pi[k] * component_prob
            log_prob += np.log(max(prob, 1e-8))
            
        return log_prob