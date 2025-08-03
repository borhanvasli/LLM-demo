class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        self.explained_variance = eigenvalues[idx[:self.n_components]]
        
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)