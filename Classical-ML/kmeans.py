import numpy as np
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return labels
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    



if __name__ == '__main__':
    print("\n=== Simple Test Case ===")

    # Create simple 2D data
    X_simple = np.array([
        [1, 1], [1.5, 2], [2, 1],     # Cluster 1
        [8, 8], [8.5, 9], [9, 8],     # Cluster 2
    ])

    print("Simple dataset:")
    print(X_simple)

    # Test with k=2
    simple_kmeans = KMeans(k=2, max_iters=10)
    simple_labels = simple_kmeans.fit(X_simple)

    print(f"Cluster assignments: {simple_labels}")
    print(f"Centroids:\n{simple_kmeans.centroids}")
    print("Expected: Two clear clusters around [1.5, 1.33] and [8.5, 8.33]")