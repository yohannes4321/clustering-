import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# K-means implementation from scratch
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=None):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return labels, centroids

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    # Run K-means
    k = 3  # You may tune this
    labels, centroids = kmeans(X, k, random_state=42)
    # Plot
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=30)
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=100, label='Centroids')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()
