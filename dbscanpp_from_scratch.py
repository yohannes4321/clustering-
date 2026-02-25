import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dbscanpp(X, k, eps, min_samples):
    # DBSCAN++: Improved centroid initialization for DBSCAN
    n = X.shape[0]
    centroids = [X[np.random.randint(n)]]
    for _ in range(1, k):
        dists = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        next_centroid = X[np.random.choice(n, p=probs)]
        centroids.append(next_centroid)
    centroids = np.array(centroids)
    # Use DBSCAN with these centroids as seeds (for demonstration)
    # In practice, DBSCAN++ is more about parameter tuning and initialization
    # Here, we just visualize the centroids
    return centroids

if __name__ == "__main__":
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    k = 3
    centroids = dbscanpp(X, k, eps=0.5, min_samples=5)
    plt.scatter(X[:,0], X[:,1], s=30, alpha=0.7)
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=100, label='DBSCAN++ Seeds')
    plt.title('DBSCAN++ Initialization')
    plt.legend()
    plt.show()
