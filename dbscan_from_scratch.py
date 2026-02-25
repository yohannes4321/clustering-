import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

def region_query(X, point_idx, eps):
    dists = np.linalg.norm(X - X[point_idx], axis=1)
    return np.where(dists <= eps)[0]

def dbscan(X, eps, min_samples):
    n = X.shape[0]
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(X, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = deque(neighbors)
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = region_query(X, j, eps)
                    if len(j_neighbors) >= min_samples:
                        queue.extend(j_neighbors)
                if labels[j] == -1:
                    labels[j] = cluster_id
                elif labels[j] == -1 or labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

if __name__ == "__main__":
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    eps = 0.5
    min_samples = 5
    labels = dbscan(X, eps, min_samples)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=30)
    plt.title('DBSCAN Clustering')
    plt.show()
