import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def core_distances(X, min_samples):
    from heapq import nsmallest
    n = X.shape[0]
    core_dists = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        core_dists[i] = nsmallest(min_samples, dists)[-1]
    return core_dists

def mutual_reachability_dist(X, core_dists):
    n = X.shape[0]
    mrd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(X[i] - X[j])
            mrd[i, j] = max(core_dists[i], core_dists[j], d)
    return mrd

def hdbscan(X, min_samples=5, min_cluster_size=5):
    n = X.shape[0]
    core_dists = core_distances(X, min_samples)
    mrd = mutual_reachability_dist(X, core_dists)
    mst = minimum_spanning_tree(csr_matrix(mrd)).toarray()
    # Flatten MST edges
    edges = []
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                edges.append((i, j, mst[i, j]))
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    # Union-find for clustering
    parent = np.arange(n)
    size = np.ones(n, dtype=int)
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    clusters = n
    labels = -np.ones(n, dtype=int)
    cluster_id = 0
    for i, j, w in edges:
        u, v = find(i), find(j)
        if u != v:
            if size[u] + size[v] >= min_cluster_size:
                parent[v] = u
                size[u] += size[v]
                clusters -= 1
    # Assign cluster labels
    for i in range(n):
        root = find(i)
        if size[root] >= min_cluster_size:
            if labels[root] == -1:
                labels[root] = cluster_id
                cluster_id += 1
            labels[i] = labels[root]
    return labels

if __name__ == "__main__":
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    labels = hdbscan(X, min_samples=5, min_cluster_size=5)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab20', s=30)
    plt.title('HDBSCAN Clustering')
    plt.show()
