import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans_from_scratch import kmeans
from dbscan_from_scratch import dbscan
from hdbscan_from_scratch import hdbscan
from gmm_from_scratch import gmm

gmm_labels, gmm_means, _, _ = gmm(X, k, random_state=42)
def silhouette_score_scratch(X, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) == len(labels):
        return np.nan
    silhouette_vals = []
    for i in range(len(X)):
        same_cluster = labels == labels[i]
        other_clusters = [l for l in unique_labels if l != labels[i] and l != -1]
        if len(other_clusters) == 0 or np.sum(same_cluster) == 1 or labels[i] == -1:
            silhouette_vals.append(0)
            continue
        a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1))
        b = np.min([
            np.mean(np.linalg.norm(X[i] - X[labels == l], axis=1))
            for l in other_clusters if np.any(labels == l)
        ])
        s = (b - a) / max(a, b)
        silhouette_vals.append(s)
    return np.mean(silhouette_vals)

dbscan_results = []
for eps in [0.2, 0.3, 0.4, 0.5, 0.6]:
    for min_samples in [3, 4, 5]:
        labels = dbscan(X, eps=eps, min_samples=min_samples)
        score = silhouette_score_scratch(X, labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        dbscan_results.append((eps, min_samples, score, n_clusters, labels))

dbscan_results = sorted(dbscan_results, key=lambda x: (-(x[2] if x[2] is not None else -1)))
best_dbscan = dbscan_results[0]
dbscan_labels = best_dbscan[4]
hdbscan_results = []
for min_samples in [3, 4, 5]:
    for min_cluster_size in [3, 4, 5]:
        labels = hdbscan(X, min_samples=min_samples, min_cluster_size=min_cluster_size)
        score = silhouette_score_scratch(X, labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        hdbscan_results.append((min_samples, min_cluster_size, score, n_clusters, labels))

hdbscan_results = sorted(hdbscan_results, key=lambda x: (-(x[2] if x[2] is not None else -1)))
best_hdbscan = hdbscan_results[0]
hdbscan_labels = best_hdbscan[4]
kmeans_sil = silhouette_score_scratch(X, kmeans_labels)
dbscan_sil = best_dbscan[2]
hdbscan_sil = best_hdbscan[2]
gmm_sil = silhouette_score_scratch(X, gmm_labels)
print(f"K-means Silhouette: {kmeans_sil:.3f}")
print(f"Best DBSCAN Silhouette: {dbscan_sil:.3f} (eps={best_dbscan[0]}, min_samples={best_dbscan[1]}, clusters={best_dbscan[3]})")
print(f"Best HDBSCAN Silhouette: {hdbscan_sil:.3f} (min_samples={best_hdbscan[0]}, min_cluster_size={best_hdbscan[1]}, clusters={best_hdbscan[3]})")
print(f"GMM Silhouette: {gmm_sil:.3f}")
summary = f"""
\n# Clustering Results Summary\n
| Algorithm | Silhouette | Parameters | #Clusters |
|-----------|------------|------------|-----------|
| K-means   | {kmeans_sil:.3f}     | k={k}        | {len(set(kmeans_labels))}        |
| DBSCAN    | {dbscan_sil:.3f}     | eps={best_dbscan[0]}, min_samples={best_dbscan[1]} | {best_dbscan[3]}        |
| HDBSCAN   | {hdbscan_sil:.3f}     | min_samples={best_hdbscan[0]}, min_cluster_size={best_hdbscan[1]} | {best_hdbscan[3]}        |
| GMM       | {gmm_sil:.3f}     | k={k}        | {len(set(gmm_labels))}        |
"""
with open("README.md", "a") as f:
    f.write(summary)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs[0,0].scatter(X[:,0], X[:,1], c=kmeans_labels, cmap='viridis', s=40, edgecolor='k')
axs[0,0].scatter(kmeans_centroids[:,0], kmeans_centroids[:,1], c='red', marker='x', s=120, label='Centroids')
axs[0,0].set_title('K-means Clustering')
axs[0,0].legend()
axs[0,1].scatter(X[:,0], X[:,1], c=dbscan_labels, cmap='tab10', s=40, edgecolor='k')
axs[0,1].set_title(f'DBSCAN (eps={best_dbscan[0]}, min_samples={best_dbscan[1]})')
axs[0,2].scatter(X[:,0], X[:,1], c=hdbscan_labels, cmap='tab20', s=40, edgecolor='k')
axs[0,2].set_title(f'HDBSCAN (min_samples={best_hdbscan[0]}, min_cluster_size={best_hdbscan[1]})')
axs[1,0].scatter(X[:,0], X[:,1], c='grey', s=40, edgecolor='k', label='All Points')
axs[1,0].scatter(X[dbscan_labels==-1,0], X[dbscan_labels==-1,1], c='red', s=60, edgecolor='k', label='DBSCAN Outliers')
axs[1,0].set_title('DBSCAN Outliers')
axs[1,0].legend()
axs[1,1].scatter(X[hdbscan_labels==-1,0], X[hdbscan_labels==-1,1], c='red', s=60, edgecolor='k', label='HDBSCAN Outliers')
axs[1,1].set_title('HDBSCAN Outliers')
axs[1,1].legend()
def plot_cluster_sizes(ax, labels, title):
    from collections import Counter
    counts = Counter(labels)
    if -1 in counts: del counts[-1]
    ax.bar(list(counts.keys()), list(counts.values()), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Cluster Label')
    ax.set_ylabel('Size')

plot_cluster_sizes(axs[1,2], kmeans_labels, 'K-means Cluster Sizes')

plt.tight_layout()
plt.show()