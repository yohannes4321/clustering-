
import sys

from kmeans_from_scratch import kmeans
from dbscan_from_scratch import dbscan
from hdbscan_from_scratch import hdbscan
from gmm_from_scratch import gmm


# Load data and set X, k
df = pd.read_csv("data/clustering_data.csv")
X = df.values
k = 3

# Choose algorithm from command-line argument
algo = None
if len(sys.argv) > 1:
    algo = sys.argv[1].lower()
else:
    print("Usage: python compare_clustering.py [kmeans|dbscan|hdbscan|gmm]")
    sys.exit(1)

if algo == "kmeans":
    gmm_labels, gmm_means, _, _ = gmm(X, k, random_state=42)
    metrics = {
        'davies_bouldin': davies_bouldin_score_scratch(X, gmm_labels),
        'calinski_harabasz': calinski_harabasz_score_scratch(X, gmm_labels),
        'n_clusters': len(set(gmm_labels)),
        'n_outliers': np.sum(gmm_labels == -1) if -1 in gmm_labels else 0
    }
    print("GMM Metrics:", metrics)
    plt.scatter(X[:,0], X[:,1], c=gmm_labels, cmap='tab20c', s=40, edgecolor='k')
    plt.scatter(gmm_means[:,0], gmm_means[:,1], c='red', marker='x', s=120, label='GMM Means')
    plt.title('GMM Clustering')
    plt.legend()
    plt.tight_layout()
    plt.show()
elif algo == "dbscan":
    dbscan_eps = 0.4
    dbscan_min_samples = 4
    dbscan_labels = dbscan(X, eps=dbscan_eps, min_samples=dbscan_min_samples)
    metrics = {
        'davies_bouldin': davies_bouldin_score_scratch(X, dbscan_labels),
        'calinski_harabasz': calinski_harabasz_score_scratch(X, dbscan_labels),
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'n_outliers': np.sum(dbscan_labels == -1)
    }
    print("DBSCAN Metrics:", metrics)
    plt.scatter(X[:,0], X[:,1], c=dbscan_labels, cmap='tab10', s=40, edgecolor='k')
    plt.scatter(X[dbscan_labels==-1,0], X[dbscan_labels==-1,1], c='red', s=60, edgecolor='k', label='DBSCAN Outliers')
    plt.title(f'DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})')
    plt.legend()
    plt.tight_layout()
    plt.show()
elif algo == "hdbscan":
    hdbscan_min_samples = 4
    hdbscan_min_cluster_size = 4
    hdbscan_labels = hdbscan(X, min_samples=hdbscan_min_samples, min_cluster_size=hdbscan_min_cluster_size)
    metrics = {
        'davies_bouldin': davies_bouldin_score_scratch(X, hdbscan_labels),
        'calinski_harabasz': calinski_harabasz_score_scratch(X, hdbscan_labels),
        'n_clusters': len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
        'n_outliers': np.sum(hdbscan_labels == -1)
    }
    print("HDBSCAN Metrics:", metrics)
    plt.scatter(X[:,0], X[:,1], c=hdbscan_labels, cmap='tab20', s=40, edgecolor='k')
    plt.scatter(X[hdbscan_labels==-1,0], X[hdbscan_labels==-1,1], c='red', s=60, edgecolor='k', label='HDBSCAN Outliers')
    plt.title(f'HDBSCAN (min_samples={hdbscan_min_samples}, min_cluster_size={hdbscan_min_cluster_size})')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Unknown algorithm. Use one of: kmeans, dbscan, hdbscan, gmm")
    sys.exit(1)

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
# Clustering Results Summary
| Algorithm | Silhouette | Parameters | #Clusters |
|-----------|------------|------------|-----------|
| K-means   | {kmeans_sil:.3f}     | k={k}        | {len(set(kmeans_labels))}        |
| DBSCAN    | {dbscan_sil:.3f}     | eps={dbscan_eps}, min_samples={dbscan_min_samples} | {dbscan_n_clusters}        |
| HDBSCAN   | {hdbscan_sil:.3f}     | min_samples={hdbscan_min_samples}, min_cluster_size={hdbscan_min_cluster_size} | {hdbscan_n_clusters}        |
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
    if -1 in counts:
        del counts[-1]
    ax.bar(list(counts.keys()), list(counts.values()), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Cluster Label')
    ax.set_ylabel('Size')
    ax.set_xlabel('Cluster Label')
    ax.set_ylabel('Size')
print(f"DBSCAN Silhouette: {dbscan_sil:.3f} (eps={dbscan_eps}, min_samples={dbscan_min_samples}, clusters={dbscan_n_clusters})")
print(f"HDBSCAN Silhouette: {hdbscan_sil:.3f} (min_samples={hdbscan_min_samples}, min_cluster_size={hdbscan_min_cluster_size}, clusters={hdbscan_n_clusters})")

plt.tight_layout()
plt.show()