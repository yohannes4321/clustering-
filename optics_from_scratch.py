import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OPTICS implementation (basic, from scratch)
def optics(X, min_samples=5, max_eps=np.inf):
    n = X.shape[0]
    reachability = np.full(n, np.inf)
    processed = np.zeros(n, dtype=bool)
    ordering = []
    for point in range(n):
        if processed[point]:
            continue
        neighbors = np.where(np.linalg.norm(X - X[point], axis=1) <= max_eps)[0]
        processed[point] = True
        ordering.append(point)
        seeds = []
        if len(neighbors) >= min_samples:
            core_dist = np.partition(np.linalg.norm(X[neighbors] - X[point], axis=1), min_samples-1)[min_samples-1]
            for neighbor in neighbors:
                if not processed[neighbor]:
                    new_reach = max(core_dist, np.linalg.norm(X[point] - X[neighbor]))
                    if reachability[neighbor] == np.inf:
                        reachability[neighbor] = new_reach
                        seeds.append(neighbor)
                    elif new_reach < reachability[neighbor]:
                        reachability[neighbor] = new_reach
            seeds = np.array(seeds)[np.argsort(reachability[seeds])]
            for seed in seeds:
                if not processed[seed]:
                    processed[seed] = True
                    ordering.append(seed)
    return ordering, reachability

if __name__ == "__main__":
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    ordering, reachability = optics(X, min_samples=5, max_eps=0.5)
    plt.plot(np.arange(len(reachability)), reachability[ordering], marker='o')
    plt.title('OPTICS Reachability Plot')
    plt.xlabel('Point ordering')
    plt.ylabel('Reachability distance')
    plt.show()
