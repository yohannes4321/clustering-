import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gaussian Mixture Model (GMM) from scratch (EM algorithm)
def gmm(X, k, max_iters=100, tol=1e-4, random_state=None):
    np.random.seed(random_state)
    n, d = X.shape
    # Initialize means, covariances, and mixing coefficients
    means = X[np.random.choice(n, k, replace=False)]
    covs = np.array([np.cov(X, rowvar=False)] * k)
    pis = np.ones(k) / k
    log_likelihood = 0
    for _ in range(max_iters):
        # E-step: responsibilities
        resp = np.zeros((n, k))
        for i in range(k):
            diff = X - means[i]
            inv_cov = np.linalg.inv(covs[i])
            det_cov = np.linalg.det(covs[i])
            norm_const = 1.0 / (np.power(2 * np.pi, d/2) * np.sqrt(det_cov))
            exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
            resp[:, i] = pis[i] * norm_const * exp_term
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / resp_sum
        # M-step: update parameters
        Nk = resp.sum(axis=0)
        means = (resp.T @ X) / Nk[:, None]
        for i in range(k):
            diff = X - means[i]
            covs[i] = (resp[:, i][:, None] * diff).T @ diff / Nk[i]
        pis = Nk / n
        # Check for convergence (log-likelihood)
        new_log_likelihood = np.sum(np.log(resp_sum))
        if np.abs(new_log_likelihood - log_likelihood) < tol:
            break
        log_likelihood = new_log_likelihood
    labels = np.argmax(resp, axis=1)
    return labels, means, covs, pis

if __name__ == "__main__":
    df = pd.read_csv("data/clustering_data.csv")
    X = df.values
    k = 3
    labels, means, covs, pis = gmm(X, k, random_state=42)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=30)
    plt.scatter(means[:,0], means[:,1], c='red', marker='x', s=100, label='GMM Means')
    plt.title('GMM Clustering')
    plt.legend()
    plt.show()
