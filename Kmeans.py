import time
import numpy as np

class Kmeans:
    def __init__(self, k, max_iter=300, tol=1e-4, init="kpp"):
        self.k = k
        self.max_iter = max_iter 
        self.tol = tol # if reached, stop
        self.init = init

    def _init_centroids(self, X):
        n_samples = X.shape[0]

        if self.init == "random":
            idx = np.random.choice(n_samples, self.k, replace=False)
            return X[idx]

        # K-Means++
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.k):
            # Distance to nearest centroid
            distances = np.min(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]),
                axis=0
            )
            
            # Sanity check: Ensure no negative distances due to float errors
            distances = np.maximum(distances, 0)
            
            total = distances.sum()

            # Check for 0 OR NaN
            if total <= 1e-20 or np.isnan(total):
                idx = np.random.randint(n_samples)
                centroids.append(X[idx])
                continue

            probs = distances / total
            
            # Double check probs explicitly before choice
            if np.isnan(probs).any():
                 # Fallback to uniform if probs are broken
                 probs = np.ones(n_samples) / n_samples
            
            idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)


    def fit(self, X):
        self.centroids = self._init_centroids(X)
        self.inertia_history = []

        for iteration in range(self.max_iter):
            # Assign clusters
            dist = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            self.labels = np.argmin(dist, axis=1)

            new_centroids = []

            for i in range(self.k):
                cluster_points = X[self.labels == i]

                if len(cluster_points) == 0:
                    new_centroids.append(
                        X[np.random.randint(len(X))]
                    )
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            new_centroids = np.array(new_centroids)

            # Inertia
            inertia = np.sum((X - self.centroids[self.labels]) ** 2)
            self.inertia_history.append(inertia)

            # Convergence check
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                self.centroids = new_centroids
                self.n_iter_ = iteration + 1
                break

            self.centroids = new_centroids

        else:
            self.n_iter_ = self.max_iter


    def predict(self, X):
        dist = np.linalg.norm(X[:, None] - self.centroids, axis=2) # Distance matrix
        return np.argmin(dist, axis=1) # Assign clusters
    

