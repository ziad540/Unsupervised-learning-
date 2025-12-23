import numpy as np

class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4, init="kpp"):
        self.k = k
        self.max_iter = max_iter 
        self.tol = tol # if reached, stop
        self.init = init

    def _init_centroids(self, X):
        if self.init == "random":
            idx = np.random.choice(len(X), self.k, replace=False)
            return X[idx]

        # KMeans++
        centroids = [X[np.random.randint(len(X))]] # Choose first centroid randomly
        for _ in range(1, self.k):
            dist = np.min([np.sum((X-c)**2, axis=1) for c in centroids], axis=0) # Distance to nearest centroid
            probs = dist / dist.sum() # Probability proportional to squared distance
            centroids.append(X[np.random.choice(len(X), p=probs)]) # Choose next centroid
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)
        self.inertia_history = []

        for _ in range(self.max_iter):
            dist = np.linalg.norm(X[:, None] - self.centroids, axis=2) # Distance matrix
            self.labels = np.argmin(dist, axis=1) # Assign clusters

            new_centroids = np.array([ # Update centroids
                X[self.labels == i].mean(axis=0)  
                for i in range(self.k)
            ])

            inertia = np.sum((X - self.centroids[self.labels]) ** 2) # Compute inertia
            self.inertia_history.append(inertia)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol: # Check for convergence
                break

            self.centroids = new_centroids # Update centroids

    def predict(self, X):
        dist = np.linalg.norm(X[:, None] - self.centroids, axis=2) # Distance matrix
        return np.argmin(dist, axis=1) # Assign clusters