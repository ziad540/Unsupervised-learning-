import numpy as np


class ManualPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
        self.eigen_values = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Standardizing the data
        # move the data to have mean 0 and std 1
        X_standardized = (X - self.mean) / self.std

        # Covariance matrix, tells us how the dimensions vary from the mean with respect to each other
        cov_matrix = np.cov(X_standardized.T)

        # Eigenvectors: directions, Eigenvalues: amount of variance in these directions
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        indexes = np.argsort(eigen_values)[::-1] # sort the eigen values in descending order
        self.eigen_values = eigen_values[indexes]
        sorted_vectors = eigen_vectors[:, indexes]

        self.components = sorted_vectors[:, : self.n_components] # from eigen vectors, pick the top n_components

        total_var = np.sum(self.eigen_values)
        self.explained_variance_ratio = (
            self.eigen_values[: self.n_components] / total_var # How much information did we keep?
        )

    # projection phase
    def transform(self, X):
        X_standardized = (X - self.mean) / self.std
        # z = U.T (x - mu)
        return np.dot(X_standardized, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_projected):
        # x = U z + mu
        X_original = np.dot(X_projected, self.components.T)
        X_original = X_original * self.std + self.mean
        return X_original

    def compute_reconstruction_error(self, X, X_projected):
        X_reconstructed = self.inverse_transform(X_projected)
        error = np.mean((X - X_reconstructed) ** 2)
        return error
