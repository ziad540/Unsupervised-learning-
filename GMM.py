import numpy as np


class GaussianMixtureModel:
    def __init__(self, n_components, covariance_type="full",
                 max_iter=100, tol=1e-4, eps=1e-6, random_state=None):
        self.K = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

        if random_state is not None:
            np.random.seed(random_state)

        # Parameters learned during training
        self.pi = None
        self.mu = None
        self.sigma = None
        self.log_likelihoods = []

    # ========================
    # Core math
    # ========================
    def _gaussian_pdf(self, X, mean, cov):
        N, D = X.shape
        cov = cov + self.eps * np.eye(D)

        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)

        norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * det)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

        return norm_const * np.exp(exponent)

    # ========================
    # Initialization
    # ========================
    def _initialize_parameters(self, X):
        N, D = X.shape

        self.pi = np.ones(self.K) / self.K
        indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[indices]

        if self.covariance_type == "full":
            self.sigma = np.array([np.cov(X.T) for _ in range(self.K)])
        elif self.covariance_type == "tied":
            self.sigma = np.cov(X.T)
        elif self.covariance_type == "diag":
            self.sigma = np.array([np.var(X, axis=0) for _ in range(self.K)])
        elif self.covariance_type == "spherical":
            self.sigma = np.array([np.var(X) for _ in range(self.K)])
        else:
            raise ValueError("Invalid covariance type")

    # ========================
    # E-step
    # ========================
    def _e_step(self, X):
        N, D = X.shape
        gamma = np.zeros((N, self.K))

        for k in range(self.K):
            if self.covariance_type == "full":
                cov = self.sigma[k]
            elif self.covariance_type == "tied":
                cov = self.sigma
            elif self.covariance_type == "diag":
                cov = np.diag(self.sigma[k])
            elif self.covariance_type == "spherical":
                cov = np.eye(D) * self.sigma[k]

            gamma[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], cov)

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    # ========================
    # M-step
    # ========================
    def _m_step(self, X, gamma):
        N, D = X.shape
        Nk = gamma.sum(axis=0)

        self.pi = Nk / N
        self.mu = (gamma.T @ X) / Nk[:, None]

        if self.covariance_type == "full":
            self.sigma = np.zeros((self.K, D, D))
            for k in range(self.K):
                diff = X - self.mu[k]
                self.sigma[k] = (gamma[:, k][:, None] * diff).T @ diff / Nk[k]

        elif self.covariance_type == "tied":
            self.sigma = np.zeros((D, D))
            for k in range(self.K):
                diff = X - self.mu[k]
                self.sigma += (gamma[:, k][:, None] * diff).T @ diff
            self.sigma /= N

        elif self.covariance_type == "diag":
            self.sigma = np.zeros((self.K, D))
            for k in range(self.K):
                diff = X - self.mu[k]
                self.sigma[k] = (gamma[:, k][:, None] * diff ** 2).sum(axis=0) / Nk[k]

        elif self.covariance_type == "spherical":
            self.sigma = np.zeros(self.K)
            for k in range(self.K):
                diff = X - self.mu[k]
                self.sigma[k] = (
                    gamma[:, k] * np.sum(diff ** 2, axis=1)
                ).sum() / (Nk[k] * D)

    # ========================
    # Likelihood
    # ========================
    def _log_likelihood(self, X):
        N, D = X.shape
        likelihood = np.zeros((N, self.K))

        for k in range(self.K):
            if self.covariance_type == "full":
                cov = self.sigma[k]
            elif self.covariance_type == "tied":
                cov = self.sigma
            elif self.covariance_type == "diag":
                cov = np.diag(self.sigma[k])
            elif self.covariance_type == "spherical":
                cov = np.eye(D) * self.sigma[k]

            likelihood[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], cov)

        return np.sum(np.log(likelihood.sum(axis=1)))

    # ========================
    # Public API
    # ========================
    def fit(self, X):
        self._initialize_parameters(X)

        for i in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)

            ll = self._log_likelihood(X)
            self.log_likelihoods.append(ll)

            if i > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X):
        return self._e_step(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        return self._log_likelihood(X)
