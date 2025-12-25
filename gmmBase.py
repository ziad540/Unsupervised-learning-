import numpy as np
from scipy.special import logsumexp

class StandardScaler:
    """Standardizes data: mean=0, variance=1"""
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0 
        return (X - self.mean) / self.std

class GaussianMixtureModel:
    def __init__(self, n_components, covariance_type="full",
                 max_iter=100, tol=1e-3, reg_covar=1e-2, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        if random_state is not None:
            np.random.seed(random_state)
            
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_history_ = []

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        variance = np.var(X, axis=0) + self.reg_covar
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.diag(variance) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances_ = np.diag(variance)
        elif self.covariance_type == 'diag':
            self.covariances_ = np.array([variance.copy() for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.array([np.mean(variance) for _ in range(self.n_components)])

    def _estimate_log_gaussian_prob(self, X):
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        log_2pi = n_features * np.log(2 * np.pi)
        
        for k in range(self.n_components):
            mu = self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k] + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diagonal(L)))
                z = np.linalg.solve(L, (X - mu).T).T
                mahalanobis = np.sum(z**2, axis=1)
            elif self.covariance_type == 'tied':
                cov = self.covariances_ + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diagonal(L)))
                z = np.linalg.solve(L, (X - mu).T).T
                mahalanobis = np.sum(z**2, axis=1)
            elif self.covariance_type == 'diag':
                cov = self.covariances_[k] + self.reg_covar
                log_det = np.sum(np.log(cov))
                mahalanobis = np.sum(((X - mu)**2) / cov, axis=1)
            elif self.covariance_type == 'spherical':
                cov = self.covariances_[k] + self.reg_covar
                log_det = n_features * np.log(cov)
                mahalanobis = np.sum((X - mu)**2, axis=1) / cov
                
            log_prob[:, k] = -0.5 * (log_2pi + log_det + mahalanobis)
        return log_prob

    def _e_step(self, X):
        log_prob = self._estimate_log_gaussian_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-20)
        log_likelihood_samples = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_likelihood_samples[:, np.newaxis]
        return np.exp(log_resp), np.sum(log_likelihood_samples)

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        resp = np.clip(resp, 1e-15, 1)
        nk = np.sum(resp, axis=0)
        self.weights_ = nk / n_samples
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                self.covariances_[k].flat[::n_features + 1] += self.reg_covar
        elif self.covariance_type == 'tied':
            total_cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                total_cov += np.dot(resp[:, k] * diff.T, diff)
            self.covariances_ = total_cov / n_samples
            self.covariances_.flat[::n_features + 1] += self.reg_covar
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff_sq = (X - self.means_[k])**2
                self.covariances_[k] = np.sum(resp[:, k, np.newaxis] * diff_sq, axis=0) / nk[k]
                self.covariances_[k] += self.reg_covar
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff_sq = np.sum((X - self.means_[k])**2, axis=1)
                self.covariances_[k] = np.sum(resp[:, k] * diff_sq) / (nk[k] * n_features)
                self.covariances_[k] += self.reg_covar

    def fit(self, X):
        self._initialize_parameters(X)
        prev_ll = -np.inf
        self.log_likelihood_history_ = []
        for i in range(self.max_iter):
            resp, total_ll = self._e_step(X)
            self.log_likelihood_history_.append(total_ll / X.shape[0])
            if abs(total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll
            self._m_step(X, resp)
        return self

    def _count_params(self, n_features):
        if self.covariance_type == 'full': cp = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied': cp = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag': cp = self.n_components * n_features
        else: cp = self.n_components
        return (self.n_components - 1) + (self.n_components * n_features) + cp

    def bic(self, X):
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        n_params = self._count_params(n_features)
        return n_params * np.log(n_samples) - 2 * log_likelihood

    def aic(self, X):
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        n_params = self._count_params(n_features)
        return 2 * n_params - 2 * log_likelihood