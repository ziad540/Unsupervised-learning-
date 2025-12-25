import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class StandardScaler:
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
        
        # Initialize means using random samples from X
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # initialize covariances by adding regularization to variance of data to avoid singularities
        variance = np.var(X, axis=0) + self.reg_covar
        if self.covariance_type == 'full': # Clusters can be stretched differently in every direction.
            self.covariances_ = np.array([np.diag(variance) for _ in range(self.n_components)]) # diagonal covariance matrices
        elif self.covariance_type == 'tied': # All clusters share the same covariance matrix.
            self.covariances_ = np.diag(variance) # single shared covariance matrix for all clusters
        elif self.covariance_type == 'diag': # Clusters have different variances along each dimension, but no covariance.(no rotation)
            self.covariances_ = np.array([variance.copy() for _ in range(self.n_components)]) # variance of each feature
        elif self.covariance_type == 'spherical': # Clusters have the same variance in all directions (perfectly round).
            self.covariances_ = np.array([np.mean(variance) for _ in range(self.n_components)]) # single variance value per cluster

      # calculate log(N(xi; mu_k, Sigma_k))
    def _estimate_log_gaussian_prob(self, X):
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        log_2pi = n_features * np.log(2 * np.pi) #Dln(2pi)
        
        #measures how close a data point is to the mean, adjusted for the shape of the distribution
        for k in range(self.n_components):
            mu = self.means_[k]
            if self.covariance_type == 'full':
                cov = self.covariances_[k] + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov) #instead of inverting covariance matrix, use Cholesky decomposition for numerical stability
                # cov = L L^T
                #ln(cov) = 2* sum(ln(diagonal elements of L))
                log_det = 2 * np.sum(np.log(np.diagonal(L)))
                #mahalanobis distance: (x-mu)^T cov^-1 (x-mu) = || L^-1 (x-mu) ||^2
                # z= L^-1 (x-mu)
                z = np.linalg.solve(L, (X - mu).T).T # compute L^-1 (x-mu) without inverting L
                mahalanobis = np.sum(z**2, axis=1)
            elif self.covariance_type == 'tied': #same as full but shared covariance
                cov = self.covariances_ + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diagonal(L)))
                z = np.linalg.solve(L, (X - mu).T).T
                mahalanobis = np.sum(z**2, axis=1)
            elif self.covariance_type == 'diag': #covariance is a vector of variances
                cov = self.covariances_[k] + self.reg_covar
                log_det = np.sum(np.log(cov)) # det here is just product of its diagonal elements
                mahalanobis = np.sum(((X - mu)**2) / cov, axis=1) #as there are no correlations
            elif self.covariance_type == 'spherical': #covariance is a single variance value (variance same in all directions)
                cov = self.covariances_[k] + self.reg_covar
                log_det = n_features * np.log(cov) # det is cov^n_features
                mahalanobis = np.sum((X - mu)**2, axis=1) / cov #just standard euclidean distance scaled by variance
                
            log_prob[:, k] = -0.5 * (log_2pi + log_det + mahalanobis)
        return log_prob

    # r_ik = P(z_i=k | x_i) = P(x_i|z_i=k)P(z_i=k) / P(x_i)
    # we perform it here but in log space for numerical stability
    # numerator: log prop(likelihood) + log weights(Prior)
    # Denominator: logsumexp(numerator) across all components (to normalize)
    def _e_step(self, X):
        log_prob = self._estimate_log_gaussian_prob(X)
        # add log of the mixing weights
        weighted_log_prob = log_prob + np.log(self.weights_ + 1e-20)
        # normalize to get responsibilities
        log_likelihood_samples = logsumexp(weighted_log_prob, axis=1) # calculates denominator safely
        log_resp = weighted_log_prob - log_likelihood_samples[:, np.newaxis]
        return np.exp(log_resp), np.sum(log_likelihood_samples) # return responsibilities and total log likelihood

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        
        # Clip responsibilities to avoid tiny values creating NaNs during division
        resp = np.clip(resp, 1e-15, 1)
        nk = np.sum(resp, axis=0)
        # piK= (1/n_samples) * sum(resp_ik)
        self.weights_ = nk / n_samples
        # muK = (1/sum(resp_ik)) * sum(resp_ik * xi)
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                # 1/nk * sum(resp_ik * (xi - muK)(xi - muK)^T)
                self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                self.covariances_[k].flat[::n_features + 1] += self.reg_covar # add regularization to diagonal
        elif self.covariance_type == 'tied':
            total_cov = np.zeros((n_features, n_features))
            # one single averaged covariance matrix for all clusters
            # sum(sum(resp_ik * (xi - muK)(xi - muK)^T)) / n_samples
            for k in range(self.n_components):
                diff = X - self.means_[k]
                total_cov += np.dot(resp[:, k] * diff.T, diff)
            self.covariances_ = total_cov / n_samples
            self.covariances_.flat[::n_features + 1] += self.reg_covar
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff_sq = (X - self.means_[k])**2 #just variance along each dimension
                self.covariances_[k] = np.sum(resp[:, k, np.newaxis] * diff_sq, axis=0) / nk[k]
                self.covariances_[k] += self.reg_covar
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff_sq = np.sum((X - self.means_[k])**2, axis=1) 
                # one single number (scalar) represent radius of cluster
                self.covariances_[k] = np.sum(resp[:, k] * diff_sq) / (nk[k] * n_features)
                self.covariances_[k] += self.reg_covar

    def fit(self, X):
        self._initialize_parameters(X)
        prev_ll = -np.inf
        for i in range(self.max_iter):
            resp, total_ll = self._e_step(X)
            current_avg_ll = total_ll / X.shape[0]
            self.log_likelihood_history_.append(current_avg_ll)
            
            if abs(total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll
            self._m_step(X, resp)
        return self

    ## Information Criteria for model selection, Add penalty for number of parameters to avoid overfitting
    # penality of complexity - goodness of fit
    # we want the lowest AIC/BIC , Higher is better for log likelihood
    def bic(self, X):
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        # unique values in a symmetric matrix= D(D+1)/2 , K*D(D+1)/2 for full covariance
        if self.covariance_type == 'full': cp = self.n_components * n_features * (n_features + 1) / 2 
        elif self.covariance_type == 'tied': cp = n_features * (n_features + 1) / 2 # 1*D(D+1)/2
        elif self.covariance_type == 'diag': cp = self.n_components * n_features # only diagonal D*K
        else: cp = self.n_components # only learn radius per cluster 1*K
        # weights + means + covariances
        # -1 bcs if we have k clusters we can know last one by summing to 1
        # Each cluster center has D coordinates (x, y, z...). We have K clusters. So K × D
        n_params = (self.n_components - 1) + (self.n_components * n_features) + cp
        return n_params * np.log(n_samples) - 2 * log_likelihood # penalty klgn, prefers simpler models (fewer clusters).

    def aic(self, X):
        n_samples, n_features = X.shape
        _, log_likelihood = self._e_step(X)
        if self.covariance_type == 'full': cp = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied': cp = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag': cp = self.n_components * n_features
        else: cp = self.n_components
        n_params = (self.n_components - 1) + (self.n_components * n_features) + cp
        return 2 * n_params - 2 * log_likelihood # penalty 2k, slightly more clusters



