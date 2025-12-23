import numpy as np

class EvaluationMetrics:

    @staticmethod
    def silhouette_score(X, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return 0.0

        n = X.shape[0]
        score = 0.0

        for i in range(n):
            same_cluster = X[labels == labels[i]]
            same_cluster = same_cluster[same_cluster != X[i]].reshape(-1, X.shape[1])

            if len(same_cluster) == 0:
                continue

            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))

            b = np.inf
            for l in unique_labels:
                if l != labels[i]:
                    other_cluster = X[labels == l]
                    dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                    b = min(b, dist)

            score += (b - a) / max(a, b)

        return score / n

    @staticmethod
    def davies_bouldin(X, labels):
        clusters = np.unique(labels)
        centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])

        S = np.array([
            np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
            for i, c in enumerate(clusters)
        ])

        R = []
        for i in range(len(clusters)):
            r_ij = []
            for j in range(len(clusters)):
                if i != j:
                    r_ij.append((S[i] + S[j]) /
                                np.linalg.norm(centroids[i] - centroids[j]))
            R.append(max(r_ij))

        return np.mean(R)

    @staticmethod
    def purity(y_true, y_pred):
        total = 0
        for c in np.unique(y_pred):
            labels_in_cluster = y_true[y_pred == c]
            total += np.max(np.bincount(labels_in_cluster))
        return total / len(y_true)



