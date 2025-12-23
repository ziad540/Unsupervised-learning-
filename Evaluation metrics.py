import numpy as np
def silhouette_score(X, labels):
    n = len(X)
    score = 0

    for i in range(n):
        same = X[labels == labels[i]]
        other = [X[labels == l] for l in set(labels) if l != labels[i]]

        a = np.mean(np.linalg.norm(same - X[i], axis=1))
        b = min(np.mean(np.linalg.norm(o - X[i], axis=1)) for o in other)

        score += (b - a) / max(a, b)

    return score / n

def davies_bouldin(X, labels):
    clusters = np.unique(labels)
    centroids = np.array([X[labels==c].mean(axis=0) for c in clusters])

    S = np.array([
        np.mean(np.linalg.norm(X[labels==c] - centroids[i], axis=1))
        for i,c in enumerate(clusters)
    ])

    R = []
    for i in range(len(clusters)):
        r = []
        for j in range(len(clusters)):
            if i != j:
                r.append((S[i] + S[j]) / np.linalg.norm(centroids[i]-centroids[j]))
        R.append(max(r))
    return np.mean(R)

def purity(y_true, y_pred):
    total = 0
    for c in np.unique(y_pred):
        true_labels = y_true[y_pred == c]
        total += np.max(np.bincount(true_labels))
    return total / len(y_true)
