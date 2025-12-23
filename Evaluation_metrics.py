import numpy as np

class EvaluationMetrics:
    
    @staticmethod
    def silhouette_score(X, labels):
        """
        Ranges from -1 to 1. 
        High value: Clusters are well apart.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0

        n = X.shape[0]
        silhouette_vals = np.zeros(n)

        # Pre-calculate distances could be faster, but loop is acceptable for N=569
        for i in range(n):
            current_label = labels[i]
            
            # Calculate a(i): Mean distance to own cluster
            mask_same = (labels == current_label)
            # Remove self from the calculation
            mask_same[i] = False 
            
            same_cluster = X[mask_same]
            
            if len(same_cluster) == 0:
                a = 0
            else:
                a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))

            # Calculate b(i): Mean distance to nearest other cluster
            b = np.inf
            for label in unique_labels:
                if label == current_label:
                    continue
                
                other_cluster = X[labels == label]
                if len(other_cluster) > 0:
                    dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                    b = min(b, dist)
            
            # If b is still inf (only 1 cluster), score is 0
            if b == np.inf:
                silhouette_vals[i] = 0.0
            else:
                silhouette_vals[i] = (b - a) / max(a, b)

        return np.mean(silhouette_vals)

    @staticmethod
    def davies_bouldin(X, labels):
        """
        Lower is better. Measures cluster similarity.
        """
        unique_labels = np.unique(labels)
        # Handle outliers (-1) if any, though standard KMeans won't produce them
        unique_labels = unique_labels[unique_labels != -1]
        
        n_clusters = len(unique_labels)
        if n_clusters < 2:
            return 0.0

        # 1. Compute Centroids
        centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        
        # 2. Compute average distance inside each cluster (S_i)
        S = np.zeros(n_clusters)
        for i, k in enumerate(unique_labels):
            cluster_points = X[labels == k]
            S[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

        # 3. Compute Separation (R_ij)
        R = np.zeros(n_clusters)
        for i in range(n_clusters):
            max_val = -np.inf
            for j in range(n_clusters):
                if i != j:
                    dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                    if dist_centroids == 0:
                        val = 0
                    else:
                        val = (S[i] + S[j]) / dist_centroids
                    
                    if val > max_val:
                        max_val = val
            R[i] = max_val

        return np.mean(R)

    @staticmethod
    def calinski_harabasz(X, labels):
        """
        Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion.
        """
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0

        overall_mean = np.mean(X, axis=0)
        
        # Within-cluster scatter (SSW) and Between-cluster scatter (SSB)
        SSW = 0.0
        SSB = 0.0
        
        for k in unique_labels:
            cluster_k = X[labels == k]
            centroid_k = cluster_k.mean(axis=0)
            
            # SSB: Weight * dist(centroid, global_mean)^2
            SSB += len(cluster_k) * np.sum((centroid_k - overall_mean) ** 2)
            
            # SSW: dist(points, centroid)^2
            SSW += np.sum((cluster_k - centroid_k) ** 2)

        if SSW == 0:
            return np.inf

        return (SSB / (n_clusters - 1)) / (SSW / (n_samples - n_clusters))

    @staticmethod
    def purity(y_true, y_pred):
        """
        Higher is better. 
        """
        # Fix: Use np.unique to handle string labels or non-consecutive integers
        total_intersection = 0
        for k in np.unique(y_pred):
            # Get true labels for points in this cluster
            labels_in_cluster = y_true[y_pred == k]
            
            if len(labels_in_cluster) == 0:
                continue
                
            # Find most frequent label
            unique, counts = np.unique(labels_in_cluster, return_counts=True)
            total_intersection += counts.max()
            
        return total_intersection / len(y_true)

    @staticmethod
    def adjusted_rand_index(labels_true, labels_pred):
        """
        Measures similarity between two clusterings. (0 = random, 1 = perfect)
        """
        # Create contingency table (confusion matrix)
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        
        n_classes = len(classes)
        n_clusters = len(clusters)
        
        contingency_matrix = np.zeros((n_classes, n_clusters))
        
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(clusters):
                # Count intersection
                contingency_matrix[i, j] = np.sum((labels_true == true_cls) & (labels_pred == pred_cls))
        
        # Calculate sums
        sum_rows = np.sum(contingency_matrix, axis=1) # a_i
        sum_cols = np.sum(contingency_matrix, axis=0) # b_j
        n = len(labels_true)
        
        # Helper for "n choose 2" = n*(n-1)/2
        def binom2(x):
            return x * (x - 1) / 2
            
        sum_nij_binom = np.sum(binom2(contingency_matrix))
        sum_a_binom = np.sum(binom2(sum_rows))
        sum_b_binom = np.sum(binom2(sum_cols))
        n_binom = binom2(n)
        
        # ARI Formula
        index = sum_nij_binom
        expected_index = (sum_a_binom * sum_b_binom) / n_binom
        max_index = (sum_a_binom + sum_b_binom) / 2
        
        if max_index == expected_index:
            return 0.0
            
        return (index - expected_index) / (max_index - expected_index)

    @staticmethod
    def normalized_mutual_info(labels_true, labels_pred):
        """
        Measures agreement between two clusterings.
        """
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        n = len(labels_true)
        
        contingency_matrix = np.zeros((len(classes), len(clusters)))
        for i, cls in enumerate(classes):
            for j, clus in enumerate(clusters):
                contingency_matrix[i, j] = np.sum((labels_true == cls) & (labels_pred == clus))

        # 1. Mutual Information
        mi = 0.0
        eps = 1e-10 # epsilon to avoid log(0)
        
        for i in range(len(classes)):
            for j in range(len(clusters)):
                nij = contingency_matrix[i, j]
                if nij > 0:
                    ni = np.sum(contingency_matrix[i, :])
                    nj = np.sum(contingency_matrix[:, j])
                    mi += (nij / n) * np.log((nij * n) / (ni * nj) + eps)

        # 2. Entropy
        h_true = 0.0
        for i in range(len(classes)):
            ni = np.sum(contingency_matrix[i, :])
            if ni > 0:
                h_true -= (ni / n) * np.log(ni / n + eps)
                
        h_pred = 0.0
        for j in range(len(clusters)):
            nj = np.sum(contingency_matrix[:, j])
            if nj > 0:
                h_pred -= (nj / n) * np.log(nj / n + eps)

        # 3. Normalized MI (Arithmetic Mean)
        if (h_true + h_pred) == 0:
            return 0.0
            
        return 2 * mi / (h_true + h_pred)
