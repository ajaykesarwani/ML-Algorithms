import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None  # centroids (set after fit)
        self.labels_ = None           # cluster assignments (set after fit)
        self.inertia_ = None          # sum of squared distances to centroids (set after fit)

    def fit(self, X):
        """
        Fit the K-Means model to the data.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Seed the random number generator for reproduucibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialization: Choosing K initial cluster centroids.
        initial_centroids = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[initial_centroids].copy()

        for i in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()

            # 2. Assignment Step: Assign each data point to the cluster whose centroid is closest
            self.labels_ = self.predict(X)

            # 3. Update Step: Recalculate the centroids of the K clusters by taking the mean of all data points assigned to each respective cluster.
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.any(mask):
                    self.cluster_centers_[k] = X[mask].mean(axis=0)
                else:
                    # If a cluster is empty then re-initialize it to a random point
                    self.cluster_centers_[k] = X[np.random.choice(n_samples)]

            # Convergence Check: Repeat steps 2 and 3 until the centroids no longer change significantly between iterations, or a maximum number of iterations is reached.
            center_shift = np.linalg.norm(self.cluster_centers_ - old_centers)
            if center_shift < self.tol:
                break

        # Calculate Inertia i.e. Within-cluster sum of squares (WCSS)
        self.inertia_ = self._calculate_inertia(X)
        return self

    def _calculate_inertia(self, X):
        # Objective Function: Within-Cluster Sum of Squares (WCSS)
        # WCSS is the sum of the squared distances between each data point and its assigned cluster centroid.
        assigned_centroid = self.cluster_centers_[self.labels_]
        return np.sum((X - assigned_centroid) ** 2)

    def predict(self, X):
        """
        Predict cluster labels for new data points.
        """
        X = np.array(X)
        # Using Euclidean distance to find the closest cluster center
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

class KMeansPlusPlus(KMeans):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(n_clusters, max_iter, tol, random_state)
        
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Seed the random number generator for reproduucibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Choose the first centroid: Select one data point uniformly at random from the dataset to be the first centroid
        centers = [X[np.random.choice(n_samples)]]

        # 2. Picking the remaining k-1 centers
        for _ in range(1, self.n_clusters):
            # Calculate the squared distance from each point to the nearest already chosen center
            # D(x)^2
            dist_sq = np.array([min([np.sum((x - c)**2) for c in centers]) for x in X])
            
            # 3. Choose the next center with probability proportional to D(x)^2
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            
            # Find index where cumulative probability exceeds random value r
            next_idx = np.searchsorted(cumulative_probs, r)
            centers.append(X[next_idx])

        self.cluster_centers_ = np.array(centers)

        # Now that centers are initialized, run the standard K-Means optimization
        # (This avoids code duplication)
        return self._run_kmeans_logic(X)

    def _run_kmeans_logic(self, X):
        """Standard K-Means iterations after initialization."""
        n_samples = X.shape[0]
        for i in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            
            self.labels_ = self.predict(X)
            
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.any(mask):
                    self.cluster_centers_[k] = X[mask].mean(axis=0)
            
            if np.linalg.norm(self.cluster_centers_ - old_centers) < self.tol:
                break
                
        self.inertia_ = self._calculate_inertia(X)
        return self