import numpy as np

# Define data points
M1 = np.array([1, 4.5])
M2 = np.array([3, 6.5])
M3 = np.array([4, 4.5])
M4 = np.array([7.5, 3.2])
M5 = np.array([6, 2.3])
M6 = np.array([2.5, 3.8])
M7 = np.array([5, 5.5])

# Define cluster centroids
C1 = np.array([3, 4])
C2 = np.array([6, 4])

# Perform K-Means clustering
def k_means(data_points, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data_points:
        distances = [np.linalg.norm(point - c) for c in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
    return clusters

clusters = k_means([M1, M2, M3, M4, M5, M6, M7], [C1, C2])

# Print cluster members
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")
