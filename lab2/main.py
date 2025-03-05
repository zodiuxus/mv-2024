import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import hsv_to_rgb

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def create_clusters(max_iter, K):
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0)

    n_samples, n_features = data.shape

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    centroids = data[np.random.choice(n_samples, K, replace=False)]
    for iteration in range(max_iterations):
        labels = []
        for point in data:
            distances = euclidean_distance(point, centroids)
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(K)])

        if np.all(centroids == new_centroids):
            break

    return new_centroids, data, labels

max_iterations = 100
K = 5

colors_rgb = [tuple(np.random.random(3)) for _ in range(K)]
hsv_colors = [hsv_to_rgb((np.random.random(), 1, 1)) for _ in range(K)]
lab_colors = [(np.random.random(), 0.5, 0.5) for _ in range(K)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

rgb_centroids, rgb_data, rgb_labels = create_clusters(max_iterations, K) 
axes[0].set_title("K-means Clustering in RGB")
for cluster_idx in range(K):
    cluster_points = rgb_data[rgb_labels == cluster_idx]
    axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors_rgb[cluster_idx], label=f"Cluster {cluster_idx+1}")
axes[0].scatter(rgb_centroids[:, 0], rgb_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
axes[0].legend()

hsv_centroids, hsv_data, hsv_labels = create_clusters(max_iterations, K)
axes[1].set_title("K-means Clustering in HSV")
for cluster_idx in range(K):
    cluster_points = hsv_data[hsv_labels == cluster_idx]
    axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], color=hsv_colors[cluster_idx], label=f"Cluster {cluster_idx+1}")
axes[1].scatter(hsv_centroids[:, 0], hsv_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
axes[1].legend()

lab_centroids, lab_data, lab_labels = create_clusters(max_iterations, K)
axes[2].set_title("K-means Clustering in Lab")
for cluster_idx in range(K):
    cluster_points = lab_data[lab_labels == cluster_idx]
    axes[2].scatter(cluster_points[:, 0], cluster_points[:, 1], color=lab_colors[cluster_idx], label=f"Cluster {cluster_idx+1}")
axes[2].scatter(lab_centroids[:, 0], lab_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
axes[2].legend()

plt.tight_layout()
plt.show()
