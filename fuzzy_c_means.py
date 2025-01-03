

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x = np.array([1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.8])
y = np.array([1.0, 2.0, 4.0, 7.0, 5.0, 5.5, 6.0])
data = np.vstack((x, y))
n_clusters = 3


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, n_clusters, m=2.0, error=0.005, maxiter=1000, init=None
)

cluster_membership = np.argmax(u, axis=0)
for i in range(n_clusters):
    plt.scatter(x[cluster_membership == i], y[cluster_membership == i], label=f"Cluster {i + 1}")

plt.scatter(cntr[:, 0], cntr[:, 1], c="red", marker="x", s=200, label="Centroids")
plt.legend()
plt.title("Fuzzy C-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

!pip install scikit-fuzzy
