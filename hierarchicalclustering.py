
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
X = iris.data[:, :3]


plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', s=50)
plt.title("Iris Dataset (First Two Features)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

print("Agglomerative (Bottom-Up) Clustering:")


linked = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Agglomerative Hierarchical Clustering Dendrogram")
plt.show()


agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(X)


plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=agg_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("Agglomerative Clustering (Bottom-Up)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


print("Divisive (Top-Down) Clustering:")

div_linked = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(div_linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Divisive Hierarchical Clustering Dendrogram")
plt.show()
