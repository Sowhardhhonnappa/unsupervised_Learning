

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import numpy as np


iris = load_iris()
X = iris.data
y_true = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_test)

rand_index = rand_score(y_test, y_pred)
print(f"Rand Index (RI): {rand_index:.4f}")

adjusted_rand_index = adjusted_rand_score(y_test, y_pred)
print(f"Adjusted Rand Index (ARI): {adjusted_rand_index:.4f}")
print(" - Rand Index measures agreement between true labels and clustering.")
print(" - Adjusted Rand Index accounts for chance clustering, making it a better metric.")
