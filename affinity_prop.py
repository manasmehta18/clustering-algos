from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import AffinityPropagation
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics

digits = load_digits()

X = digits.data
y = digits.target

n_samples, n_features = X.shape

np.random.seed(0)

# Compute Affinity Propagation
clustering = AffinityPropagation(preference=None, damping=0.99175).fit(X)
cluster_centers_indices = clustering.cluster_centers_indices_
labels = clustering.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

print confusion_matrix(y, clustering.labels_)

print metrics.fowlkes_mallows_score(y, clustering.labels_)
