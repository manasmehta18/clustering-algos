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

mapped = zip(clustering.labels_, y)

lst = [[(-1, -1)], [(-1, -1)], [(-1, -1)], [(-1, -1)], [(-1, -1)],
       [(-1, -1)], [(-1, -1)], [(-1, -1)], [(-1, -1)], [(-1, -1)], ]

for i in mapped:
    if i[0] == 0:
        lst[0].append(i)
    elif i[0] == 1:
        lst[1].append(i)
    elif i[0] == 2:
        lst[2].append(i)
    elif i[0] == 3:
        lst[3].append(i)
    elif i[0] == 4:
        lst[4].append(i)
    elif i[0] == 5:
        lst[5].append(i)
    elif i[0] == 6:
        lst[6].append(i)
    elif i[0] == 7:
        lst[7].append(i)
    elif i[0] == 8:
        lst[8].append(i)
    elif i[0] == 9:
        lst[9].append(i)
    else:
        lst[10].append(i)

print lst[9][2:]

print confusion_matrix(y, clustering.labels_)

print metrics.fowlkes_mallows_score(y, clustering.labels_)
