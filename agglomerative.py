from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import itertools as it
from sklearn.metrics import confusion_matrix
from sklearn import metrics

digits = load_digits()

X = digits.data
y = digits.target

n_samples, n_features = X.shape

np.random.seed(0)

clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=10)
clustering.fit(X)

# for pred_label, act_label in it.izip(clustering.labels_, y):
#     print pred_label-act_label

data = pd.DataFrame(clustering.labels_)

print confusion_matrix(y, clustering.labels_)

print metrics.fowlkes_mallows_score(y, clustering.labels_)


