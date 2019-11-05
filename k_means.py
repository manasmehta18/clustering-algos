from sklearn.datasets import load_digits
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import itertools as it
from sklearn.metrics import confusion_matrix
from sklearn import metrics

digits = load_digits()

X = digits.data
y = digits.target

n_samples, n_features = X.shape

np.random.seed(0)

clustering = KMeans(n_clusters=10, random_state=0).fit(X)

# for pred_label, act_label in it.izip(clustering.labels_, y):
#     print pred_label-act_label

data = pd.DataFrame(clustering.labels_)

print confusion_matrix(y, clustering.labels_)

print metrics.fowlkes_mallows_score(y, clustering.labels_)