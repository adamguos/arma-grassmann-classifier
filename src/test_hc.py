import os
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection, pipeline, preprocessing
import time

import arma
import hermite
from hermite_classifier import HermiteClassifier
import metrics

"""
data = []
labels = []
raw = loadmat("../data/videoclipdata_naoki/LipData.mat")["data"][0, 0]

for row in raw:
    data.append(row)

for d in range(5):
    for i in range(10):
        labels.append(d + 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.4,
        random_state=0)
"""

"""
X_train = np.load("../data/eeg_irvine/X_train.npy")
X_test = np.load("../data/eeg_irvine/X_test.npy")
y_train = np.load("../data/eeg_irvine/y_train.npy")
y_test = np.load("../data/eeg_irvine/y_test.npy")
"""

X = np.load("../data/audio_data_percus/X.npy", allow_pickle=True)
y = np.load("../data/audio_data_percus/y.npy")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4,
        random_state=0)

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

pipe_hc = pipeline.Pipeline([
    ("grassmann", arma.GrassmannSignal(hidden_dim=1)),
    ("hclf", HermiteClassifier())
])
pipe_hc.fit(X_train, y_train)
pred = pipe_hc.predict(X_test)
print(sum(pred == y_test) / len(pred))

"""
frob = lambda X, Y : np.linalg.norm(X - Y)
flatten = lambda X, Y : np.linalg.norm(X.flatten() - Y.flatten())
proj = lambda X, Y : np.sqrt(metrics.projection_metric_sq(X, Y))
arc = lambda X, Y : np.sqrt(metrics.arc_length_sq(X, Y))
grid_params = {
    # "hclf__metric": [frob, flatten, proj, arc]
    "hclf__q": np.arange(31) * 2 + 2
}

clf = model_selection.GridSearchCV(pipe_hc, grid_params, verbose=2, n_jobs=os.cpu_count())
clf.fit(np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))
print("Best score:", clf.best_score_)
print("Best params:", clf.best_params_)
"""

"""
grid_params = {
    "hclf__alpha": np.linspace(0, 5, 11),
    "hclf__n": np.linspace(2, 8, 4),
    "hclf__q": np.linspace(3, 10, 8)
}

clf = model_selection.GridSearchCV(pipe_hc, grid_params, verbose=2, n_jobs=os.cpu_count())
clf.fit(np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))
print("Best score:", clf.best_score_)
print("Best params:", clf.best_params_)
"""

"""
X = np.random.random((100, 2)) * 2 - 1
y = np.sign(X[:, 0])

pipe_hc = pipeline.Pipeline([
    ("hermite_classifier", HermiteClassifier())
])
grid_params = {
    "hermite_classifier__alpha": np.linspace(0, 5, 11),
    "hermite_classifier__n": np.linspace(2, 8, 4),
    "hermite_classifier__q": np.linspace(3, 10, 8)
}

clf = model_selection.GridSearchCV(pipe_hc, grid_params, verbose=2, n_jobs=os.cpu_count())
start = time.time()
clf.fit(X, y)
print("\nTotal time elapsed:", time.time() - start)
print("Best score:", clf.best_score_)
print("Best params:", clf.best_params_)
"""
