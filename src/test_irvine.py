import numpy as np
import os
import pdb
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import discretiser
import hermite_classifier
import manifold_svm
import metrics
import preprocessing
import signal_transform

ifreq = False

X_train, X_test, y_train, y_test = preprocessing.eeg_irvine(ifreq)

if X_train.ndim == 4:
    X_train = X_train[:, :, :, 0]
    X_test = X_test[:, :, :, 0]

le = sklearn.preprocessing.LabelEncoder()
le.fit(np.concatenate((y_train, y_test)))
y_train = le.transform(y_train)
y_test = le.transform(y_test)

X = np.concatenate((X_train, X_test), 0)
y = np.concatenate((y_train, y_test))

import matplotlib.pyplot as plt
for i in range(X_train.shape[2]):
    plt.plot(np.linspace(0, 1, X_train.shape[1]), X_train[0, :, i], linewidth=1)
plt.show()

"""

### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("dct", signal_transform.DCT()),
    ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
    ("hclf", hermite_classifier.HermiteClassifier(alpha=1, n=8, q=10)),
    ("disc", discretiser.Discretiser())
])

pipe_hc.fit(X_train, y_train)
print("hc:", pipe_hc.score(X_test, y_test))

no_dct = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
    ("hclf", hermite_classifier.HermiteClassifier(alpha=1, n=8, q=10)),
    ("disc", discretiser.Discretiser())
])

no_dct.fit(X_train, y_train)
print("no dct:", no_dct.score(X_test, y_test))

"""

### SVM

# no ifreq
pipe_svm = sklearn.pipeline.Pipeline([
    ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
    ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
])

# ifreq
# pipe_svm = sklearn.pipeline.Pipeline([
#     ("grassmann", arma.GrassmannSignal(hidden_dim=9)),
#     ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
# ])

# grid_params = {
#     "grassmann__hidden_dim": np.arange(6) + 5,
#     "svm__kern_gamma": np.logspace(-2, 1, 6)
# }

# pipe_svm.fit(X_train, y_train)
# print(pipe_svm.score(X_test, y_test))

# ps = sklearn.model_selection.PredefinedSplit(([-1] * len(X_train)) + ([1] * len(X_test)))
kfold = sklearn.model_selection.RepeatedKFold(n_splits=3, n_repeats=5)
cv = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1, cv=kfold)
print(cv["test_score"])

# clf = sklearn.model_selection.GridSearchCV(pipe_svm, grid_params, cv=ps, n_jobs=-1, verbose=2)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)

### Hermite SVM

# proj = lambda X, Y : np.sqrt(metrics.projection_metric_sq(X, Y))
# arc = lambda X, Y : np.sqrt(metrics.arc_length_sq(X, Y))
# 
# if not ifreq:
#     pipe_hsvm = sklearn.pipeline.Pipeline([
#         ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
#         ("hsvm", manifold_svm.HermiteSVM(n=0, q=10, kern_metric=proj))
#     ])
# else:
#     pipe_hsvm = sklearn.pipeline.Pipeline([
#         ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
#         ("hsvm", manifold_svm.HermiteSVM(n=0, q=10, kern_metric=proj))
#     ])
# 
# grid_params = {
#     "hsvm__n": np.arange(3) * 4 + 8,
#     "hsvm__q": np.arange(5) * 2 + 1,
#     # "hsvm__kern_metric": [proj, arc, None],
# }
# 
# cv_hsvm = sklearn.model_selection.cross_validate(pipe_hsvm, X, y, n_jobs=-1, verbose=2)
# print(cv_hsvm["test_score"], cv_hsvm["test_score"].mean())

# clf = sklearn.model_selection.GridSearchCV(pipe_hsvm, grid_params, n_jobs=-1, verbose=2, cv=3)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)
