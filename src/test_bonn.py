import numpy as np
import os
import pdb
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import balance_classes
import discretiser
import hermite_classifier
import manifold_svm
import metrics
import preprocessing
import signal_transform

ifreq = False

X, y = preprocessing.eeg_bonn(ifreq)
X, y = balance_classes.oversample(X, y)
X = np.expand_dims(X, 2) if not ifreq else X

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4)

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

# if not ifreq:
#     pipe_svm = sklearn.pipeline.Pipeline([
#         ("trimmer", preprocessing.Trimmer()),
#         ("grassmann", arma.GrassmannSignal(hidden_dim=1, truncate=10)),
#         ("svm", manifold_svm.ManifoldSVM(kern_gamma=10))
#     ])
# else:
#     pipe_svm = sklearn.pipeline.Pipeline([
#         ("trimmer", preprocessing.Trimmer()),
#         ("grassmann", arma.GrassmannSignal(hidden_dim=1, truncate=10)),
#         ("svm", manifold_svm.ManifoldSVM(kern_gamma=10))
#     ])
 
# grid_params = {
#     "svm__kern_gamma": np.logspace(-2, 1, 10),
#     "grassmann__hidden_dim": np.arange(4) + 1
# }
 
# clf = sklearn.model_selection.GridSearchCV(pipe_svm, grid_params, n_jobs=-1, verbose=2, cv=3)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)

# pipe_svm.fit(X_train, y_train)
# print(pipe_svm.score(X_test, y_test))

# cv = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1)
# print(cv["test_score"], cv["test_score"].mean())

### Hermite SVM

proj = lambda X, Y : np.sqrt(metrics.projection_metric_sq(X, Y))
arc = lambda X, Y : np.sqrt(metrics.arc_length_sq(X, Y))

if not ifreq:
    pipe_hsvm = sklearn.pipeline.Pipeline([
        ("grassmann", arma.GrassmannSignal(hidden_dim=1, truncate=10)),
        ("hsvm", manifold_svm.HermiteSVM(n=8, q=1, kern_metric=proj))
    ])
else:
    pipe_hsvm = sklearn.pipeline.Pipeline([
        ("grassmann", arma.GrassmannSignal(hidden_dim=1, truncate=10)),
        ("hsvm", manifold_svm.HermiteSVM(n=8, q=1, kern_metric=proj))
    ])

grid_params = {
    "hsvm__n": np.arange(3) * 4 + 8,
    "hsvm__q": np.arange(5) * 2 + 1,
    "hsvm__kern_metric": [proj, arc, None],
}

cv_hsvm = sklearn.model_selection.cross_validate(pipe_hsvm, X, y, n_jobs=-1, verbose=2, cv=8)
print(cv_hsvm["test_score"], cv_hsvm["test_score"].mean())

# clf = sklearn.model_selection.GridSearchCV(pipe_hsvm, grid_params, n_jobs=-1, verbose=2, cv=3)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)
