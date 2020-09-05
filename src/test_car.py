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
import sso

X, y = preprocessing.vehicle_audio_percus()

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4)

"""
### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer(start=-7*480000)),
    ("grassmann", arma.GrassmannSignal(hidden_dim=2, truncate=10)),
    ("hclf", hermite_classifier.HermiteClassifier(alpha=1, n=32, q=3)),
    ("disc", discretiser.Discretiser())
])

cv_hc = sklearn.model_selection.cross_validate(pipe_hc, X, y, n_jobs=-1, verbose=2)
print(cv_hc["test_score"])
"""

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    # ("trimmer", preprocessing.Trimmer(start=-336000)),
    ("grassmann", arma.GrassmannSignal(hidden_dim=2, truncate=10)),
    ("svm", manifold_svm.ManifoldSVM(kern_gamma=10))
])

cv = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1, verbose=2)
cv2 = sklearn.model_selection.cross_validate(pipe_svm[1:], X, y, n_jobs=-1, verbose=2)
print(cv["test_score"], cv["test_score"].mean())
print(cv2["test_score"], cv2["test_score"].mean())

### Hermite SVM

# proj = lambda X, Y : np.sqrt(metrics.projection_metric_sq(X, Y))
# arc = lambda X, Y : np.sqrt(metrics.arc_length_sq(X, Y))
# 
# pipe_hsvm = sklearn.pipeline.Pipeline([
#     ("grassmann", arma.GrassmannSignal(hidden_dim=2, truncate=10)),
#     ("hsvm", manifold_svm.HermiteSVM(n=8, q=1, kern_metric=proj))
# ])

# grid_params = {
#     "hsvm__n": np.arange(3) * 4 + 8,
#     "hsvm__q": np.arange(5) * 2 + 1,
#     "hsvm__kern_metric": [proj, arc, None],
#     "grassmann__truncate": np.arange(10) + 1
# }

# cv_hsvm = sklearn.model_selection.cross_validate(pipe_hsvm, X, y, n_jobs=-1, verbose=2, cv=8)
# print(cv_hsvm["test_score"], cv_hsvm["test_score"].mean())

# clf = sklearn.model_selection.GridSearchCV(pipe_hsvm, grid_params, n_jobs=-1, verbose=2, cv=3)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)
