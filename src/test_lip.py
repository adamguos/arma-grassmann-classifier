import numpy as np
import os
import pdb
import pprint
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import discretiser
import hermite_classifier
import manifold_svm
import metrics
import preprocessing
import signal_transform
import sso

X, y = preprocessing.lip_naoki()

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

"""

### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("dct", signal_transform.DCT()),
    ("grassmann", arma.GrassmannSignal()),
    ("hclf", hermite_classifier.HermiteClassifier(n=8)),
    ("disc", discretiser.Discretiser())
])

cv_hc = sklearn.model_selection.cross_validate(pipe_hc, X, y, n_jobs=-1)

"""

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
    ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
])

kfold = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=20)
cv = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1, verbose=2, cv=kfold)
print(cv["test_score"], cv["test_score"].mean())

### Hermite SVM

# proj = lambda X, Y : np.sqrt(metrics.projection_metric_sq(X, Y))
# arc = lambda X, Y : np.sqrt(metrics.arc_length_sq(X, Y))
# 
# pipe_hsvm = sklearn.pipeline.Pipeline([
#     ("grassmann", arma.GrassmannSignal(hidden_dim=5)),
#     ("hsvm", manifold_svm.HermiteSVM(n=1, q=1, kern_metric=proj))
# ])
# 
# results = []
# for i in np.arange(16) + 1:
#     pipe_hsvm.set_params(hsvm__n=i)
#     cv_hsvm = sklearn.model_selection.cross_validate(pipe_hsvm, X, y, n_jobs=-1, verbose=0)
#     print(i, cv_hsvm["test_score"], cv_hsvm["test_score"].mean())
#     results.append(cv_hsvm["test_score"].mean())

# grid_params = {
#     # "grassmann__hidden_dim": np.arange(6) + 5,
#     # "hsvm__n": np.arange(8) + 1,
#     "hsvm__q": np.arange(10) + 1,
#     # "hsvm__kern_metric": [proj, arc, None]
# }

# clf = sklearn.model_selection.GridSearchCV(pipe_hsvm, grid_params, n_jobs=-1, verbose=2)
# clf.fit(X, y)
# print(clf.best_score_)
# print(clf.best_params_)
# pp = pprint.PrettyPrinter()
# pp.pprint(clf.cv_results_)
