import numpy as np
import os
import pdb
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import discretiser
import hermite_classifier
import manifold_svm
import preprocessing
import signal_transform

X_train, X_test, y_train, y_test = preprocessing.eeg_irvine()

le = sklearn.preprocessing.LabelEncoder()
le.fit(np.concatenate((y_train, y_test)))
y_train = le.transform(y_train)
y_test = le.transform(y_test)

X = np.concatenate((X_train, X_test), 0)
y = np.concatenate((y_train, y_test))

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

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal(hidden_dim=12)),
    ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
])

pipe_svm.fit(X_train, y_train)
print(pipe_svm.score(X_test, y_test))

cv = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1)
print(cv["test_score"])
