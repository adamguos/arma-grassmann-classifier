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

X, y = preprocessing.vehicle_audio_percus()

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4)

"""

### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("dct", signal_transform.DCT()),
    ("grassmann", arma.GrassmannSignal()),
    ("hclf", hermite_classifier.HermiteClassifier()),
    ("disc", discretiser.Discretiser())
])

pipe_hc.set_params(grassmann__hidden_dim=2, grassmann__truncate=10, hclf__alpha=1, hclf__n=16,
        hclf__q=3, trimmer__start=-7*48000)
cv_hc = sklearn.model_selection.cross_validate(pipe_hc, X, y, n_jobs=-1)

"""

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer(start=-336000)),
    ("grassmann", arma.GrassmannSignal(hidden_dim=2)),
    ("svm", manifold_svm.ManifoldSVM(kern_gamma=10))
])

grid_params = {
    "svm__kern_gamma": np.logspace(-3, 1, 8)
}

clf = sklearn.model_selection.GridSearchCV(pipe_svm, grid_params, verbose=2, n_jobs=-1)
clf.fit(X, y)
print("Best score:", clf.best_score_)
print("Best params:", clf.best_params_)
