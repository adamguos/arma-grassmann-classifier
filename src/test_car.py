import numpy as np
import os
import pdb
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import discretiser
import hermite_classifier
import manifold_svm
import preprocessing

X, y = preprocessing.vehicle_audio_percus()

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4)

### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal()),
    ("hclf", hermite_classifier.HermiteClassifier()),
    ("disc", discretiser.Discretiser())
])

pipe_hc.set_params(grassmann__hidden_dim=2, hclf__alpha=0.0001, hclf__n=2, hclf__q=2,
        trimmer__start=-7*48000)
cv_hc = sklearn.model_selection.cross_validate(pipe_hc, X, y, n_jobs=-1)

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal()),
    ("svm", manifold_svm.ManifoldSVM())
])

pipe_svm.set_params(grassmann__hidden_dim=2, svm__kern_gamma=0.01, trimmer__start=-6*48000)
cv_svm = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1)

### results

print("hc:", cv_hc["test_score"])
print("svm:", cv_svm["test_score"])
