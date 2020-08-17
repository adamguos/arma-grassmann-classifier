import numpy as np
import os
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import hermite_classifier
import preprocessing
import manifold_svm

X = np.load("../data/audio_data_percus/X.npy", allow_pickle=True)
y = np.load("../data/audio_data_percus/y.npy")

le = sklearn.preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

### Hermite classifier

pipe_hc = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal()),
    ("hclf", hermite_classifier.HermiteClassifier())
])

pipe_hc.set_params(grassmann__hidden_dim=1, hclf__alpha=0.0001, hclf__n=2, hclf__q=2)
cv_hc = sklearn.model_selection.cross_validate(pipe_hc, X, y, n_jobs=-1, cv=3)

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal()),
    ("svm", manifold_svm.ManifoldSVM())
])

pipe_svm.set_params(grassmann__hidden_dim=1, svm__kern_gamma=0.01, trimmer__start=-6*48000)
cv_svm = sklearn.model_selection.cross_validate(pipe_svm, X, y, n_jobs=-1, cv=3)

### results

print("hc:", cv_hc["test_score"])
print("svm:", cv_svm["test_score"])
