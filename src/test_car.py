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

pipe_hc.set_params(grassmann__hidden_dim=1, hclf__alpha=0.0001, hclf__n=2, hclf__q=2,
        trimmer__start=-7*48000)

grid_params = {
    "trimmer__start": -(np.arange(5) + 6) * 48000,
    "grassmann__hidden_dim": np.array([1, 2]),
    "hclf__n": (np.arange(4) + 1) * 2,
    "hclf__q": (np.arange(5) + 1) * 2,
    "hclf__alpha": np.logspace(-4, 0, 5)
}

clf_hc = sklearn.model_selection.GridSearchCV(pipe_hc, grid_params, verbose=2,
        n_jobs=os.cpu_count())
clf_hc.fit(X, y)

### SVM

pipe_svm = sklearn.pipeline.Pipeline([
    ("trimmer", preprocessing.Trimmer()),
    ("grassmann", arma.GrassmannSignal()),
    ("svm", manifold_svm.ManifoldSVM())
])

grid_params = {
    "trimmer__start": -(np.arange(5) + 6) * 48000,
    "grassmann__hidden_dim": np.array([1, 2]),
    "svm__kern_gamma": np.logspace(-2, 0.5, 5)
}

clf_svm = sklearn.model_selection.GridSearchCV(pipe_svm, grid_params, verbose=2,
        n_jobs=os.cpu_count())
clf_svm.fit(X, y)

### results

print("hc best score:", clf_hc.best_score_)
print("hc best params:", clf_hc.best_params_)
print("svm best score:", clf_svm.best_score_)
print("svm best params:", clf_svm.best_params_)
