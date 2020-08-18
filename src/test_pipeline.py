import numpy as np
import numpy as np
import os
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing

import arma
import hermite_classifier
import preprocessing

X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.4,
        random_state=0)
pipe = pipeline.Pipeline([
    ("grassmann", arma.GrassmannSignal()),
    ("svm", manifold_svm.ManifoldSVM())
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

X_train = np.load("../data/eeg_irvine/X_train.npy")
X_test = np.load("../data/eeg_irvine/X_test.npy")
y_train = np.load("../data/eeg_irvine/y_train.npy")
y_test = np.load("../data/eeg_irvine/y_test.npy")
pipe.set_params(svm__kern_gamma=0.19306977288832497)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(pipe.get_params())

pdb.set_trace()

grid_params = {
    "svm__kern_gamma": np.logspace(-2, 1)
}
clf = model_selection.GridSearchCV(pipe, grid_params, verbose=2)
# clf.fit(X_train, y_train)
clf.fit(data, labels)
print("Best score:", clf.best_score_)
print("Best params:", clf.best_params_)
