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

tests = [
    # "alc_euclidean",
    # "alc_grassmann",
    # "car_euclidean",
    # "car_grassmann",
    # "lip_euclidean",
    "lip_grassmann"
]
results = []

for test in tests:
    eva = "cv"
    dataset = test.split("_")[0]
    
    if dataset == "alc":
        X_train, X_test, y_train, y_test = preprocessing.eeg_irvine()
        X = np.concatenate((X_train, X_test), 0)
        y = np.concatenate((y_train, y_test))
        split = sklearn.model_selection.PredefinedSplit(([-1] * len(X_train)) + ([1] * len(X_test)))
    elif dataset == "car":
        X, y = preprocessing.vehicle_audio_percus()
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5)
    elif dataset == "lip":
        X, y = preprocessing.lip_naoki()
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5)
    
    if test == "alc_euclidean":
        pipeline = sklearn.pipeline.Pipeline([
            ("flatten", preprocessing.Flatten()),
            ("svm", sklearn.svm.SVC())
        ])
    elif test == "alc_grassmann":
        pipeline = sklearn.pipeline.Pipeline([
            ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
            ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
        ])
    elif test == "car_euclidean":
        pipeline = sklearn.pipeline.Pipeline([
            ("trimmer", preprocessing.Trimmer(start=-336000)),
            ("average", preprocessing.Average()),
            ("svm", sklearn.svm.SVC())
        ])
    elif test == "car_grassmann":
        pipeline = sklearn.pipeline.Pipeline([
            ("trimmer", preprocessing.Trimmer(start=-336000)),
            ("grassmann", arma.GrassmannSignal(hidden_dim=2, truncate=10)),
            ("manifold_svm", manifold_svm.ManifoldSVM(kern_gamma=10))
        ])
    elif test == "lip_euclidean":
        pipeline = sklearn.pipeline.Pipeline([
            ("pca", preprocessing.PCA(n_components=30)),
            ("flatten", preprocessing.Flatten()),
            ("svm", sklearn.svm.SVC())
        ])
    elif test == "lip_grassmann":
        pipeline = sklearn.pipeline.Pipeline([
            # ("pca", preprocessing.PCA(n_components=30)),
            ("grassmann", arma.GrassmannSignal(hidden_dim=10)),
            ("svm", manifold_svm.ManifoldSVM(kern_gamma=0.2))
        ])
    
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    if eva == "cv":
        cv = sklearn.model_selection.cross_validate(pipeline, X, y, n_jobs=-1, verbose=1, cv=split)
        print(test, cv["test_score"].mean(), cv["test_score"].std(), "\n")
        results.append(cv)
    elif eva == "single_test":
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.5)
        tr, te = list(split.split(X, y))[0]

        X_train = [X[i] for i in tr]
        X_test = [X[i] for i in te]
        y_train = [y[i] for i in tr]
        y_test = [y[i] for i in te]

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        pdb.set_trace()

for i, r in enumerate(results):
    print("{}  {:.4f}  {:.4f}".format(tests[i], r["test_score"].mean(), r["test_score"].std()))
