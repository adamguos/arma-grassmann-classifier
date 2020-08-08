import arma
import numpy as np
import pdb
from scipy.io import loadmat
import sklearn
import warnings

import manifold_svm

def handle_warning(message, category, filename, lineno, file=None, line=None):
    print('A warning occurred:')
    print(message)
    print('Do you wish to continue?')

    while True:
        response = input('y/n: ').lower()
        if response not in {'y', 'n'}:
            print('Not understood.')
        else:
            break

    if response == 'n':
        raise category(message)

warnings.showwarning = handle_warning

data = []
labels = []
raw = loadmat("../data/videoclipdata_naoki/LipData.mat")["data"][0, 0]

for row in raw:
    data.append(row)

for d in range(5):
    for i in range(10):
        labels.append(d + 1)

obs = arma.subspace_representation(data, 5, 5)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        obs, labels, test_size=0.4, random_state=0)

clf = manifold_svm.manifold_svm(X_train, y_train)
print(clf.score(X_test, y_test))

X_train = np.load("../data/eeg_irvine/X_train.npy")
X_test = np.load("../data/eeg_irvine/X_test.npy")
y_train = np.load("../data/eeg_irvine/y_train.npy")
y_test = np.load("../data/eeg_irvine/y_test.npy")

obs_train = arma.subspace_representation(X_train, 5, 5)
obs_test = arma.subspace_representation(X_test, 5, 5)

clf = manifold_svm.manifold_svm(obs_train, y_train)
print(clf.score(obs_test, y_test))
