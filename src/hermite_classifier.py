"""
References:
    [1] Mhaskar: A direct approach for function approximation on data defined manifolds
"""

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
import pdb

import hermite

class HermiteClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier using Hermite approximator.

    [1] equation (3.8)
    """

    def __init__(self, n, q, alpha):
        self.n = n
        self.q = q
        self.alpha = alpha

    def fit(self, X_train, y_train):
        self.coef = self.n ** (self.q * (1 - self.alpha)) / len(X_train)
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        norms = scipy.spatial.distance_matrix(self.X_train, X_test)
        labels = np.repeat([self.y_train], len(X_test), axis=0).transpose()
        Phis = np.zeros_like(labels)
        for j in range(Phis.shape[0]):
            # norms = np.linalg.norm(np.repeat([self.y_train], len(X_test), axis=0) - X_test, axis=1)
            Phis[j, :] = hermite.Phi(self.n, self.q, norms[j, :])

        return self.coef * np.sum(labels * Phis, axis=0)
