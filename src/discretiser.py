import numpy as np
import pdb
from sklearn.base import BaseEstimator, ClassifierMixin

class Discretiser(BaseEstimator, ClassifierMixin):
    """
    Converts continuous predictions to discrete classification labels.
    """
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.n_classes = len(np.unique(y_train))
        return self

    def predict(self, X_test):
        X_test = X_test - X_test.min()

        if not X_test.max() == 0:
            X_test = (X_test / X_test.max() * self.n_classes).astype(int)

        X_test[X_test >= self.n_classes] = self.n_classes - 1
        return X_test

