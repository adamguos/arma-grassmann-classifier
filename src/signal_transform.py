import numpy as np
from scipy.fftpack import dct, idct
from sklearn.base import BaseEstimator, ClassifierMixin

class DCT(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        d = []
        for x in X:
            assert x.ndim == 2
            d.append(dct(dct(x.T, norm="ortho").T, norm="ortho"))
        return d
        # return dct(dct(X.transpose(0, 2, 1), norm="ortho").transpose(0, 2, 1), norm="ortho")

    def inverse_transform(self, X, y=None):
        d = []
        for x in X:
            assert x.ndim == 2
            d.append(idct(idct(x.T, norm="ortho").T, norm="ortho"))
        return d
        # return idct(idct(X.transpose(0, 2, 1), norm="ortho").transpose(0, 2, 1), norm="ortho")
