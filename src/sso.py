import numpy as np
import pdb
from sklearn.base import BaseEstimator

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath("../lib/sso_han", nargout=0)

class SSO(BaseEstimator):
    """
    Transforms time series data into instantaneous frequencies and amplitudes using MATLAB code by
    Ningning Han. For use in sklearn.pipeline.

    Currently returns 1st (lowest-energy) instantaneous frequency of each channel of each signal
    only.

    References:
    Chui, Han, Mhaskar - Theory inspired deep network for instantaneous-frequency extraction and
    sub-signals recovery from discrete blind-source data.
    """
    def __init__(self, n_components=1, delta=1.0):
        self.n_components = n_components
        self.delta = 1.0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, list)

        X_mat = []
        for i in range(len(X)):
            X_mat.append(matlab.double(X[i].tolist()))

        _, freqs_mat = eng.Decompose3D(X_mat, self.delta, self.n_components, 1, nargout=2)

        freqs = []
        for i in range(len(freqs_mat)):
            freqs.append(np.asarray(freqs_mat[i]).squeeze())

        return freqs

if __name__ == "__main__":
    X = []
    for i in range(3):
        X.append(np.random.random((1024, 5)))
    sso = SSO()
    f = sso.transform(X)
