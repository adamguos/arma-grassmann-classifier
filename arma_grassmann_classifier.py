from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.pipeline
import sys

sys.path.append("src/")

import arma
import manifold_svm

class ARMAGrassmannClassifier(sklearn.pipeline.Pipeline):
    """
    Returns sklearn.pipeline.Pipeline object that performs ARMA Grassmann classification on supplied
    time series data. Supports Pipeline functions such as fit, transform, etc.
    """
    def __init__(self, hidden_dim=10, truncate=None, kern_gamma=0.2):
        self.hidden_dim = hidden_dim
        self.truncate = hidden_dim if truncate is None else truncate
        self.kern_gamma = kern_gamma

        super().__init__([
            ("grassmann", arma.GrassmannSignal(hidden_dim=self.hidden_dim, truncate=self.truncate)),
            ("svm", manifold_svm.ManifoldSVM(kern_gamma=self.kern_gamma))
        ])
