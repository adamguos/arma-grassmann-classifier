import numpy as np
import pdb
from sklearn import svm

import hermite
import metrics

"""
References:
    [1] Jayasumana et al: Kernel methods on Riemannian manifolds with Gaussian RBF kernels
"""

def gaussian_kernel(Y1, Y2, metric_sq, gamma):
    """
    Computes Gaussian (rbf) kernel matrix on data arrays Y1 and Y2, using supplied metric function.

    Parameters:
        Y1, Y2:     arrays of data points
        metric_sq:  square of metric function to use
        gamma:      hyperparameter controlling Gaussian variance
    """
    kern_matrix = np.zeros((len(Y1), len(Y2)))
    for i in range(len(Y1)):
        for j in range(len(Y2)):
            kern_matrix[i, j] = np.exp(-gamma * metric_sq(Y1[i], Y2[j]))
    return kern_matrix

def hermite_kernel(Y1, Y2, metric, n, q):
    """
    Computes Hermite kernel matrix on data arrays Y1 and Y2, using supplied metric function.

    Parameters:
        Y1, Y2:     arrays of data points
        metric:     metric function to use
        n:          hyperparameter, floor(n^2/2) = max degree of Hermite polynomial
        q:          hyperparameter, dimension of space that manifold is embedded in
    """
    distances = metrics.distance_matrix(Y1, Y2, metric).flatten()
    kern_matrix = hermite.Phi(n, q, distances).reshape(len(Y1), len(Y2))
    return kern_matrix

def manifold_svm(X, y, gamma):
    """
    Returns scikit-learn svm object using Gaussian projection metric kernel on Grassmann manifold.

    Parameters:
        X:      data
        y:      class labels
        gamma:  hyperparameter controlling Gaussian variance
    """
    kern = lambda Y1, Y2 : gaussian_kernel(Y1, Y2, metrics.projection_metric_sq, gamma)
    clf = svm.SVC(kernel=kern)
    clf.fit(X, y)
    return clf

class ManifoldSVM(svm.SVC):
    """
    SVM classifier using Gaussian projection metric kernel on Grassmann manifold. For use in
    sklearn.pipeline.
    """
    def __init__(self, kern_gamma=0.2):
        self.kern_gamma = kern_gamma
        kern = lambda Y1, Y2 : gaussian_kernel(Y1, Y2, metrics.projection_metric_sq,
                self.kern_gamma)
        super().__init__(kernel=kern)

    def transform(self, X):
        return super().predict(X)

class HermiteSVM(svm.SVC):
    """
    SVM classifier using Hermite kernel on Grassmann manifold. Rescales input such that maximum norm
    distance between points is 1.0. For use in sklearn.pipeline.
    """
    def __init__(self, n=8, q=6, kern_metric=None):
        self.n = n
        self.q = q
        self.kern_metric = kern_metric

        if kern_metric is None:
            kern_metric = lambda X, Y : np.linalg.norm(X - Y)

        kern = lambda Y1, Y2 : hermite_kernel(Y1, Y2, kern_metric, n, q)
        super().__init__(kernel=kern)

    def transform(self, X):
        return super().predict(X)
