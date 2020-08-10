import numpy as np
from sklearn import svm

"""
References:
    [1] Jayasumana et al: Kernel methods on Riemannian manifolds with Gaussian RBF kernels
"""

def projection_metric_sq(Y1, Y2):
    """
    Computes projection metric between points Y1 and Y2 on Grassmann manifold. Refer to [1] section
    6.4.
    """
    assert Y1.shape == Y2.shape
    return Y1.shape[1] - (np.linalg.norm(Y1.transpose() @ Y2) ** 2)

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

def manifold_svm(X, y, gamma):
    """
    Returns scikit-learn svm object using Gaussian projection metric kernel on Grassmann manifold.

    Parameters:
        X:      data
        y:      class labels
        gamma:  hyperparameter controlling Gaussian variance
    """
    kern = lambda Y1, Y2 : gaussian_kernel(Y1, Y2, projection_metric_sq, gamma)
    clf = svm.SVC(kernel=kern)
    clf.fit(X, y)
    return clf

class ManifoldSVM(svm.SVC):
    """
    SVM classifier using Gaussian projection metric kernel on Grassmann manifold. For use in
    sklearn.pipeline.
    """
    def __init__(self, kern_gamma=0.5):
        self.kern_gamma = kern_gamma
        kern = lambda Y1, Y2 : gaussian_kernel(Y1, Y2, projection_metric_sq, self.kern_gamma)
        super().__init__(kernel=kern)
