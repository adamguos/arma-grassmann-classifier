import numpy as np

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
