import numpy as np
import scipy

"""
References:
    [1] Jayasumana et al: Kernel methods on Riemannian manifolds with Gaussian RBF kernels
"""

def distance_matrix(X, Y, metric):
    """
    Computes distance matrix between elements of X and Y, using metric. Returns 2D array distance,
    where distance[i, j] = metric(X[i], Y[j]).
    """
    distance = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            m = metric(X[i], Y[j])
            if np.isnan(m):
                pdb.set_trace()
            distance[i, j] = m
    return distance

def projection_metric_sq(Y1, Y2):
    """
    Computes square of projection metric between points Y1 and Y2 on Grassmann manifold. Refer to
    [1] section 6.4.
    """
    assert Y1.shape == Y2.shape

    # abs used to ensure distance is non-negative in case of numerical imprecision
    # round m down to 0 in case of numerical imprecision
    m = Y1.shape[1] - (np.linalg.norm(Y1.transpose() @ Y2) ** 2)
    if np.isclose(m, 0):
        m = 0
    return m

def arc_length_sq(Y1, Y2):
    """
    Computes square of arc length metric (the geodesic) between points Y1 and Y2 on Grassmann
    manifold. Refer to [1] section 6.4.
    """
    assert Y1.shape == Y2.shape

    s = scipy.linalg.svdvals(Y1.transpose() @ Y2)

    # handle numerical imprecision
    s[np.isclose(s, 1)] = 1
    s[np.isclose(s, -1)] = -1
    theta = np.arccos(s)

    return (theta ** 2).sum()
