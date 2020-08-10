import pdb
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator

def arma_params(data, hidden_dim):
    """
    Implements closed-form solution to estimate parameters of ARMA model, given time series data.

    Parameters:
    data:       list of 2-dim ndarrays, each representing a signal to be computed. For each list
                element, time samples are across axis 0, features across axis 1.
    hidden_dim: size of hidden state vector. [2] recommends 5-10.

    Returns:
    params:     list of ARMA parameters. Each list item corresponds to an independent signal and is
                a tuple with 2 ndarrays: A, C in order

    References:
    [1] Aggarwal, Roy Chowdhury, Chellappa -- "A system identification approach for
    video-based face recognition"
    [2] Turaga, Veeraraghavan, Srivastava, Chellappa -- "Statistical computations on Grassmann and
    Stiefel manifolds for image and video-based recognition"
    """

    for d in data:
        assert d.ndim == 2

    params = []

    for d in data:
        signal = d.transpose()
        U, s, Vh = linalg.svd(signal, full_matrices=False)

        # the k-largest singular values of signal are kept, where k = hidden_dim
        U = U[:, :hidden_dim]
        Vh = Vh[:hidden_dim, :]
        Sigma = np.diag(s[:hidden_dim])

        n_timesteps = signal.shape[1]   # tau in [1] and [2]
        D1 = np.block([
            [np.zeros((1, n_timesteps - 1)), np.zeros((1, 1))],
            [np.eye(n_timesteps - 1), np.zeros((n_timesteps - 1, 1))]
        ])
        D2 = np.block([
            [np.eye(n_timesteps - 1), np.zeros((n_timesteps - 1, 1))],
            [np.zeros((1, n_timesteps - 1)), np.zeros((1, 1))]
        ])

        C = U
        A = Sigma @ Vh @ D1 @ Vh.transpose() @ linalg.inv(Vh @ D2 @ Vh.transpose()) @ linalg.inv(
                Sigma)

        params.append((A, C))

    return params

def observability_matrix(data, hidden_dim, truncate):
    """
    Computes finite observability matrix based on ARMA parameters for data.

    Parameters:
    data:       list of 2-dim ndarrays, each representing a signal to be computed. For each list
                element, time samples are across axis 0, features across axis 1.
    hidden_dim: size of hidden state vector. [2] recommends 5-10.
    truncate:   number of block matrices to include in observability matrix.

    Returns:
    obs:        list of finite observability matrices, each of size mp x d, where m = truncate, p =
                data[i].shape[1], d = hidden_dim

    References:
    [1] Aggarwal, Roy Chowdhury, Chellappa -- "A system identification approach for
    video-based face recognition"
    [2] Turaga, Veeraraghavan, Srivastava, Chellappa -- "Statistical computations on Grassmann and
    Stiefel manifolds for image and video-based recognition"
    """
    params = arma_params(data, hidden_dim)

    m = truncate
    d = hidden_dim
    obs = []

    for i, param in enumerate(params):
        p = data[i].shape[1]
        ob = np.zeros((m * p, d))
        A = param[0]
        C = param[1]

        for mi in range(m):
            ob[(mi * p):((mi + 1) * p), :] = C @ np.linalg.matrix_power(A, mi)

        obs.append(ob)

    return obs

def subspace_representation(data, hidden_dim, truncate):
    """
    Computes the representation of given time series data as a point on a Grassmann manifold (i.e. a
    subspace of Euclidean space).

    Parameters:
    data:       list of 2-dim ndarrays, each representing a signal to be computed. For each list
                element, time samples are across axis 0, features across axis 1.
    hidden_dim: size of hidden state vector. [2] recommends 5-10.
    truncate:   number of block matrices to include in observability matrix.

    Returns:
    subs:       list of orthonormal matrices whose columns span the subspace representing each
                signal

    References:
    [2] Turaga, Veeraraghavan, Srivastava, Chellappa -- "Statistical computations on Grassmann and
    Stiefel manifolds for image and video-based recognition"
    """
    obs = observability_matrix(data, hidden_dim, truncate)
    subs = []

    for ob in obs:
        subs.append(linalg.orth(ob))

    return subs

class GrassmannSignal(BaseEstimator):
    """
    Transforms time series data into points on a Grassmann manifold, represented by an orthonormal
    matrix. For use in sklearn.pipeline.
    """
    def __init__(self, hidden_dim=5, truncate=5):
        self.hidden_dim = hidden_dim
        self.truncate = truncate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return subspace_representation(X, self.hidden_dim, self.truncate)
