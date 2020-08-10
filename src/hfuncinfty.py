import numpy as np

def hfuncinfty(x):
    y = np.zeros_like(x)
    x = np.abs(x)
    y[x <= 1/2] = 1

    ind = [i for i in range(len(x)) if 1/2 < x[i] and x[i] < 1]
    y[ind] = np.exp(-np.exp(2 / (1 - 2 * x[ind])) / (1 - x[ind]))

    return y
