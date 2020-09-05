from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import resample

def oversample(X, y):
    """
    Balances classes in X and y by oversampling minority class. Only supports two classes.
    """
    classes = np.unique(y)

    # ensure classes[0] is majority class, and classes[1] is minority class
    if sum(y == classes[0]) < sum(y == classes[1]):
        classes[0], classes[1] = classes[1], classes[0]

    new_samples = resample(X[y == classes[1]], replace=True, n_samples=sum(y == classes[0]))

    X = np.concatenate((X[y == classes[0]], new_samples), axis=0)
    y = np.concatenate((np.array([classes[0]] * sum(y == classes[0])), \
            np.array([classes[1]] * len(new_samples))), axis=0)

    return X, y

def undersample(X, y):
    """
    Balances classes in X and y by undersampling majority class. Only supports two classes.
    """
    classes = np.unique(y)

    # ensure classes[0] is majority class, and classes[1] is minority class
    if sum(y == classes[0]) < sum(y == classes[1]):
        classes[0], classes[1] = classes[1], classes[0]

    new_samples = resample(X[y == classes[0]], replace=False, n_samples=sum(y == classes[1]))

    X = np.concatenate((new_samples, X[y == classes[1]]), axis=0)
    y = np.concatenate((np.array([classes[0]] * len(new_samples)), \
            np.array([classes[1]] * sum(y == classes[1]))), axis=0)

    return X, y

def smote(X, y):
    """
    Balances classes in X and y by generating new samples of minority class. Only supports two
    classes.
    """
    classes = np.unique(y)

    # ensure classes[0] is majority class, and classes[1] is minority class
    if sum(y == classes[0]) < sum(y == classes[1]):
        classes[0], classes[1] = classes[1], classes[0]

    sm = SMOTE()
    X, y = sm.fit_sample(X, y)

    return X, y
