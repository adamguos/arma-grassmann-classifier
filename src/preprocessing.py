import numpy as np
import os
import pdb
import scipy.io.wavfile
from sklearn.base import BaseEstimator

def vehicle_audio_percus():
    # dirs = ["../data/audio_data_percus/132_0430", "../data/audio_data_percus/133_0430"]
    dirs = ["../data/audio_data_percus/whiteblackred"]
    target = "../data/audio_data_percus"

    X = []
    y = []
    # labels = ["pickup", "sedan", "suv"]
    labels = ["white", "black", "red"]

    for dname in dirs:
        for fname in sorted(os.listdir(dname)):
            if not fname.split(".")[-1] == "wav":
                continue

            wav = scipy.io.wavfile.read(os.path.join(dname, fname))
            X.append(wav[1])
            
            for l in labels:
                if l in fname:
                    y.append(l)
                    break

    X = np.array(X, dtype=object)
    y = np.array(y)

    np.save(os.path.join(target, "X"), X)
    np.save(os.path.join(target, "y"), y)

    return X, y

class Trimmer(BaseEstimator):
    """
    Trims time series data. Expects timesteps to span axis 1. For use in sklearn.pipeline.

    Parameters:
    start, end: indices to start and end trim, supports Python rules. None means no trim.
    """
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        trimmed = []
        for x in X:
            trimmed.append(x[self.start:self.end])
        return np.array(trimmed)

if __name__ == "__main__":
    X, y = vehicle_audio_percus()
