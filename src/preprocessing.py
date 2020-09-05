import numpy as np
import os
import pandas as pd
import pdb
import scipy.signal
from scipy.io import wavfile, loadmat
from sklearn.base import BaseEstimator

def vehicle_audio_percus():
    # dirs = ["../data/audio_data_percus/132_0430", "../data/audio_data_percus/133_0430"]
    dirs = ["../data/audio_data_percus/whiteblackred"]

    X = []
    y = []
    # labels = ["pickup", "sedan", "suv"]
    labels = ["white", "black", "red"]

    for dname in dirs:
        for fname in sorted(os.listdir(dname)):
            if not fname.split(".")[-1] == "wav":
                continue

            wav = wavfile.read(os.path.join(dname, fname))
            X.append(wav[1])
            
            for l in labels:
                if l in fname:
                    y.append(l)
                    break

    return X, np.array(y)

def lip_naoki():
    X = []
    y = []
    raw = loadmat("../data/videoclipdata_naoki/LipData.mat")["data"][0, 0]
    
    for r in raw:
        X.append(r)
    
    for d in range(5):
        for i in range(10):
            y.append(d + 1)

    return X, np.array(y)

def opportunity_activity():
    dname = "../data/opportunity/OpportunityUCIDataset/dataset"
    for fname in sorted(os.listdir(dname)):
        if not fname.split(".")[-1] == "dat":
            continue

        df = pd.read_csv(os.path.join(dname, fname), sep=" ", header=None)
        pdb.set_trace()

def eeg_irvine(ifreq=False):
    if ifreq:
        X_train = np.load("../data/eeg_irvine/freqs_train.npy")
        X_test = np.load("../data/eeg_irvine/freqs_test.npy")
    else:
        X_train = np.load("../data/eeg_irvine/X_train.npy")
        X_test = np.load("../data/eeg_irvine/X_test.npy")

    y_train = np.load("../data/eeg_irvine/y_train.npy")
    y_test = np.load("../data/eeg_irvine/y_test.npy")

    return X_train, X_test, y_train, y_test

def eeg_bonn(ifreq=False):
    if ifreq:
        X = np.load("../data/eeg_bonn/data/freqs.npy")
    else:
        X = np.load("../data/eeg_bonn/data/X.npy")

    y = np.load("../data/eeg_bonn/data/y.npy")

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
        return trimmed

class Detrend(BaseEstimator):
    """
    Removes linear trend from time series data. Expects timesteps to span axis 1. For use in
    sklearn.pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        detrended = []
        for x in X:
            detrended.append(scipy.signal.detrend(x, axis=0))
        return detrended
