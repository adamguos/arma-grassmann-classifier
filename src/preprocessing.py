import ffmpeg
import numpy as np
import os
import pdb
import scipy.io.wavfile

def vehicle_audio_percus():
    dirs = ["../data/audio_data_percus/132_0430", "../data/audio_data_percus/133_0430"]
    target = "../data/audio_data_percus"

    X = []
    y = []
    labels = ["pickup", "sedan", "suv"]

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

if __name__ == "__main__":
    X, y = vehicle_audio_percus()
