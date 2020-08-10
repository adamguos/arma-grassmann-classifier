import numpy as np
import time
from sklearn.model_selection import train_test_split

from hermite_classifier import HermiteClassifier

total = 0
for i in range(100):
    start = time.time()
    hc = HermiteClassifier(8, 3, 0.5)
    
    X = np.random.random((100, 2)) * 2 - 1
    y = np.sign(X[:, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    hc.fit(X_train, y_train)
    score = np.sum(np.sign(hc.predict(X_test)) == y_test) / len(X_test)
    total += score
    print(score, total / (i + 1), time.time() - start)
