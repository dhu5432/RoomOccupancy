import K
import numpy as np


def run(L, X, y):
    n = y.shape[0]
    alpha = np.zeros(n)
    for iter_ in range(L):
        for t in range(n):
            pred = 0
            for i in range(n):
                pred += alpha[i] * y[i] * K.run(X[i, :], X[t, :])
            if y[t] * pred <= 0:
                alpha[t] += 1
    return alpha.reshape(-1, 1)

