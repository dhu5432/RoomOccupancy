import K


def run(alpha, X, y, z):
    n, result = y.shape[0], 0
    for i in range(n):
        result += alpha[i] * y[i] * K.run(X[i, :], z)
    label = 1 if result > 0 else -1
    return label

