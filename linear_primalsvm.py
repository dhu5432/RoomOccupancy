import numpy as np
import cvxopt as co
def run(X, y):
	H = np.identity(len(X[0]))
	f = np.zeros((len(X[0]), 1))
	A = np.array([ [-y[i] * X[i][j] for j in range(len(X[0]))] for i in range(len(X))]).reshape(len(X), len(X[0]))
	b = np.negative(np.ones((len(X), 1)))
	
	return np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'), co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
