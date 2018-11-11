import numpy as np
# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
def run(L,X,y):
    (n,d)=np.shape(X)
    theta = np.zeros((d, 1))
    for i in range(0, L):
	#X, y = shuffle_in_unison(X,y)
        for t in range(0, n):
            if (y[t] * (np.dot(X[t],  theta))[0]) <= 0:
                theta = theta + np.array([y[t]* X[t]]).T

    return theta

def shuffle_in_unison(a,b):
	assert len(a)==len(b)
	shuffled_a = np.empty(a.shape, dtype=a.dtype)
	shuffled_b = np.empty(b.shape, dtype=b.dtype)
	permutation = np.random.permutation(len(a))
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index]
		shuffled_b[new_index] = b[old_index]
	return shuffled_a, shuffled_b
