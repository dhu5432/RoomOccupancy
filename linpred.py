import numpy as np
# Input: numpy vector theta of d rows, 1 column
#        numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
def run(theta,x):
    return 1 if np.dot(theta,x) > 0 else -1
