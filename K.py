import numpy as np
import math


# Input: numpy vector x of d rows, 1 column
#        numpy vector z of d rows, 1 column
# Output: kernel K(x,z) = exp(-1/2 * norm(x-z)^2)
# Example on how to call the script:
#     import K
#     v = K.run( np.array([[1], [4], [3]]) , np.array([[2], [7], [-1]]) )
def run(x, z):
    x = x.flatten()
    z = z.flatten()
    if x.size != z.size:
        raise ValueError
    return math.exp(-0.5 * np.sum((x - z) ** 2))

