import numpy as np

# print np.random.randn(10, 10)

# # 1
# print np.argmax([2,5,3])

# vector = [1, 2]
# matrix = [[1, 1], [2, 2], [3, 3], [4,4]]
# print np.dot(matrix, vector)
# w = [[[1st neuron in l+1: weight for 1st neuron in l, weight for 2nd neuron in l, etc], [2nd neuron in l+1]], [[1st neuron in l+2]]
#
# w from layer with 3 neurons to layer with 2:
w = np.matrix([[1, 0, -1], [0, 1, 1]])
# activations for the layer with 3 neurons:
# (note that 'array' is the name used for a vector in numpy)
a = np.array([1, 0, 1])
# desired result for w * a:
# [1 + 0 + -1, 0 + 0 + 1] => [0, 1]
print np.dot(w, a)

# now the same thing with as:
az = np.matrix([[1, 0, 1], [0, 0, 0]])
# => [[0, 1], [0, 0]]
# az = [
# [1, 0],
# [0, 0],
# [1, 0]
# ]

# doesn't work
# print az * w
# neither does this
# print np.sum(a*w)
# print np.dot(w, az)
#
print np.dot(az, w.transpose())

invertible = np.matrix([
  [3, 20],
  [5, 4]
  ])
inverse = invertible.getI()

# both should be the identity matrixpl
print invertible * inverse
print inverse * invertible

vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
np.outer(vector1, vector2)

