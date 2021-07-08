import numpy as np

from util.functions import naive_relu, naive_add, numpy_add, naive_add_matrix_and_vector

x = np.array([[1, 2], [1, 2]])

y = np.array([[-1, 2], [0, 6]])

# print(naive_relu(y))

# print(naive_add(x, y))

x1 = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]])

y1 = np.array([[[-1, 2], [0, 6]], [[-1, 2], [0, 6]], [[-1, 2], [0, 6]]])

# numpy_add(x1, y1)

x2 = np.array([[1, 3, 3], [1, 2, 6]])
y2 = np.array([1, 3, 3])

# print(naive_add_matrix_and_vector(x2, y2))
