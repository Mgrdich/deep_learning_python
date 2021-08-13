from functools import reduce
from typing import List, Callable
import math

from util.Util_lib import Util_Lib


def vector_add(v: List, w: List) -> List:
    """
    Addition of two Vectors
    The only thing is being not same size the zip is discarding
    the remaining ones
    :param v: first vector
    :param w: second vector
    :return: vector
    """
    return [
        v_i + w_i for v_i, w_i in zip(v, w)
    ]


def vector_subtract(v: List, w: List) -> List:
    """
    Subtraction of two Vectors
    :param v: first vector
    :param w: second vector
    :return: vector
    """
    return [
        v_i - w_i for v_i, w_i in zip(v, w)
    ]


def vectors_sum(v: List[List]) -> List:
    """
    Sum all the corresponding elements
    :param v: List of Lists
    :return: a Vector
    """

    result = v[0]
    for vector in v[1:]:
        result = vector_add(result, vector)

    return result


def vector_sum_v2(v: List[List]) -> List:
    """
    Sum all the corresponding elements
    :param v: List of Lists
    :return: a Vector
    """

    return reduce(vector_add, v)


def scalar_multiply_vector(c: float, v: List) -> List:
    """
    Multiple a Vector By a Scalar
   :param c: Number a scalar
   :param v: the vector
   :return: Result Vector
   """
    return [c * v_i for v_i in v]


def vectors_mean(v: List[List]) -> List:
    """
    compute the vector whose ith element is the mean of the ith
    elements of the input vectors
    :param v:List of Vectors
    :return: Vector mean
    """
    # TODO make me a decorator perhaps?
    if not Util_Lib.isTensor(v):
        raise Exception('Not a matrix')

    # isListSameLength function should be used  TODO Bad point no check of same size

    n = len(v)
    return scalar_multiply_vector(1 / n, vectors_sum(v))


def vector_dot(v: List, w: List) -> float:
    """
    The dot product of two vectors
    v_1 * w_1 + v_2 * w_2 + .... + v_n * w_n
    :param v: First vector
    :param w: Second vector
    :return: Number
    """

    # every turn gets added in the parameter # ASK COOL ONE
    return sum(
        v_i * w_i for v_i, w_i in zip(v, w)
    )


def sum_of_squares(v: List) -> float:
    """
    The sum of the Squares of a Vector
    :param v: a Vector
    :return: a Number
    """
    return vector_dot(v, v)


def vector_magnitude(v: List) -> float:
    return math.sqrt(sum_of_squares(v))


def vector_squared_distance(v: List, w: List) -> float:
    """
    (v_1 - w_2) **2 + ... + (v_n - w_n) **2
    :param v: First Vector
    :param w: Second Vector
    :return: a vector of squared distance
    """
    return sum_of_squares(vector_subtract(v, w))


def vector_distance(v: List, w: List) -> float:
    """
    Calculates the Vectors distance
    :param v: First Vector
    :param w: Second Vector
    :return: Distance
    """
    return math.sqrt(vector_squared_distance(v, w))


def vector_distance_v2(v: List, w: List) -> float:
    """
    Calculates the Vectors distance
    :param v: First Vector
    :param w: Second Vector
    :return: Distance
    """
    return vector_magnitude(vector_subtract(v, w))


# Matrices
def matrix_shape(A: List[List]) -> tuple:
    """
    :param A: Matrix
    :return: tuple represent the shape
    """

    # TODO make me a decorator perhaps?
    if not Util_Lib.isTensor(A):
        raise Exception('Not a matrix')

    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_matrix_row(A: List[List], i: int) -> List:
    """
    :param A: Matrix
    :param i: the row index
    :return: the current row
    """
    return A[i]


def get_column_row(A: List[List], i: int) -> List:
    """
    :param A: Matrix
    :param i: the row index
    :return: the current row
    """
    return [
        A_i[i] for A_i in A
    ]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable) -> List[List]:
    # TODO if not passed entry_fn make the default the indexes

    """
    :param num_rows: number of rows
    :param num_cols: number of cols
    :param entry_fn:
    :return: Matrix
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]  # create one list for each i


def make_identity_matrix(square_matrix_length: int) -> List[List]:
    """
    :param square_matrix_length:
    :return: Matrix
    """
    return make_matrix(square_matrix_length, square_matrix_length, lambda i, j: 1 if i == j else 0)
