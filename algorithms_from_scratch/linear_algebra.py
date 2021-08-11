from functools import reduce
from typing import List
import math


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

    TODO Bad point no check of same size

    :param v:List of Vectors
    :return: Vector mean
    """
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
    return math.sqrt(vector_squared_distance(v, w))  # TODO check me out


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
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols
