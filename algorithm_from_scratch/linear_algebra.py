from typing import List

from util.Util_lib import Util_Lib


def vector_add(v: List, w: List) -> List:
    """
    Addition of two Vectors
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


def vector_sum(v: List[List]) -> List:
    """
    Sum all the corresponding elements
    :param v: List of Lists
    :return: a Vector
    """

    result = v[0]
    for vector in v[1:]:

        result = vector_add(result,vector)

