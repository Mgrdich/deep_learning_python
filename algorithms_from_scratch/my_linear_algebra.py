from __future__ import annotations

import math
from functools import wraps
from random import randint
from typing import Union, Callable, List, Tuple
from util.Util_lib import Util_Lib as uL
from util.typings import NUMBER, ITERABLE, NUMBER_OR_ITERABLE
from util.decorators import exception_decorators

exception_is_number: Callable = exception_decorators['is_number']


def vector_other_validation(function: Callable):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Vector):
            raise Exception('The parameter Not Vector Type')

        if args[0].length != args[1].length:
            raise Exception('Two Vectors not of the same shape')

        return function(*args, **kwargs)

    return wrapper


def vector_out_of_bounds_validation(function: Callable):
    @wraps(function)
    def wrapper(*args, **kwargs):
        # args[0] --> self
        vector_length = args[0].length
        key: NUMBER = args[1]

        equality: bool = (key >= vector_length) if uL.is_positive(key) else (key < (vector_length * -1))

        if equality:
            raise Exception('Index out of bounds')

        return function(*args, **kwargs)

    return wrapper


def matrix_other_validation(function: Callable):
    @wraps(function)
    def wrapper(*args):
        return function(*args)

    return wrapper


class Vector:
    def __init__(self, iterator: ITERABLE):
        self.__local_vector: list = []
        self.__length: int = 0

        if not uL.is_iterable(iterator):
            raise Exception('Parameter must by an iterator object')

        if uL.is_dictionary(iterator):
            iterator = iterator.values()

        iterator = iterator.copy()

        for elem in iterator:
            if not uL.isNumber(elem):
                raise Exception('Not all of excepted type')

            self.__local_vector.append(elem)

        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    def __add__(self, other: VECTOR_OR_NUMBER) -> Vector:
        return self.__element_vector_operator(other, lambda i, j: i + j)

    def __sub__(self, other: VECTOR_OR_NUMBER) -> Vector:
        return self.__element_vector_operator(other, lambda i, j: i - j)

    def __mul__(self, other: VECTOR_OR_NUMBER) -> Vector:
        return self.__element_vector_operator(other, lambda i, j: i * j)

    def __truediv__(self, other: VECTOR_OR_NUMBER) -> Vector:
        return self.__element_vector_operator(other, lambda i, j: i / j)

    def __floordiv__(self, other: VECTOR_OR_NUMBER) -> Vector:
        return self.__element_vector_operator(other, lambda i, j: i // j)

    def __eq__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i == j)

    def __ne__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i != j)

    def __gt__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i > j)

    def __ge__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i >= j)

    def __lt__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i < j)

    def __le__(self, other: Vector) -> Vector:
        return self.__element_condition_operator(other, lambda i, j: i <= j)

    def __ceil__(self) -> Vector:
        return self.__element_operation(lambda i: math.ceil(i))

    def __floor__(self) -> Vector:
        return self.__element_operation(lambda i: math.floor(i))

    def __float__(self) -> Vector:
        return self.__element_operation(lambda i: float(i))

    def __pow__(self, power: NUMBER, modulo=None) -> Vector:
        if not uL.isNumber(power):
            raise Exception('power must be a number')

        return self.__element_operation(lambda i: math.pow(i, power))

    @vector_out_of_bounds_validation
    def __getitem__(self, key: int) -> NUMBER:
        return self.__local_vector[key]

    @vector_out_of_bounds_validation
    def __setitem__(self, key: int, value: NUMBER):
        self.__local_vector[key] = value

    def __iter__(self) -> Vector:
        self.n = 0
        return self

    def __next__(self) -> NUMBER:
        if self.n < self.__length:
            result = self.__local_vector[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return "Vector({0})".format(self.__local_vector)

    def __repr__(self) -> str:
        return self.__str__()

    @vector_other_validation
    def dot(self, other: Vector) -> NUMBER:
        return sum(self * other)

    @vector_other_validation
    def squared_distance(self, other: Vector) -> NUMBER:
        new_vector: Vector = self - other
        return new_vector.dot(new_vector)

    @vector_other_validation
    def distance(self, other: Vector) -> NUMBER:
        return math.sqrt(self.squared_distance(other))

    @exception_is_number(True)
    def append(self, value: VECTOR_OR_NUMBER_OR_ITERABLE):
        self.__local_vector.append(value)
        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    @exception_is_number
    def insert_at(self, value: VECTOR_OR_NUMBER_OR_ITERABLE, index: int):

        if index >= self.length:  # TODO here validation negative check
            raise Exception('index is out of bound')

        self.__local_vector.insert(index, value)
        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    def sum_of_squares(self) -> NUMBER:
        return self.dot(self)

    @property
    def length(self) -> int:
        return self.__length

    @property
    def shape(self) -> tuple:
        return self.length,

    @property
    def magnitude(self) -> NUMBER:
        return math.sqrt(self.sum_of_squares())

    @classmethod
    def rand_int(cls, length: int, start: int, end: int) -> Vector:
        vector = cls([])

        if not end:
            end = start
            start = 0

        if start > end:
            raise Exception('Parameter is wrong')

        for i in range(length):
            vector.append(randint(start, end))

        return vector

    def __element_vector_operator(self, other: VECTOR_OR_NUMBER, func: Callable[[NUMBER, NUMBER], NUMBER]) -> Vector:
        """
        Private function that acts as a helper for normal arithmetic operations
        :param other: a Number or a Vector
        :param func: must be a pure function that does the operation and returns a Number
        :return Vector instance
        """
        if uL.isNumber(other):
            return Vector([
                func(i, other) for i in self
            ])

        if isinstance(other, self.__class__):

            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.length)

            for ind in range(self.length):
                new_vector[ind] = func(self[ind], other[ind])

            return new_vector

        raise Exception('Element type is not supported')

    @vector_other_validation
    def __element_condition_operator(self, other: VECTOR_OR_NUMBER, func: Callable[[NUMBER, NUMBER], NUMBER]) -> Vector:
        """
        Private function that acts as a helper for one to one vector operations
        :param other: a Number or a Vector
        :param func: must be a pure function that does the operation and returns a Number
        :return Vector instance
        """
        return Vector([func(self[i], other[i]) for i in range(self.length)])

    def __element_operation(self, func: Callable[[NUMBER], NUMBER]) -> Vector:
        """
        Private function that acts as a helper for one to one vector operations
        :param func: must be a pure function that does the operation and returns a Number
        :return Vector instance
        """
        return Vector([func(i) for i in self])


class Matrix:
    def __init__(self, iterator: ITERABLE[Vector, ITERABLE]):
        self.__local_matrix: List[Vector] = []
        last_vector_shape = None

        for i in iterator:
            if isinstance(i, Vector):
                vector_element = i
            else:
                vector_element = Vector(i)

            if not uL.is_none(last_vector_shape) and vector_element.shape != last_vector_shape:
                raise Exception('Nested Vectors are not of the same length')

            last_vector_shape = vector_element.shape

            self.__shape: tuple = (0, vector_element.shape)
            self.__local_matrix.append(vector_element)

        self.__shape: tuple = (len(self.__local_matrix), last_vector_shape)

    def __add__(self, other: MATRIX_OR_NUMBER) -> Matrix:
        return self.__element_matrix_operator(other, lambda i, j: i + j)

    def __sub__(self, other: MATRIX_OR_NUMBER) -> Matrix:
        return self.__element_matrix_operator(other, lambda i, j: i - j)

    def __mul__(self, other: MATRIX_OR_NUMBER) -> Matrix:
        return self.__element_matrix_operator(other, lambda i, j: i * j)

    def __truediv__(self, other: MATRIX_OR_NUMBER) -> Matrix:
        return self.__element_matrix_operator(other, lambda i, j: i / j)

    def __floordiv__(self, other: MATRIX_OR_NUMBER) -> Matrix:
        return self.__element_matrix_operator(other, lambda i, j: i // j)

    def __eq__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i == j)

    def __ne__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i != j)

    def __gt__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i > j)

    def __ge__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i >= j)

    def __lt__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i < j)

    def __le__(self, other: Matrix) -> Matrix:
        return self.__element_condition_operator(other, lambda i, j: i <= j)

    def __ceil__(self) -> Matrix:
        return self.__element_operation(lambda i: math.ceil(i))

    def __floor__(self) -> Matrix:
        return self.__element_operation(lambda i: math.floor(i))

    def __float__(self) -> Matrix:
        return self.__element_operation(lambda i: float(i))

    def __pow__(self, power, modulo=None) -> Matrix:
        if not uL.isNumber(power):
            raise Exception('power must be a number')

        return self.__element_operation(lambda i: math.pow(i, power))

    def __iter__(self) -> Matrix:
        self.n = 0
        return self

    def __next__(self) -> Vector:
        if self.n < self.row:
            result = self.__local_matrix[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, key: int) -> Vector:
        # TODO Decorator
        if key >= self.row:
            raise Exception('Matrix out of bounds')

        return self.__local_matrix[key]

    def __setitem__(self, key: int, value: Vector):
        if key >= self.row:
            raise Exception('Matrix out of bounds')

        if not isinstance(value, Vector):
            raise Exception('Value is not a Vector')

        if value.shape != self.column:
            raise Exception('Vector size not compatible')

        self.__local_matrix[key] = value

    def __str__(self) -> str:
        return "Matrix({0})".format(self.__local_matrix)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.row

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @property
    def row(self) -> int:
        return self.__shape[0]

    @property
    def column(self) -> int:
        return self.__shape[1]

    @classmethod
    def ones(cls, shape: tuple) -> Matrix:
        pass

    @classmethod
    def eyes(cls, shape: tuple) -> Matrix:
        pass

    @classmethod
    def identity(cls, shape: tuple) -> Matrix:
        pass

    def __element_matrix_operator(self, other: MATRIX_OR_NUMBER, func: MATRIX_OPERATOR_CALLBACK) -> Matrix:
        """
        Private function that acts as a helper for normal arithmetic operations for Matrix
        :param other: a Number or a Matrix
        :param func: must be a pure function that does the operation and returns a Number
        :return Matrix instance
        """
        if uL.isNumber(other):
            return Matrix([
                func(i, other) for i in self
            ])

        if isinstance(other, Vector):
            if uL.can_broadcast(self.shape, other.shape):
                pass

        if isinstance(other, self.__class__):

            if self.shape != other.shape:
                raise Exception('Two Matrices not of the same shape')

            # Creates a new vector and allocate memory
            new_matrix = Matrix([[0] * self.column] * self.row)

            for ind in range(self.row):
                new_matrix[ind] = func(self[ind], other[ind])

            return new_matrix

        raise Exception('Element type is not supported')

    def __element_condition_operator(self, other: Matrix, func: Callable[[NUMBER, NUMBER], NUMBER]) -> Matrix:
        """
        Private function that acts as a helper for one to one vector operations
        :param other: a Number or a Matrix # TODO or a vector ?
        :param func: must be a pure function that does the operation and returns a Number
        :return Matrix instance
        """
        pass

    def __element_operation(self, func: Callable[[Vector], NUMBER]) -> Matrix:
        """
        Private function that acts as a helper for one to one vector operations
        :param func: must be a pure function that does the operation and returns a Number
        :return Matrix instance
        """
        return Matrix([func(i) for i in self])


# TYPINGS
VECTOR_OR_NUMBER = Union[Vector, NUMBER]
VECTOR_OR_NUMBER_OR_ITERABLE = Union[Vector, NUMBER_OR_ITERABLE]
MATRIX_OR_NUMBER = Union[Matrix, NUMBER]
MATRIX_VECTOR_NUMBER = Union[Matrix, Vector, NUMBER]
MATRIX_OR_VECTOR = Union[Matrix, Vector]
MATRIX_OPERATOR_CALLBACK = Callable[[Vector, VECTOR_OR_NUMBER], Vector]

#
# s = Vector([True])
# ss1 = Vector([1, 2, 3])
# ss2 = Vector([3, 4, 5])
# ss3 = Vector([3, 4, 9])
# ss4 = Vector([3, 4])


# print(ss1)

# m = Matrix([[1.5, 2.5, 4.7], [3.7, 4.1, 5.1]])
# m2 = Matrix([Vector([1, 23]), Vector([23, 6])])
