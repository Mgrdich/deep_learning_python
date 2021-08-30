from __future__ import annotations

import math
from typing import Union, Callable, List, Tuple
from util.Util_lib import Util_Lib as uL
from util.typings import NUMBER, ITERABLE


def vector_other_validation(function: Callable):
    def wrapper(*args):
        if not isinstance(args[1], Vector):
            raise Exception('The parameter Not Vector Type')

        if args[0].shape != args[1].shape:
            raise Exception('Two Vectors not of the same shape')

        return function(*args)

    return wrapper


def matrix_other_validation(function: Callable):
    def wrapper(*args):
        return function(*args)

    return wrapper


class Vector:
    def __init__(self, iterator: ITERABLE):
        self.__local_vector: list = []
        self.__length: int = 0

        if not uL.isIterable(iterator):
            raise Exception('Parameter must by an iterator object')

        if uL.isDictionary(iterator):
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

    def __getitem__(self, key: int) -> NUMBER:
        if key >= self.shape:
            raise Exception('Vector out of bounds')

        return self.__local_vector[key]

    def __setitem__(self, key: int, value: NUMBER):
        if key >= self.shape:
            raise Exception('Vector out of bounds')

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
        return self.shape

    def __str__(self) -> str:
        return "Vector({0})".format(self.__local_vector)

    def __repr__(self) -> str:
        return self.__str__()

    @vector_other_validation
    def dot(self, other: Vector) -> NUMBER:
        return sum(self * other)

    @vector_other_validation
    def sum_of_squares_with(self, other: Vector) -> NUMBER:
        return self.dot(other)

    @vector_other_validation
    def squared_distance(self, other: Vector) -> NUMBER:
        return self.sum_of_squares_with(self - other)

    @vector_other_validation
    def distance(self, other: Vector) -> NUMBER:
        return math.sqrt(self.sum_of_squares_with(other))

    # TODO rename the property to something else and make shape return tuple
    @property
    def shape(self) -> int:
        return self.__length

    @property
    def shape_tuple(self) -> tuple:
        return self.shape,

    @property
    def sum_of_squares(self) -> NUMBER:
        return self.dot(self)

    @property
    def magnitude(self) -> NUMBER:
        return math.sqrt(self.sum_of_squares)

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
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
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
        return Vector([func(self[i], other[i]) for i in range(self.shape)])

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

            if not uL.isNone(last_vector_shape) and vector_element.shape != last_vector_shape:
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
            if uL.canBroadcast(self.shape, other.shape_tuple):
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

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @property
    def row(self) -> int:
        return self.__shape[0]

    @property
    def column(self) -> int:
        return self.__shape[1]


# TYPINGS
VECTOR_OR_NUMBER = Union[Vector, NUMBER]
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
