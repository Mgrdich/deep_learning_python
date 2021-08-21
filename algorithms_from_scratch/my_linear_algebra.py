from __future__ import annotations

import math
from typing import Union, List, Callable
from util.Util_lib import Util_Lib as uL
from util.typings import NUMBER, ITERABLE


def vector_other_validation(function):
    def wrapper(*args):
        if not isinstance(args[1], Vector):
            raise Exception('The parameter Not Vector Type')

        if args[0].shape != args[1].shape:
            raise Exception('Two Vectors not of the same shape')

        return function(*args)

    return wrapper


class Vector:
    def __init__(self, iterator: ITERABLE):
        self.__local_vector: List = []
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

    def __add__(self, other: Union[int, float, Vector]) -> Vector:
        if uL.isNumber(other):
            return Vector([
                i + other for i in self
            ])

        if isinstance(other, self.__class__):

            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
                new_vector[ind] = self[ind] + other[ind]

            return new_vector

        raise Exception('Element type is not supported')

    def __sub__(self, other: Union[int, float, Vector]) -> Vector:
        if uL.isNumber(other):
            return Vector([
                i - other for i in self
            ])

        if isinstance(other, self.__class__):

            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
                new_vector[ind] = self[ind] - other[ind]

            return new_vector

        raise Exception('Element type is not supported')

    def __mul__(self, other: Union[int, float, Vector]) -> Vector:
        if uL.isNumber(other):
            return Vector([
                i * other for i in self
            ])

        if isinstance(other, self.__class__):

            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
                new_vector[ind] = self[ind] * other[ind]

            return new_vector

        raise Exception('Element type is not supported')

    def __truediv__(self, other: Union[int, float, Vector]) -> Vector:
        if uL.isNumber(other):
            return Vector([
                i / other for i in self
            ])

        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
                new_vector[ind] = self[ind] / other[ind]

            return new_vector

        raise Exception('Element type is not supported')

    def __floordiv__(self, other: Union[int, float, Vector]) -> Vector:
        if uL.isNumber(other):
            return Vector([
                i // other for i in self
            ])

        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise Exception('Two Vectors not of the same shape')

            # Creates a new vector and allocate memory
            new_vector = Vector([0] * self.shape)

            for ind in range(self.shape):
                new_vector[ind] = self[ind] // other[ind]

            return new_vector

        raise Exception('Element type is not supported')

    @vector_other_validation
    def __eq__(self, other: Vector) -> Vector:
        return Vector([self[i] == other[i] for i in range(self.shape)])

    @vector_other_validation
    def __ne__(self, other: Vector) -> Vector:
        return Vector([self[i] != other[i] for i in range(self.shape)])

    @vector_other_validation
    def __gt__(self, other: Vector) -> Vector:
        return Vector([self[i] > other[i] for i in range(self.shape)])

    @vector_other_validation
    def __ge__(self, other: Vector) -> Vector:
        return Vector([self[i] >= other[i] for i in range(self.shape)])

    @vector_other_validation
    def __lt__(self, other: Vector) -> Vector:
        return Vector([self[i] < other[i] for i in range(self.shape)])

    @vector_other_validation
    def __le__(self, other: Vector) -> Vector:
        return Vector([self[i] <= other[i] for i in range(self.shape)])

    def __ceil__(self) -> Vector:
        return Vector([math.ceil(i) for i in self])

    def __floor__(self) -> Vector:
        return Vector([math.floor(i) for i in self])

    def __getitem__(self, key: int) -> NUMBER:
        if key >= self.shape:
            raise Exception('Vector out of bounds')

        return self.__local_vector[key]

    def __setitem__(self, key: int, value: NUMBER):
        if key >= self.shape:
            raise Exception('Vector out of bounds')

        self.__local_vector[key] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.__length:
            result = self.__local_vector[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.shape

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

    @property
    def shape(self) -> int:
        return self.__length

    @property
    def vector(self) -> List:
        return self.__local_vector

    @property
    def sum_of_squares(self) -> NUMBER:
        return self.dot(self)

    @property
    def magnitude(self) -> NUMBER:
        return math.sqrt(self.sum_of_squares)

    def __element_operator(self, other: Union[int, float, Vector], func: Callable[[NUMBER, NUMBER], NUMBER]):
        """
        Private function that acts as a helper for normal arithmetic operations
        :param other: a Number or a Vector
        :param func: must be a pure function that does the operation and returns a Number
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


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass


# TYPINGS
VECTOR_NUMBER = Union[Vector, NUMBER]

# s = Vector([True])
# ss1 = Vector([1, 2, 3])
# ss2 = Vector([3, 4, 5])
# ss3 = Vector([3, 4, 9])
# ss4 = Vector([3, 4])
