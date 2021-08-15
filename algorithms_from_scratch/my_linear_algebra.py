from typing import Union, List

from util.Util_lib import Util_Lib as uL


class Vector:
    def __init__(self, iterator: Union[List, tuple, dict]):
        self.__local_vector = []
        self.__length = 0

        if not uL.isIterable(iterator):
            raise Exception('Parameter must by an iterator object')

        if uL.isDictionary(iterator):
            iterator = iterator.values()

        iterator = iterator.copy()

        # TODO check all the elements are numbers
        for elem in iterator:
            if not uL.isNumber(elem):
                raise Exception('Not all parameters are Numbers')

            self.__local_vector.append(elem)

        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    def __add__(self, other):
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

    def __sub__(self, other):
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

    def __mul__(self, other):
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

    def __getitem__(self, key: int) -> Union[int, float]:
        if key >= self.shape:
            raise Exception('Vector out of bounds')

        return self.__local_vector[key]

    def __setitem__(self, key: int, value: Union[int, float]):
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

    def dot(self, other) -> Union[int, float]:
        if isinstance(other, self.__class__):
            # TODO check the way one should deal with extra variables
            return sum(self * other)

        raise Exception('Element type is not supported')

    @property
    def shape(self):
        return self.__length

    @property
    def vector(self):
        return self.__local_vector


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass
