from typing import Union, List

from util.Util_lib import Util_Lib as uL


class Vector:
    __local_vector = []
    __length = 0

    def __init__(self, iterator: Union[List, tuple, dict]):
        if not uL.isIterable(iterator):
            raise Exception('Parameter must by an iterator object')

        if uL.isDictionary(iterator):
            iterator = iterator.values()

        # TODO check all the elements are numbers
        for elem in iterator:
            self.__local_vector.append(elem)

        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    def __add__(self, element):
        if uL.isNumber(element):
            return Vector([
                i + element for i in self
            ])

        if isinstance(element, self.__class__):
            v, iterator = (element, self) if element.shape > self.shape else (self, element)
            # memory allocate size of the biggest
            print('local Vector', v.__local_vector.copy())
            new_vector = Vector(v.__local_vector.copy())

            for ind in range(iterator.shape):
                new_vector[ind] = self[ind] + element[ind]

            return new_vector

        raise Exception('Element type is not supported')

    def __sub__(self, other):
        if uL.isNumber(other):
            return Vector([
                i - other for i in self
            ])

        if isinstance(other, self.__class__):
            iterator = other if other.shape > self.shape else self
            new_vector = Vector([0] * iterator.shape)  # memory allocate

            for ind in range(iterator.shape):
                new_vector[ind] = self[ind] - other[ind]

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

    @property
    def shape(self):
        return self.__length


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass
