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

        for elem in iterator:
            self.__local_vector.append(elem)

        # len() is O(1) operation
        self.__length = len(self.__local_vector)

    def __add__(self, element):
        if uL.isNumber(element):
            return Vector([
                i + element for i in self
            ])

        print(self.__class__)
        if isinstance(element, self.__class__):
            print('Yeah', self.__class__)

        raise Exception('Element type is not supported')

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

    @property
    def shape(self):
        return self.__length


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass
