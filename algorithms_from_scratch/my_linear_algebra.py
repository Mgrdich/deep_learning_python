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

        self.__length = len(self.__local_vector)

    def __add__(self, other):
        print(other)
        pass

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.length:
            result = self.__local_vector[self.n]
            self.n += 1
            return result

    @property
    def length(self):
        return self.__length


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass
