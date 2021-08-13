from typing import Union, List


class Vector:
    __local_vector = []

    def __init__(self, iterator: Union[List, tuple, dict]):


        for elem in iterator:
            self.__local_vector.append(elem)
        pass

    def __add__(self, other):
        pass

    def __next__(self):
        pass


class Matrix:
    def __init__(self):
        pass

    def __add__(self, other):
        pass
