from typing import Iterable

from util.typings import NUMBER


class Util_Lib:

    @staticmethod
    def is_list(element) -> bool:
        return isinstance(element, list)

    @staticmethod
    def is_tuple(element) -> bool:
        return isinstance(element, tuple)

    @staticmethod
    def is_dictionary(element) -> bool:
        return isinstance(element, dict)

    @staticmethod
    def is_iterable(element) -> bool:
        return isinstance(element, Iterable)

    @staticmethod
    def is_float(element) -> bool:
        return isinstance(element, float)

    @staticmethod
    def is_integer(element) -> bool:
        return isinstance(element, int)

    @staticmethod
    def isNumber(element) -> bool:
        return Util_Lib.is_integer(element) or Util_Lib.is_float(element)

    @staticmethod
    def is_negative(element: NUMBER) -> bool:
        return element < 0

    @staticmethod
    def is_positive(element: NUMBER) -> bool:
        return element > 0

    @staticmethod
    def is_tensor(element) -> bool:
        return all(isinstance(ele, list) for ele in element)

    @staticmethod
    def is_none(element) -> bool:
        return element is None

    @staticmethod
    def is_probability(element) -> bool:
        return Util_Lib.isNumber(element) and (0 <= element <= 1)

    @staticmethod
    def can_broadcast(shape: tuple, other_shape: tuple) -> bool:
        pass
