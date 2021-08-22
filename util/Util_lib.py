from typing import Iterable


class Util_Lib:

    @staticmethod
    def isList(element) -> bool:
        return isinstance(element, list)

    @staticmethod
    def isTuple(element) -> bool:
        return isinstance(element, tuple)

    @staticmethod
    def isDictionary(element) -> bool:
        return isinstance(element, dict)

    @staticmethod
    def isIterable(element) -> bool:
        return isinstance(element, Iterable)

    @staticmethod
    def isFloat(element) -> bool:
        return isinstance(element, float)

    @staticmethod
    def isInteger(element) -> bool:
        return isinstance(element, int)

    @staticmethod
    def isNumber(element) -> bool:
        return Util_Lib.isInteger(element) or Util_Lib.isFloat(element)

    @staticmethod
    def isTensor(element) -> bool:
        return all(isinstance(ele, list) for ele in element)

    @staticmethod
    def isNone(element) -> bool:
        return element is None

    @staticmethod
    def isListEleSameLength(element: list) -> bool:
        initial_length = len(element[0])  # TODO not tested yet
        return all(initial_length == len(ele) for ele in element)
