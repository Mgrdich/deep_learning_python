from typing import List, Iterable


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
    def isTensor(element) -> bool:
        return all(isinstance(ele, list) for ele in element)

    @staticmethod
    def isListEleSameLength(element: List) -> bool:
        initial_length = len(element[0])  # TODO not tested yet
        return all(initial_length == len(ele) for ele in element)
