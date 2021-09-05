from __future__ import division
from collections import Counter

from my_linear_algebra import Vector
from util.typings import NUMBER


# TODO maybe make an inheritance with the Vector method

class Statistics:

    @staticmethod
    def mean(x: Vector) -> NUMBER:
        return sum(x) / len(x)

    @staticmethod
    def median(x: Vector) -> NUMBER:
        n = len(x)
        sorted_arr = sorted(x)
        mid_point = n // 2

        if n % 2 == 1:
            return sorted_arr[mid_point]

        lo = mid_point - 1
        hi = mid_point
        return (sorted_arr[lo] + sorted_arr[hi]) / 2

    @staticmethod
    def quantile(x: Vector, p: NUMBER) -> NUMBER:
        """
        :param x: a Vector of Numbers
        :param p: the percentile value
        :return pth-percentile value in x
        """
        p_index = int(p * len(x))
        return sorted(x)[p_index]

    @staticmethod
    def mode(x: Vector) -> list:
        counts = Counter(x)
        max_count = max(counts.values())
        return [x_i for x_i, count in counts.items()
                if count == max_count]

    @staticmethod
    def data_range(x: Vector) -> NUMBER:
        return max(x) - min(x)

    @staticmethod
    def de_mean(x: Vector) -> list:
        x_bar = Statistics.mean(x)
        return [x_i - x_bar for x_i in x]
