from __future__ import division

import math
from collections import Counter

from my_linear_algebra import Vector
from util.typings import NUMBER


# TODO turn it with inheritance
class Statistics:

    @staticmethod
    def mean(x: Vector) -> NUMBER:
        return sum(x) / len(x)

    @staticmethod
    def median(x: Vector) -> NUMBER:
        n: int = len(x)
        sorted_arr: list = sorted(x)
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
    def mode(x: Vector) -> Vector:
        counts: dict = Counter(x)
        max_count: NUMBER = max(counts.values())
        return Vector([x_i for x_i, count in counts.items()
                       if count == max_count])

    @staticmethod
    def data_range(x: Vector) -> NUMBER:
        return max(x) - min(x)

    @staticmethod
    def de_mean(x: Vector) -> Vector:
        """
        translate x by subtracting its mean so the (result has a mean 0)
        :param x is a Vector
        :return result Vector
        """
        x_bar: NUMBER = Statistics.mean(x)
        return Vector([x_i - x_bar for x_i in x])

    @staticmethod
    def variance(x: Vector) -> NUMBER:
        n: int = len(x)
        deviations: Vector = Statistics.de_mean(x)
        return deviations.sum_of_squares() / (n - 1)

    @staticmethod
    def standard_deviation(x: Vector) -> NUMBER:
        return math.sqrt(Statistics.variance(x))

    @staticmethod
    def interquartile_range(x: Vector) -> NUMBER:
        return Statistics.quantile(x, .75) - Statistics.quantile(x, .25)

    @staticmethod
    def covariance(x: Vector, y: Vector) -> NUMBER:
        n: int = len(x)
        return Statistics.de_mean(x).dot(Statistics.de_mean(y)) / (n - 1)

    @staticmethod
    def correlation(x: Vector, y: Vector):
        stdev_x: NUMBER = Statistics.standard_deviation(x)
        stdev_y: NUMBER = Statistics.standard_deviation(y)

        if stdev_x > 0 and stdev_y > 0:
            return Statistics.covariance(x, y) / stdev_x / stdev_y

        return 0
