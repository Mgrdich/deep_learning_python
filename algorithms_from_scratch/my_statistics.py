from __future__ import division, annotations

import math
from collections import Counter

from my_linear_algebra import Vector
from util.typings import NUMBER, ITERABLE


class Statistics(Vector):
    def __init__(self, iterator: ITERABLE):
        super().__init__(iterator)

    def mean(self) -> NUMBER:
        return sum(self) / len(self)

    def median(self) -> NUMBER:
        n: int = len(self)
        sorted_arr: list = sorted(self)
        mid_point = n // 2

        if n % 2 == 1:
            return sorted_arr[mid_point]

        lo = mid_point - 1
        hi = mid_point
        return (sorted_arr[lo] + sorted_arr[hi]) / 2

    def quantile(self, p: NUMBER) -> NUMBER:
        """
        :param p: the percentile value
        :return pth-percentile value in x
        """
        p_index = int(p * len(self))
        return sorted(self)[p_index]

    def mode(self) -> Vector:
        counts: dict = Counter(self)
        max_count: NUMBER = max(counts.values())
        return Vector([x_i for x_i, count in counts.items()
                       if count == max_count])

    def data_range(self) -> NUMBER:
        return max(self) - min(self)

    def de_mean(self) -> Vector:
        """
        translate x by subtracting its mean so the (result has a mean 0)
        :return result Vector
        """
        x_bar: NUMBER = self.mean()
        return Vector([x_i - x_bar for x_i in self])

    def variance(self) -> NUMBER:
        n: int = len(self)
        deviations: Vector = self.de_mean()
        return deviations.sum_of_squares() / (n - 1)

    def standard_deviation(self) -> NUMBER:
        return math.sqrt(self.variance())

    def interquartile_range(self) -> NUMBER:
        return self.quantile(.75) - self.quantile(.25)

    def covariance(self, y: Statistics) -> NUMBER:
        n: int = len(self)
        return self.de_mean().dot(y.de_mean()) / (n - 1)

    def correlation(self, y: Statistics):
        stdev_x: NUMBER = self.standard_deviation()
        stdev_y: NUMBER = y.standard_deviation()

        if stdev_x > 0 and stdev_y > 0:
            return self.covariance(y) / stdev_x / stdev_y

        return 0
