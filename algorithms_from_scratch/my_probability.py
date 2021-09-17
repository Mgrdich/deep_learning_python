import math
from collections import Callable

from util.decorators import exception_decorators
from util.typings import NUMBER

exception_is_number: Callable = exception_decorators['is_number']


class Probability:

    @staticmethod
    @exception_is_number
    def uniform_pdf(x: NUMBER):
        return 1 if 0 <= x < 1 else 0

    @staticmethod
    @exception_is_number
    def uniform_cdf(x: NUMBER):
        if x < 0:
            return False
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    @exception_is_number
    def normal_pdf(x: NUMBER, mu=0, sigma=1):
        sqrt_two_pi: NUMBER = math.sqrt(2 * math.pi)
        return math.exp(-(x - mu) ** 2 / 2 * sigma ** 2) / (sqrt_two_pi * sigma)

    @staticmethod
    @exception_is_number
    def normal_cdf(x: NUMBER, mu=0, sigma=1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
