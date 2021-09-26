import math
import random
from collections import Callable

from util.Util_lib import Util_Lib as uL

from util.decorators import exception_decorators
from util.typings import NUMBER

exception_is_number: Callable = exception_decorators['is_number']


class Probability:

    @staticmethod
    @exception_is_number
    def uniform_pdf(x: NUMBER) -> NUMBER:
        return 1 if 0 <= x < 1 else 0

    @staticmethod
    @exception_is_number
    def uniform_cdf(x: NUMBER) -> NUMBER:
        if x < 0:
            return False
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    @exception_is_number
    def normal_pdf(x: NUMBER, mu=0, sigma=1) -> NUMBER:
        sqrt_two_pi: NUMBER = math.sqrt(2 * math.pi)
        return math.exp(-(x - mu) ** 2 / 2 * sigma ** 2) / (sqrt_two_pi * sigma)

    @staticmethod
    @exception_is_number
    def normal_cdf(x: NUMBER, mu=0, sigma=1) -> NUMBER:
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

    @staticmethod
    def bernoulli_trial(p: NUMBER) -> NUMBER:
        if not uL.is_probability(p):
            raise Exception('p is no in probability form')
        return 1 if random.random() < p else 0

    @staticmethod
    def binomial(n: int, p: NUMBER) -> NUMBER:
        if not uL.is_integer(n):
            raise Exception('n should be an integer')

        if not uL.is_probability(p):
            raise Exception('p is no in probability form')

        return sum(Probability.bernoulli_trial(i) for i in range(n))
