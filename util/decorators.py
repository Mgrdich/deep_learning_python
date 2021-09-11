from collections import Callable
from functools import wraps
from util.Util_lib import Util_Lib as uL

exception_decorators = dict()


def register_exception_decorators(function: Callable):
    # TODO add plugin unique name error
    exception_decorators[function.__name__] = function
    return function


@register_exception_decorators
def is_number(is_method: bool):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if is_method:
                element = args[1]
            else:
                element = args[0]

            if not uL.isNumber(element):
                raise Exception('Not a Number')

            return function(*args, **kwargs)

        return wrapper

    return decorator
