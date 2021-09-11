from collections import Callable
from functools import wraps
from util.Util_lib import Util_Lib as uL

exception_decorators = dict()


def register_exception_decorators(function: Callable):
    @wraps(function)
    def wrapper(*args, **kwargs):
        exception_decorators['exc_' + function.__name__] = function
        return function(*args, **kwargs)

    return wrapper


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
