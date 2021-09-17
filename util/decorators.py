from collections import Callable
from functools import wraps
from util.Util_lib import Util_Lib as uL

exception_decorators = dict()


def register_exception_decorators(function: Callable):
    if function.__name__ in exception_decorators:
        raise Exception('Function with this name has been registered')

    exception_decorators[function.__name__] = function
    return function


@register_exception_decorators
def is_number(is_method: bool):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if is_method and len(args) == 2:
                element = args[1]
            elif len(args) == 1:
                element = args[0]
            else:
                raise Exception('The function needs a parameter')

            if not uL.isNumber(element):
                raise Exception('Not a Number')

            return function(*args, **kwargs)

        return wrapper

    return decorator
