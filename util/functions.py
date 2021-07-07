import numpy


def naive_relu(x: numpy.ndarray) -> numpy.ndarray:
    """
        naive relu for 2 dimensional numpy array
    """
    assert len(x.shape) == 2

    x = x.copy()  # pure function no side effects
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

    return x


def naive_add(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """
       adds 2 dimensional numpy array
    """
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


def numpy_add(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    assert x.shape == y.shape
    print(x.shape)
    pass


def helper_numpy_add():
    pass
