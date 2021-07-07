import numpy


def naive_relu(x: numpy.ndarray) -> numpy.ndarray:
    assert len(x.shape) == 2

    x = x.copy()  # pure function no side effects
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

    return x


# def naive_add(x, y):
