from typing import List, Callable

import numpy as np


# naive implementations

def naive_relu(x: np.ndarray) -> np.ndarray:
    """
        naive relu for 2 dimensional np array
    """
    assert len(x.shape) == 2

    x = x.copy()  # pure function no side effects
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

    return x


def naive_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
       adds 2 dimensional np array
    """
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


def naive_add_matrix_and_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):  # gives the illusion of duplication :)
            x[i, j] += y[j]

    return x


def naive_vector_dot(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]

    return z


def naive_matrix_vector_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[i]

    return z


def naive_matrix_vector_dot1(x: np.ndarray, y: np.ndarray):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)  # Slicing picks up the current row
    return z


def naive_matrix_vector_dot2(x: np.ndarray, y: np.ndarray):
    """
    Mathematically speaking same thing as matrix multiplications
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)

    return z


def k_fold_validation(k: int,
                      num_epochs: int,
                      train_data: np.ndarray,
                      train_targets: np.ndarray,
                      build_fn: Callable,
                      logs=True
                      ) -> List:
    num_val_samples = len(train_data) // k
    all_scores = []

    for i in range(k):
        if logs:
            print('processing fold #', i)

        # prepare validation data from partitions #k
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # prepare the training data from all the other partitions

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]
             ],
            axis=0
        )

        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]
             ],
            axis=0
        )

        model = build_fn()

        return all_scores

    return


#

#

def vectorize_sequences(sequences: np.ndarray, dimension: int) -> np.ndarray:
    """
    Encoding the array into binary matrix
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# TODO what is the difference though ???
def to_one_hot(labels, dimension) -> np.ndarray:
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results
    #


def numpy_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape
    length = len(x.shape)

    param = list(x.shape)

    # for index, val in enumerate(param):
    #     param[index] = param[index] - 1

    param2 = [0] * length

    elements = (x, y, np.zeros(x.shape))

    print(param2)

    print(param)
    print(helper_numpy_add(elements, param, param2, 0))
    pass


# TODO turn this into iterator OR iterator data structure to work with any arbitrary data structure

def helper_numpy_add(elements: tuple, arr1: List, arr2: List, index) -> np.ndarray:
    """

    :param elements: represent the given two arrays and the result
    :param arr1: last point argument
    :param arr2: ongoing argument
    :param index: the current axis
    :return: np array
    """
    if np.equal(arr1, arr2):
        return elements[2]

    for i in range(arr1[index]):
        # access the correct index
        pass

    pass
