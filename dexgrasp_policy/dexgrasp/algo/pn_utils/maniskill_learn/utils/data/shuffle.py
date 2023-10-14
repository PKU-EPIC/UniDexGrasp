import random, numpy as np


def random_shuffle(x, axis=0):
    if isinstance(x, (list, tuple)):
        assert axis == 0
        random.shuffle(x)
        return x
    elif isinstance(x, np.ndarray):
        index = np.arange(x.shape[axis])
        np.random.shuffle(index)
        return np.take(x, index, axis=axis)
    else:
        raise NotImplementedError("")
