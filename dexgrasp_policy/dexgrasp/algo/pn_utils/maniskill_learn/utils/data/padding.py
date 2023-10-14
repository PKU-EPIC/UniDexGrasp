import numpy as np


def pad_or_clip(array, n, axis=0, pad_value=None, pad_index=None):
    assert pad_value is not None or pad_index is not None
    assert pad_value is None or pad_index is None

    if array.shape[axis] >= n:
        return np.take(array, range(n), axis=axis)
    if pad_value is not None:
        padded_shape = list(array.shape)
        padded_shape[axis] = n - array.shape[axis]
        pad = np.full(padded_shape, pad_value, dtype=array.dtype)
        return np.concatenate([array, pad], axis=axis)
    elif pad_value is not None:
        pad = np.repeat(np.take(array, range(1), axis=axis), n - array.shape[axis], axis=axis)
    else:
        raise ValueError("")
    return np.concatenate([array, pad], axis=0)

