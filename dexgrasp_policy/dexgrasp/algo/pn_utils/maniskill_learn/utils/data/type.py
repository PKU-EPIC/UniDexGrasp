from numbers import Number
from collections.abc import Sequence
import numpy as np


def scalar_type(x):
    if isinstance(x, Sequence):
        x = x[0]
    if hasattr(x, 'dtype'):
        return x.dtype
    else:
        return type(x)


def get_str_dtype(x):
    assert is_arr(x)
    # torch: torch.float32, numpy: float32
    return str(x.dtype()).split('.')[0]


def str_to_dtype(x, arr_type='np'):
    assert arr_type in ['np', 'torch']
    if arr_type == 'torch':
        import torch
        return getattr(torch, x)
    elif arr_type == 'np':
        return x
    else:
        raise NotImplementedError(f"str_to_dtype {type(x), arr_type}")


def is_str(x):
    return isinstance(x, str)


def is_num(x):
    return isinstance(x, Number)


def is_type(x):
    return isinstance(x, type)


def is_arr(x, arr_type=None):
    if arr_type is None:
        import torch
        arr_type = (np.ndarray, torch.Tensor)
    elif is_str(arr_type):
        if arr_type in ['np', 'numpy']:
            arr_type = np.ndarray
        elif arr_type == 'torch':
            import torch
            arr_type = torch.Tensor
    return isinstance(x, arr_type)


def is_seq_of(seq, expected_type=None, seq_type=None):
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    if expected_type:
        for item in seq:
            if not isinstance(item, expected_type):
                return False
    return True


def is_list_of(seq, expected_type=None):
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type=None):
    return is_seq_of(seq, expected_type, seq_type=tuple)


def is_dict(x, expected_type=None):
    if not isinstance(x, dict):
        return False
    if expected_type:
        for key in x:
            if not isinstance(x[key], expected_type):
                return False
    return True
