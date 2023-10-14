import numpy as np, itertools
from .type import is_dict, is_seq_of


def concat_list_of_array(x, axis=0):
    assert is_seq_of(x)
    if len(x) == 0:
        return None
    if isinstance(x[0], dict):
        ret = {}
        for k in x[0].keys():
            ret[k] = concat_list_of_array([_[k] for _ in x], axis)
        return ret
    elif isinstance(x[0], (list, tuple)):
        ret = []
        for i in range(len(x[0])):
            ret.append(concat_list_of_array([_[i] for _ in x]))
        return type(x[0])(ret)
    elif isinstance(x[0], np.ndarray):
        if x[0].ndim == 0:
            return np.array(x)
        else:
            return np.concatenate(x, axis=axis)
    elif np.isscalar(x[0]):
        return np.array(x)
    else:
        import torch
        if is_seq_of(x, torch.Tensor):
            return torch.cat(x, dim=axis)
        else:
            raise NotImplementedError(x)


def concat_dict_of_list_array(x, axis=0):
    assert isinstance(x, dict)
    return {k: concat_list_of_array(x[k], axis) for k in x}


def stack_list_of_array(x, axis=0):
    assert is_seq_of(x), type(x)
    if len(x) == 0:
        return None
    if isinstance(x[0], dict):
        ret = {}
        for k in x[0].keys():
            ret[k] = stack_list_of_array([_[k] for _ in x], axis)
        return ret
    elif isinstance(x[0], np.ndarray):
        if x[0].ndim == 0:
            return np.array(x)
        else:
            return np.stack(x, axis=axis)
    elif np.isscalar(x[0]):
        if hasattr(x, 'dtype'):
            return np.array(x, dtype=x.dtype)
        else:
            return np.array(x)
    else:
        import torch
        if is_seq_of(x, torch.Tensor):
            return torch.stack(x, dim=axis)
        else:
            raise NotImplementedError(x)


def stack_dict_of_list_array(x, axis=0):
    assert isinstance(x, dict)
    return {k: stack_list_of_array(x[k], axis) for k in x}


def concat_seq(in_list, dtype=list):
    assert dtype in [list, tuple]
    return dtype(itertools.chain(*in_list))


def concat_list(in_list):
    return concat_seq(in_list, list)


def repeat_interleave(x, n, axis):
    if isinstance(x, np.ndarray):
        return np.repeat(x, n, axis=axis)
    elif isinstance(x, dict):
        return {key: repeat_interleave(x[key], n, axis=axis) for key in x}
    elif isinstance(x, (list, tuple)):
        return type(x)([repeat_interleave(_, n, axis=axis) for _ in x])
    else:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.repeat_interleave(x, n, dim=axis)
        else:
            raise NotImplementedError()
