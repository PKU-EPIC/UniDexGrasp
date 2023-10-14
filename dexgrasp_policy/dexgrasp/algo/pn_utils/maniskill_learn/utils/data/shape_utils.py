import numpy as np
from h5py import File, Group

from .type import is_seq_of


def get_one_shape(x):
    if isinstance(x, (dict, Group, File)):
        return get_one_shape(x[list(x.keys())[0]])
    elif is_seq_of(x):
        return get_one_shape(x[0])
    else:
        assert hasattr(x, 'shape'), type(x)
        if len(x.shape) == 1:
            return x.shape[0]
        else:
            return x.shape


def get_shape(x):
    if isinstance(x, (dict, Group, File)):
        return {k: get_shape(x[k]) for k in x}
    elif is_seq_of(x):
        return type(x)([get_shape(y) for y in x])
    elif np.isscalar(x):
        return 1
    else:
        assert hasattr(x, 'shape'), type(x)
        if len(x.shape) == 1:
            return x.shape[0]
        else:
            return x.shape


def get_shape_and_type(x):
    if isinstance(x, (dict, Group, File)):
        return {k: get_shape_and_type(x[k]) for k in x}
    elif is_seq_of(x):
        return type(x)([get_shape_and_type(y) for y in x])
    elif np.isscalar(x):
        return 1, type(x)
    else:
        assert hasattr(x, 'shape'), type(x)
        if len(x.shape) == 1:
            return x.shape[0], x.dtype
        else:
            return x.shape, x.dtype


def unsqueeze(x, axis=0):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=axis)
    elif isinstance(x, dict):
        return {key: unsqueeze(x[key], axis=axis) for key in x}
    elif isinstance(x, (list, tuple)):
        return type(x)([unsqueeze(_, axis=axis) for _ in x])
    else:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.unsqueeze(x, dim=axis)
        else:
            raise NotImplementedError()


def reshape(x, target_shape):
    if isinstance(x, np.ndarray):
        return x.reshape(*target_shape)
    elif isinstance(x, dict):
        return {key: reshape(x[key], target_shape) for key in x}
    elif isinstance(x, (list, tuple)):
        return type(x)([reshape(_, target_shape) for _ in x])
    else:
        import torch
        if isinstance(x, torch.Tensor):
            return x.reshape(*target_shape)
        else:
            raise NotImplementedError()
