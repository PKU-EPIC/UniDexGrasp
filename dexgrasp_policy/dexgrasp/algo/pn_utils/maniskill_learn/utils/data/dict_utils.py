from copy import deepcopy
import numpy as np
from .type import is_dict, is_seq_of


def update_dict(x, y):
    assert type(x) == type(y), f'{type(x), type(y)}'
    if is_dict(x):
        ret = deepcopy(x)
        for key in y:
            if key in x:
                ret[key] = update_dict(x[key], y[key])
            else:
                ret[key] = deepcopy(y[key])
    else:
        # Element
        ret = deepcopy(y)
    return ret


def update_dict_with_begin_keys(x, y, keys, begin=False, history_key=()):
    if len(keys) == 0:
        if type(x) == type(y):
            return update_dict(x, y)
        elif is_seq_of(x, dict) and is_dict(y):
            return [update_dict(_, y) for _ in x]
        else:
            raise NotImplementedError()
    if not is_dict(x):
        return deepcopy(x)

    ret = {}
    for key in x:
        if key == keys[0]:
            ret[key] = update_dict_with_begin_keys(x[key], y, keys[1:], True, history_key + (key, ))
        elif not begin:
            ret[key] = update_dict_with_begin_keys(x[key], y, keys, False, history_key + (key, ))
        else:
            ret[key] = deepcopy(x[key])
    return ret


def flatten_dict(x, title=''):
    """
    Convert a recursive dict to the dict with one layer.
    :param x:
    :param title:
    :return:
    """
    ret = {}
    for k in x:
        new_k = k if title == '' else f'{title}_{k}'
        if isinstance(x[k], dict):
            ret.update(flatten_dict(x[k], new_k))
        else:
            ret[new_k] = x[k]
    return ret


def dict_to_seq(x):
    keys = list(sorted(x.keys()))
    values = [x[k] for k in keys]
    return keys, values


def seq_to_dict(keys, values):
    return {keys[i]: values[i] for i in range(len(keys))}


def dict_to_str(x):
    ret = ''
    for key in x:
        if str != '':
            ret += " "
        if isinstance(x[key], (float, np.float32, np.float64)):
            ret += f'{key}: {x[key]:.3f}'
        else:
            ret += f'{key}: {x[key]}'
    return ret
