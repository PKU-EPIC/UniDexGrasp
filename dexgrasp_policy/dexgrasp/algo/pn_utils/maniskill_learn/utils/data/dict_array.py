"""
Dict array is a recursive dict array (numpy or torch.Tensor).
Example:
x = {'a': [1, 0, 0], 'b': {'c': {'d': [1, 0, 0]}}}

In the replay buffer, each key with non-dict value is a
np.ndarray with shape == [capacity, *(element_shape)]
(if element is scalar, then shape == [capacity])   
"""

import numpy as np
from .string_utils import h5_name_format
from h5py import File, Group, Dataset


def recursive_init_dict_array(memory, kwargs, capacity, begin_index=0):
    """
    Init a dict array structure, if not done already, that contains all the structure in kwargs.
    :param memory:
    :param kwargs:
    :param capacity:
    :param begin_index:
    :return:
    """
    if isinstance(kwargs, np.ndarray):
        if memory is None:
            memory = np.empty((capacity,) + tuple(kwargs.shape), dtype=kwargs.dtype)
            memory[:begin_index] = 0
        return memory
    elif np.isscalar(kwargs):
        if memory is None:
            memory = np.empty(capacity, dtype=np.dtype(type(kwargs)))
            memory[:begin_index] = 0
        return memory
    elif isinstance(kwargs, (list, tuple)):
        if len(kwargs) == 0:
            kwargs = np.zeros(0, np.float32)
        else:
            kwargs = np.array(kwargs, dtype=np.dtype(type(kwargs[0])))
        if memory is None:
            memory = np.empty((capacity,) + tuple(kwargs.shape), dtype=kwargs.dtype)
            memory[:begin_index] = 0
        return memory
    assert isinstance(kwargs, dict), kwargs
    if memory is None:
        memory = {}
    for key in kwargs:
        memory[key] = recursive_init_dict_array(memory.get(key, None), kwargs[key], capacity, begin_index)
    return memory


def map_func_to_dict_array(memory, func, *args, **kwargs):
    if not isinstance(memory, (dict, File, Group)):
        return func(memory, *args, **kwargs)
    ret = {}
    for key in memory:
        ret[key] = map_func_to_dict_array(memory[key], func, *args, **kwargs)
    return ret


def sample_element_in_dict_array(memory, index):
    func = lambda _, __: _[__]
    return map_func_to_dict_array(memory, func, index)


def assign_single_element_in_dict_array(memory, index, value):
    if not isinstance(memory, dict):
        # memory = np.ndarray with shape [capacity, *(value_shape)]
        if isinstance(value, np.ndarray):
            memory[index] = value.copy()
        else:
            # value is scalar
            memory[index] = value
        return
    for key in memory:
        if key in value:
            assign_single_element_in_dict_array(memory[key], index, value[key])


def store_dict_array_to_h5(memory, file):
    assert isinstance(memory, dict)
    for key in memory:
        if isinstance(memory[key], np.ndarray):
            file.create_dataset(h5_name_format(key), memory[key].shape, dtype=memory[key].dtype, data=memory[key])
        else:
            assert isinstance(memory[key], dict)
            file.create_group(h5_name_format(key))
            store_dict_array_to_h5(memory[key], file[key])


def split_in_dict_array(memory, batch_size, axis=0):
    if not isinstance(memory, dict):
        length = memory.shape[axis]
        max_num = (length + batch_size - 1) // batch_size
        return [memory[batch_size * i: min(batch_size * (i + 1), length)] for i in range(max_num)]
    ret = {}
    max_num = None
    for key in memory:
        ret[key] = split_in_dict_array(memory[key], batch_size, axis=axis)
        if max_num is None:
            max_num = len(ret[key])
        else:
            assert max_num == len(ret[key])
    ret_list = []
    for i in range(max_num):
        item_i = {}
        for key in ret:
            item_i[key] = ret[key][i]
        ret_list.append(item_i)
    return ret_list
