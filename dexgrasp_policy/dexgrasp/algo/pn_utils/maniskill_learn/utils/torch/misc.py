from functools import wraps
import numpy as np
import torch
from algo.pn_utils.maniskill_learn.utils.data import split_in_dict_array, concat_list_of_array


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed. Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def no_grad(f):
    wraps(f)

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return wrapper


def run_with_mini_batch(function, data, batch_size):
    """
    Run a pytorch function with mini-batch when the batch size of dat is very large.
    :param function: the function
    :param data: the input data which should be in dict array structure
    :param batch_size: the batch_size of the mini-batch
    :return: all the outputs.
    """
    data_list = split_in_dict_array(data, batch_size, axis=0)
    ans = []
    for data_i in data_list:
        ans_i = function(data_i)
        ans.append(ans_i)
    return concat_list_of_array(ans, axis=0)

