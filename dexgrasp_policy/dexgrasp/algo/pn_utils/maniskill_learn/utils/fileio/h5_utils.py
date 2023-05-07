import os
import os.path as osp
import time

import h5py
import numpy as np
from tqdm import tqdm

from .hash_utils import md5sum
from .serialization import dump
from ..data import (h5_name_deformat, get_one_shape, concat_list, sample_element_in_dict_array,
                    recursive_init_dict_array, store_dict_array_to_h5)
from ..math import split_num
from ..meta import get_total_memory


def load_h5_as_dict_array(h5):
    if isinstance(h5, str):
        if not osp.exists(h5):
            return []
        open_h5 = True
        h5 = h5py.File(h5, 'r')
    else:
        open_h5 = False
    if isinstance(h5, h5py.Dataset):
        return h5[()]
    elif isinstance(h5, h5py.Group):
        ret = {}
        for key in sorted(h5.keys()):
            ret[h5_name_deformat(key)] = load_h5_as_dict_array(h5[key])
    else:
        raise NotImplementedError("")
    if open_h5:
        h5.close()
    return ret


def load_h5s_as_list_dict_array(h5):
    dict_array = load_h5_as_dict_array(h5)
    ret = []
    for key in dict_array:
        ret.append(dict_array[key])
    return ret


def merge_h5_trajectory(h5_files, output_name):
    with h5py.File(output_name, 'w') as f:
        index = 0
        for h5_file in h5_files:
            h5 = h5py.File(h5_file, 'r')
            num = len(h5.keys())
            for i in range(num):
                h5.copy(f'traj_{i}', f, f'traj_{index}')
                index += 1
        print(f'Total number of trajectories {index}')


def generate_chunked_h5_replay(h5_files, name, folder, num_files):
    os.makedirs(folder, exist_ok=True)

    total_size = 0
    item = None
    from tqdm import tqdm
    print('Compute total size of all datasets.')
    for file in tqdm(h5_files):
        trajs = load_h5_as_dict_array(file)
        for key in trajs:
            total_size += get_one_shape(trajs[key])[0]
            if item is None:
                item = sample_element_in_dict_array(trajs[key], 0)
    print(f'Total size of dataset: {total_size}')
    num_files, size_per_file = split_num(total_size, num_files)
    h5_names = [osp.join(folder, f'{name}_{i}.h5') for i in range(num_files)]
    h5s = [h5py.File(h5_names[i], 'w') for i in range(num_files)]
    for i in range(num_files):
        memory = {}
        recursive_init_dict_array(memory, item, size_per_file[i])
        store_dict_array_to_h5(memory, h5s[i])

    indices = concat_list([[i for __ in range(_)] for i, _ in enumerate(size_per_file)])
    h5_index = [0 for i in range(num_files)]
    np.random.shuffle(indices)

    def assign_single_element_in_array(memory, index, value):
        if not isinstance(memory, h5py.Group):
            memory[index] = value
            return
        for key in memory:
            if key in value:
                assign_single_element_in_array(memory[key], index, value[key])

    pbar = tqdm(total=total_size)
    start_time = time.time()
    cnt = 0
    for file in h5_files:
        trajs = load_h5_as_dict_array(file)
        for key in trajs:
            batch_size = get_one_shape(trajs[key])[0]
            for i in range(batch_size):
                item = sample_element_in_dict_array(trajs[key], i)
                h5_file_index = indices[cnt]
                assign_single_element_in_array(h5s[h5_file_index], h5_index[h5_file_index], item)
                h5_index[h5_file_index] += 1
                cnt += 1
                pbar.update(1)

    for i in range(num_files):
        if h5_index[i] != size_per_file[i]:
            print('Wrong', i, h5_index[i], size_per_file[i], h5s[i][()].shape)
            exit(0)
        else:
            h5s[i].flush()
            h5s[i].close()
    md5_sums = [md5sum(_) for _ in h5_names]
    # We will store num of files, size of files and md5 for each file.
    dump([num_files, size_per_file, md5_sums], osp.join(folder, 'index.pkl'))
    print('Time & Memory', time.time() - start_time, get_total_memory())
