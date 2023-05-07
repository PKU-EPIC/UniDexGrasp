import os
import psutil


def format_memory_str(x, unit, number_only=False):
    unit_list = ['K', 'M', 'G', 'T']
    assert unit in unit_list
    unit_num = 1024 ** (unit_list.index(unit) + 1)
    if number_only:
        return x * 1.0 / unit_num
    else:
        return f"{x * 1.0 / unit_num:.2f}{unit}"


def get_total_memory(unit='G', number_only=False, init_pid=None):
    if init_pid is None:
        init_pid = os.getpid()
    process = psutil.Process(init_pid)
    ret = process.memory_info().rss
    for proc in process.children():
        ret += proc.memory_info().rss
    return format_memory_str(ret, unit, number_only)


def get_memory_list(unit='G', number_only=False, init_pid=None):
    if init_pid is None:
        init_pid = os.getpid()
    process = psutil.Process(init_pid)
    ret = [format_memory_str(process.memory_info().rss, unit, number_only), ]
    for proc in process.children():
        ret.append(format_memory_str(proc.memory_info().rss, unit, number_only))
    return ret


def get_memory_dict(unit='G', number_only=False, init_pid=None):
    if init_pid is None:
        init_pid = os.getpid()
    process = psutil.Process(init_pid)
    ret = {init_pid: format_memory_str(process.memory_info().rss, unit, number_only)}
    for i, proc in enumerate(process.children()):
        ret[proc.pid] = format_memory_str(proc.memory_info().rss, unit, number_only)
    return ret


def get_subprocess_ids(init_pid=None):
    if init_pid is None:
        init_pid = os.getpid()
    ret = [init_pid]
    process = psutil.Process(os.getpid())
    for proc in process.children():
        ret.append(proc.pid)
    return ret
