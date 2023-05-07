import torch, psutil, os
from torch.cuda import _get_device_index as get_device_index
from ..meta.process_utils import get_subprocess_ids, format_memory_str
from algo.pn_utils.maniskill_learn.utils.data import is_dict, is_seq_of


try:
    import pynvml
    from pynvml import NVMLError_DriverNotLoaded
except ModuleNotFoundError:
    print ("pynvml module not found, please install pynvml")
    exit(0)

try:
    pynvml.nvmlInit()
except NVMLError_DriverNotLoaded:
    print("cuda driver can't be loaded, is cuda enabled?")
    exit(0)


def get_gpu_memory_info(device, unit='G', number_only=False):
    device = get_device_index(device, optional=True)
    handler = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = format_memory_str(meminfo.total, unit, number_only)
    used = format_memory_str(meminfo.used, unit, number_only)
    free = format_memory_str(meminfo.free, unit, number_only)
    ratio = meminfo.used / meminfo.total
    ratio = ratio * 100 if number_only else f'{ratio * 100:.1f}%'
    return total, used, free, ratio


def get_gpu_memory_usage_by_process(process, device=None, unit='G', number_only=False):
    if not isinstance(process, (list, tuple)):
        process = [process]
    device = get_device_index(device, optional=True)
    handler = pynvml.nvmlDeviceGetHandleByIndex(device)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handler)
    mem = 0
    for p in procs:
        if p.pid in process:
            mem += p.usedGpuMemory
    return format_memory_str(mem, unit, number_only)


def get_gpu_memory_usage_by_current_program(device=None, unit='G', number_only=False):
    proc_in_current_program = get_subprocess_ids()
    return get_gpu_memory_usage_by_process(proc_in_current_program)


def get_gpu_utilization(device=None):
    device = get_device_index(device, optional=True)
    handler = pynvml.nvmlDeviceGetHandleByIndex(device)
    return pynvml.nvmlDeviceGetUtilizationRates(handler).gpu


def get_cuda_info(device=None, unit='G', number_only=True):
    current_mem = get_gpu_memory_usage_by_current_program(device, unit, number_only)
    all_mem, used, _, ratio = get_gpu_memory_info(device, unit, number_only)
    utilization = get_gpu_utilization(device)
    return {
        'gpu_mem_ratio': ratio,
        'gpu_mem': used,
        'gpu_mem_this': current_mem,
        'gpu_util': utilization if number_only else f'{utilization}%'
    }


def get_one_device(x):
    if is_dict(x):
        return get_one_device(x[list(x.keys())[0]])
    elif is_seq_of(x):
        return get_one_device(x[0])
    else:
        assert hasattr(x, 'device'), type(x)
        return x.device


def get_device(x):
    if is_dict(x):
        return {k: get_device(x[k]) for k in x}
    elif is_seq_of(x):
        return type(x)([get_device(y) for y in x])
    else:
        assert hasattr(x, 'device'), type(x)
        return x.device
