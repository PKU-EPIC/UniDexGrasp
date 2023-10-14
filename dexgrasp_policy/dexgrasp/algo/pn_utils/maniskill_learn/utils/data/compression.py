import numpy as np


def to_f32(x):
    if np.isscalar(x):
        return np.float32(x)
    else:
        return x.astype(np.float32)


def compress_image(x, depth=False):
    # Image only
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    if x.dtype == np.float32:
        if not depth:
            x = np.clip(x, 0, 1)
            x = (x * 255).astype(np.uint8)
            return x
        else:
            x = (x * 1000).astype(np.uint16)
            return x
    return x


def decompress_image(x):
    if x.dtype == np.uint8:
        return x / 255
    else:
        return x / 1000


def compress_size(x):
    """
    Convert all float64 data to float32
    """
    if isinstance(x, (tuple, list)):
        return type(x)([compress_size(_) for _ in x])
    elif isinstance(x, dict):
        return {k: compress_size(x[k]) for k in x}
    elif np.isscalar(x):
        if (hasattr(x, 'dtype') and x.dtype == np.float64) or isinstance(x, float):
            return np.float32(x)
        else:
            return x
    elif isinstance(x, np.ndarray) and x.dtype == np.float64:
        return x.astype(np.float32)
    else:
        return x
