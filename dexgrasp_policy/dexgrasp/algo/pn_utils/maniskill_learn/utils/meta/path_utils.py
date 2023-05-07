import os
import os.path as osp
import shutil
from pathlib import Path

from ..data.type import is_str


def to_abspath(x):
    return osp.abspath(x)


def get_filename(x):
    return osp.basename(str(x))


def get_dirname(x):
    return osp.dirname(str(x))


def get_filename_suffix(x):
    return get_filename(x).split('.')[-1]


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def add_suffix_to_filename(x, suffix=''):
    dirname = get_dirname(x)
    filename = get_filename(x)
    dot_split = filename.split('.')
    dot_split[-2] += f'_{suffix}'
    return osp.join(dirname, '.'.join(dot_split))


def replace_suffix(x, suffix=''):
    dirname = get_dirname(x)
    filename = get_filename(x)
    name_split = filename.split('.')
    name_split[-1] = suffix
    return osp.join(dirname, '.'.join(name_split))


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(str(filename)):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def copy_folder(from_path, to_path, overwrite=True):
    print(f'Copy files from {from_path} to {to_path}')
    from_path = str(from_path)
    to_path = str(to_path)
    if os.path.exists(to_path) and overwrite:
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def copy_folders(source_dir, folder_list, target_dir, overwrite=True):
    assert all(['/' not in _ for _ in folder_list])
    for i in folder_list:
        copy_folder(osp.join(source_dir, i), osp.join(target_dir, i), overwrite)


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the directory. Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_vcs_root(path, markers=('.git',)):
    """Finds the root directory (including itself) of specified markers.
    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.
    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None
