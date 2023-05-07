import warnings, functools, subprocess
from inspect import getfullargspec
from importlib import import_module


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return None. Otherwise, an ImportError is raise.
        Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f'custom_imports must be a list but got type {type(imports)}')

    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.', UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def check_prerequisites(prerequisites, checker, msg_tmpl='Prerequisites "{}" are required in method "{}" '
                                                         'but not found, please install them first.'):  # yapf: disable
    """A decorator factory to check if prerequisites are satisfied.
    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.
    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def requires_package(prerequisites):
    def _check_py_package(package):
        try:
            import_module(package)
        except ImportError:
            return False
        else:
            return True

    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    def _check_executable(cmd):
        if subprocess.call(f'which {cmd}', shell=True) != 0:
            return False
        else:
            return True

    return check_prerequisites(prerequisites, checker=_check_executable)


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some argments are deprecate and try to replace deprecate src_arg_name to dst_arg_name.
    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.
    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(f'"{src_arg_name}" is deprecated in `{func_name}`, please use "{dst_arg_name}" '
                                      'instead')
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        warnings.warn(f'"{src_arg_name}" is deprecated in `{func_name}`, please use "{dst_arg_name}" '
                                      'instead')
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper
