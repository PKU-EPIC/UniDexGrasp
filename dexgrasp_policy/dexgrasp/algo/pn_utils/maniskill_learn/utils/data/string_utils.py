import re


def custom_format(template_string, **kwargs):
    """
    The python format uses {...} to indicate the variable that needs to be replaced.
    custom_format uses &lformat ... &rformat to indicate the variable, which means {} can be used in the template_string
    as a normal character.
    """
    template_string = template_string.replace('{', '{{')
    template_string = template_string.replace('}', '}}')
    template_string = template_string.replace('&lformat ', '{')
    template_string = template_string.replace(' &rformat', '}')
    return template_string.format_map(kwargs)


def regex_match(string, pattern):
    """Check if the string matches the given pattern"""
    return re.match(pattern, string) is not None


def prefix_match(string, prefix=None):
    """Check if the string matches the given prefix"""
    if prefix is None or len(prefix) == 0:
        return True
    return re.match(f'({prefix})+(.*?)', string) is not None


pyrl_h5_int_starting = "int__pyrl__"


def h5_name_format(name):
    """
    HDF5 does not accept using a number as the name of a group or a dataset.
    We add a prefix here to make the number name valid in HDF5.
    """
    if isinstance(name, int):
        name = pyrl_h5_int_starting + str(name)
    elif isinstance(name, str):
        if name.isnumeric():
            name = pyrl_h5_int_starting + name
    else:
        raise TypeError(f"The type of name is {type(name)}, the value is {name}!")
    return name


def h5_name_deformat(name):
    """
    Identify the pattern that is used to represent the number name. Delete the extra prefix and make it normal.
    """
    if name.startswith(pyrl_h5_int_starting):
        return eval(name[len(pyrl_h5_int_starting):])
    else:
        return name
