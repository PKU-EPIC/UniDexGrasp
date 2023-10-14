def flatten_dict(x, title=''):
    ret = {}
    for k in x:
        new_k = k if title == '' else f'{title}_{k}'
        if isinstance(x[k], dict):
            ret.update(flatten_dict(x[k], new_k))
        else:
            ret[new_k] = x[k]
    return ret
