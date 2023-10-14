from .string_utils import regex_match
from .type import is_dict, is_seq_of, is_tuple_of, is_list_of


def custom_filter(func, x, value=True):
    def can_recursive(_):
        return is_dict(_) or is_tuple_of(_) or is_list_of(_)

    if is_dict(x):
        ret = {}
        for item in x.items():
            _ = item[int(value)]
            if not func(_):
                continue
            if can_recursive(_) and value:
                _ = custom_filter(func, _, True)
                if func(_):
                    ret[item[0]] = _
            else:
                ret[item[0]] = item[1]
        # print('Output', ret)
        if len(ret.keys()) == 0:
            return None
        else:
            return ret

        # func_kv = lambda item: func(item[int(value)])
        # return dict(filter(func_kv, x.items()))
    elif is_seq_of(x):
        assert value
        ret = []
        for _ in x:
            if not func(_):
                continue
            if can_recursive(_):
                _ = custom_filter(func, _, True)
            if func(_):
                ret.append(_)
        if len(ret) == 0:
            return None
        else:
            return type(x)(ret)
    else:
        raise NotImplementedError()


def filter_none(x):
    func = lambda _: _ is not None
    return custom_filter(func, x, True)


def filter_with_regex(x, regex, value=True):
    func = lambda _: regex_match(_, regex)
    return custom_filter(func, x, value)
