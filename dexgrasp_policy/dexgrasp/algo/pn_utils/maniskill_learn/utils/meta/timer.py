from time import strftime, localtime


def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('y',        60*60*24*365),
        ('m',       60*60*24*30),
        ('d',         60*60*24),
        ('h',        60*60),
        ('m',      60),
        ('s',      1)
    ]
    ret = ''
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            ret += f'{period_value}{period_name}'
    return ret


def get_time_stamp():
    return strftime('%Y%m%d_%H%M%S', localtime())
