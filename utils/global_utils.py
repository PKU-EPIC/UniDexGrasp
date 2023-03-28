import torch

def update_dict(old_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict):
            if not key in old_dict:
                old_dict[key] = {}
            update_dict(old_dict[key], value)
        else:
            old_dict[key] = value

def log_loss_summary(loss_dict, cnt, log_loss):
    def log_dict(d, prefix=None):
        for key, value in d.items():
            name = str(key) if prefix is None else '/'.join([prefix, str(key)])
            if isinstance(value, dict):
                log_dict(value, name)
            else:
                if key.endswith("_max"):
                    log_loss(name, d[key])
                else:
                    log_loss(name, d[key] / cnt)
    log_dict(loss_dict)

def add_dict(old_dict, new_dict):
    def copy_dict(d):
        ret = {}
        for key, value in d.items():
            if isinstance(value, dict):
                ret[key] = copy_dict(value)
            else:
                ret[key] = value
        del d
        return ret
    detach_dict(new_dict)
    for key, value in new_dict.items():
        if not key in old_dict.keys():
            if isinstance(value, dict):
                old_dict[key] = copy_dict(value)
            else:
                old_dict[key] = value
        else:
            if isinstance(value, dict):
                add_dict(old_dict[key], value)
            else:
                if key.endswith("_max"):
                    old_dict[key] = max(old_dict[key], value)
                else:
                    old_dict[key] += value

def detach_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            detach_dict(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.detach().cpu().numpy()
