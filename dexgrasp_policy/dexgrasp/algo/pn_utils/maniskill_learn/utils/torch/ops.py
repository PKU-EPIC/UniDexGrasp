import torch


def masked_average(x, axis, mask=None, keepdim=False):
    if mask is None:
        return torch.mean(x, dim=axis, keepdim=keepdim)
    else:
        return torch.sum(x * mask, dim=axis, keepdim=keepdim) / (torch.sum(mask, dim=axis, keepdim=keepdim) + 1E-6)


def masked_max(x, axis, mask=None, keepdim=False, empty_value=0):
    if mask is None:
        return torch.max(x, dim=axis, keepdim=keepdim).values
    else:
        value_with_inf = torch.max(x * mask + -1E18 * (1 - mask), dim=axis, keepdim=keepdim).values
        # The masks are all zero will cause inf
        value = torch.where(value_with_inf > -1E17, value_with_inf, torch.ones_like(value_with_inf) * empty_value)
        return value
