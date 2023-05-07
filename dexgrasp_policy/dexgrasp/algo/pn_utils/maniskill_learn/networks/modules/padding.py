import torch.nn as nn
import sys
from algo.pn_utils.maniskill_learn.utils.meta import Registry

PADDING_LAYERS = Registry('padding layer')

for module in [nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d,
               nn.ZeroPad2d, nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]:
    PADDING_LAYERS.register_module(module=module)


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.
    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.
    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)
    layer = padding_layer(*args, **kwargs, **cfg_)
    return layer
