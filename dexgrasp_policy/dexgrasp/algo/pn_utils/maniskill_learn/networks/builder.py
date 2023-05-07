import torch.nn as nn

from ..utils.meta import Registry, build_from_cfg

BACKBONES = Registry('backbone')
DENSEHEADS = Registry('policy_head')

POLICYNETWORKS = Registry('policy_network')
VALUENETWORKS = Registry('value_network')
MODELNETWORKS = Registry('model_network')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dense_head(cfg):
    return build(cfg, DENSEHEADS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_model(cfg, default_args=None):
    for model_type in [BACKBONES, POLICYNETWORKS, VALUENETWORKS]:
        if cfg['type'] in model_type.module_dict:
            return build(cfg, model_type, default_args)
    raise RuntimeError(f"No this model type:{cfg['type']}!")
