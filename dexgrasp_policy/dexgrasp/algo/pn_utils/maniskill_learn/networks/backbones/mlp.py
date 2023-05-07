import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from algo.pn_utils.maniskill_learn.utils.meta import get_root_logger
from algo.pn_utils.maniskill_learn.utils.torch import load_checkpoint
from ..builder import BACKBONES
from ..modules import ConvModule, build_init
from ..modules import build_activation_layer, build_norm_layer


@BACKBONES.register_module()
class LinearMLP(nn.Module):
    def __init__(self, mlp_spec, norm_cfg=dict(type='BN1d'), bias='auto', inactivated_output=True,
                 pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        super(LinearMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            bias_i = norm_cfg is None if bias == 'auto' else bias
            # print(mlp_spec[i], mlp_spec[i + 1], bias_i)
            self.mlp.add_module(f'linear{i}', nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=bias_i))
            if norm_cfg:
                self.mlp.add_module(f'norm{i}', build_norm_layer(norm_cfg, mlp_spec[i + 1])[1])
            if act_cfg:
                self.mlp.add_module(f'act{i}', build_activation_layer(act_cfg))
        self.init_weights(pretrained, linear_init_cfg, norm_init_cfg)

    def forward(self, input):
        input = input
        return self.mlp(input)

    def init_weights(self, pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            linear_init = build_init(linear_init_cfg) if linear_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Linear) and linear_init:
                    linear_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError('pretrained must be a str or None')


@BACKBONES.register_module()
class ConvMLP(nn.Module):
    def __init__(self, mlp_spec, norm_cfg=dict(type='BN1d'), bias='auto', inactivated_output=True,
                 pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        super(ConvMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_spec[i],
                    mlp_spec[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=True,
                    with_spectral_norm=False,
                    padding_mode='zeros',
                    order=('conv', 'norm', 'act'))
            )
        self.init_weights(pretrained, conv_init_cfg, norm_init_cfg)

    def forward(self, input):
        return self.mlp(input)

    def init_weights(self, pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            conv_init = build_init(conv_init_cfg) if conv_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Conv1d) and conv_init:
                    conv_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError('pretrained must be a str or None')
