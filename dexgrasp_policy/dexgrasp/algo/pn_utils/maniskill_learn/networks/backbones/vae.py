import torch
import torch.nn as nn

from ..utils import combine_obs_with_action
from algo.pn_utils.maniskill_learn.utils.data import get_one_shape
from algo.pn_utils.maniskill_learn.utils.torch import get_one_device
from ..builder import BACKBONES, build_backbone


@BACKBONES.register_module()
class CVAE(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, latent_dim, log_sig_min=-4, log_sig_max=15):
        # Regress P(var | con)
        super(CVAE, self).__init__()
        # assert encoder_cfg.mlp_spec[-1] // 2 + cond_dim == decoder_cfg.mlp_spec[0]
        self.encoder = build_backbone(encoder_cfg)
        self.decoder = build_backbone(decoder_cfg)
        self.latent_dim = latent_dim
        self.log_sig_min, self.log_sig_max = log_sig_min, log_sig_max

    def forward(self, cond, var):
        inputs = combine_obs_with_action(cond, var)
        z = self.encoder(inputs)
        mean = z[..., :self.latent_dim]
        log_std = z[..., self.latent_dim:].clamp(self.log_sig_min, self.log_sig_max)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        return self.decode(cond, z), mean, std

    def decode(self, cond, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            batch_size = get_one_shape(cond)[0]
            device = get_one_device(cond)
            z = torch.randn(batch_size, self.latent_dim, device=device).clamp(-0.5, 0.5)
        inputs = combine_obs_with_action(cond, z)
        return self.decoder(inputs)
