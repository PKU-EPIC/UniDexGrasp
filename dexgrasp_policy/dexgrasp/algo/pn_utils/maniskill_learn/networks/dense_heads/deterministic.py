import torch.nn as nn, torch, numpy as np
from torch.nn import Parameter
from ..builder import DENSEHEADS
from algo.pn_utils.maniskill_learn.utils.torch import ExtendedModule


@DENSEHEADS.register_module()
class DeterministicHead(ExtendedModule):
    def __init__(self, scale_prior=1, bias_prior=0, noise_std=0.1):
        # The noise is the Gaussian noise for exploration.
        super(DeterministicHead, self).__init__()
        self.scale_prior = Parameter(torch.tensor(scale_prior, dtype=torch.float32), requires_grad=False)
        self.bias_prior = Parameter(torch.tensor(bias_prior, dtype=torch.float32), requires_grad=False)
        self.noise_std = noise_std

    def forward(self, feature, num_actions=1):
        """
        Forward will return action with exploration, log p, mean action, log std, std
        """
        assert num_actions == 1
        mean = torch.tanh(feature)
        noise = mean.clone().normal_(0., std=self.noise_std)
        action = (mean + noise).clamp(-1, 1) * self.scale_prior + self.bias_prior
        mean = mean * self.scale_prior + self.bias_prior
        return action, torch.ones_like(action) * -np.inf, mean, torch.ones_like(action) * -np.inf, torch.zeros_like(action)

    def clamp_action(self, action):
        action = (action - self.bias_prior) / self.scale_prior
        action = action.clamp(-1, 1)
        return action * self.scale_prior + self.bias_prior
