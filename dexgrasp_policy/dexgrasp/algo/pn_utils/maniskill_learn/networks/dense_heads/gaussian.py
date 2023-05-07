import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions import Normal
from algo.pn_utils.maniskill_learn.utils.torch import ExtendedModule
from ..builder import DENSEHEADS


class GaussianHeadBase(ExtendedModule):
    def __init__(self, scale_prior=1, bias_prior=0, dim_action=None, epsilon=1E-6):
        super(GaussianHeadBase, self).__init__()
        self.scale_prior = Parameter(torch.tensor(scale_prior, dtype=torch.float32), requires_grad=False)
        self.bias_prior = Parameter(torch.tensor(bias_prior, dtype=torch.float32), requires_grad=False)
        if dim_action is None:
            assert self.scale_prior.ndim == 1
            self.dim_action = self.scale_prior.shape[0]
        self.epsilon = epsilon
        self.log_unif_prob = torch.log(1.0 / (2 * self.scale_prior.data)).sum().item()

    def uniform(self, sample_shape):
        return ((torch.rand(sample_shape, self.dim_action, device=self.device) * 2 - 1)
                * self.scale_prior + self.bias_prior), torch.ones(sample_shape, device=self.device) * self.log_unif_prob

    def sample(self, mean, log_std, num_actions):
        log_std = log_std.expand_as(mean)
        mean = torch.repeat_interleave(mean, num_actions, dim=0)
        log_std = torch.repeat_interleave(log_std, num_actions, dim=0)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale_prior + self.bias_prior
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.scale_prior * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.scale_prior + self.bias_prior
        return action, log_prob, mean, log_std, std


@DENSEHEADS.register_module()
class GaussianHead(GaussianHeadBase):
    def __init__(self, scale_prior=1, bias_prior=0, dim_action=None, log_sig_min=-20, log_sig_max=2, epsilon=1e-6):
        super(GaussianHead, self).__init__(scale_prior, bias_prior, dim_action, epsilon)
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max

    def forward(self, feature, num_actions=1):
        assert feature.shape[-1] % 2 == 0
        mean, log_std = feature.split(feature.shape[-1] // 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return self.sample(mean, log_std, num_actions)


@DENSEHEADS.register_module()
class SharedGaussianHead(GaussianHeadBase):
    def __init__(self, scale_prior=1, bias_prior=0, dim_action=None, epsilon=1e-6):
        super(SharedGaussianHead, self).__init__(scale_prior, bias_prior, dim_action, epsilon)
        self.log_std = nn.Parameter(torch.zeros(1, self.dim_action).float())

    def forward(self, mean, num_actions=1):
        return self.sample(mean, self.log_std, num_actions)
