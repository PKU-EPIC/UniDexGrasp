"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import build_model
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent
from ..builder import BRL


@BRL.register_module()
class BC(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super(BC, self).__init__()
        self.batch_size = batch_size

        policy_optim_cfg = policy_cfg.pop("optim_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.policy = build_model(policy_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"])
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        pred_action = self.policy(sampled_batch['obs'], mode='eval')
        policy_loss = F.mse_loss(pred_action, sampled_batch['actions'])
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return {
            'policy_abs_error': torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item(),
            'policy_loss': policy_loss.item()
        }
