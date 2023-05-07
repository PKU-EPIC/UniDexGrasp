import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import build_model, hard_update, soft_update
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from ..builder import MFRL
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent


@MFRL.register_module()
class TD3(BaseAgent):
    def __init__(self, policy_cfg, value_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99,
                 update_coeff=0.005, action_noise=0.2, noise_clip=0.5, policy_update_interval=2):
        super(TD3, self).__init__()
        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_coeff = update_coeff
        self.policy_update_interval = policy_update_interval
        self.action_noise = action_noise
        self.noise_clip = noise_clip

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape

        self.policy = build_model(policy_cfg)
        self.critic = build_model(value_cfg)

        self.target_policy = build_model(policy_cfg)
        self.target_critic = build_model(value_cfg)

        hard_update(self.target_critic, self.critic)
        hard_update(self.target_policy, self.policy)

        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]

        with torch.no_grad():
            _, _, next_mean_action, _, _ = self.target_policy(sampled_batch['next_obs'], mode='all')
            noise = (torch.randn_like(next_mean_action) * self.action_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.target_policy['policy_head'].clamp_action(next_mean_action + noise)
            q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target

        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if updates % self.policy_update_interval == 0:
            policy_loss = -self.critic(sampled_batch['obs'], self.policy(sampled_batch['obs'], mode='eval'))[
                ..., 0].mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            soft_update(self.target_critic, self.critic, self.update_coeff)
            soft_update(self.target_policy, self.policy, self.update_coeff)
        else:
            policy_loss = torch.zeros(1)

        return {
            'critic_loss': critic_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'policy_loss': policy_loss.item(),
        }
