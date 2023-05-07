"""
Soft Actor-Critic Algorithms and Applications:
    https://arxiv.org/abs/1812.05905
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor:
   https://arxiv.org/abs/1801.01290
"""
import numpy as np
from numpy.core.numeric import ones
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import build_model, hard_update, soft_update
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from ..builder import MFRL
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent


@MFRL.register_module()
class GAIL(BaseAgent):
    def __init__(self, policy_cfg, value_cfg, discriminator_cfg, obs_shape, action_shape, action_space, batch_size=128, discrim_batch=1024, gamma=0.99,
                 update_coeff=0.005, alpha=0.2, target_update_interval=1, automatic_alpha_tuning=True,
                 alpha_optim_cfg=None):
        super(GAIL, self).__init__()
        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")
        discriminator_optim_cfg = discriminator_cfg.pop("optim_cfg")

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.batch_size = batch_size
        self.discrim_batch = discrim_batch
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space
        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape
        discriminator_cfg['obs_shape'] = obs_shape
        discriminator_cfg['action_shape'] = action_shape

        self.policy = build_model(policy_cfg)
        self.critic = build_model(value_cfg)
        self.discriminator = build_model(discriminator_cfg)
        self.discrim_criterion = nn.BCELoss()

        self.target_critic = build_model(value_cfg)
        hard_update(self.target_critic, self.critic)

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -np.prod(action_shape)
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)
        self.discriminator_optim = build_optimizer(self.discriminator, discriminator_optim_cfg)

    def update_discriminator(self, expert_replay, tmp_replay):
        expert_sampled_batch = expert_replay.sample(self.discrim_batch)
        expert_sampled_batch = to_torch(expert_sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        tmp_sampled_batch = tmp_replay.sample(self.discrim_batch)
        tmp_sampled_batch = to_torch(tmp_sampled_batch, dtype='float32', device=self.device, non_blocking=True)

        expert_out = self.discriminator(expert_sampled_batch['obs'], expert_sampled_batch['actions'])
        tmp_out = self.discriminator(tmp_sampled_batch['obs'], tmp_sampled_batch['actions'])

        self.discriminator_optim.zero_grad()
        expert_loss = self.discrim_criterion(expert_out, torch.zeros((expert_out.shape[0],1), device = self.device)) 
        tmp_loss = self.discrim_criterion(tmp_out, torch.ones((expert_out.shape[0],1), device=self.device))
        discrim_loss = expert_loss + tmp_loss
            
        discrim_loss = discrim_loss.mean()
        discrim_loss.backward()
        self.discriminator_optim.step()
        
        return expert_loss.mean().item(),  tmp_loss.mean().item()

    def expert_reward(self, obs, action):
        obs = to_torch(obs, dtype='float32', device=self.device, non_blocking=True)
        action = to_torch(action, dtype='float32', device=self.device, non_blocking=True)
        exr = -torch.log(self.discriminator(obs, action))
        return exr.cpu().detach().numpy()

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)

        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]


        with torch.no_grad():
            expert_rewards = -torch.log(self.discriminator(sampled_batch['obs'], sampled_batch['actions']))
            sampled_batch['rewards'] = 0.3*sampled_batch['rewards'] + 0.7*expert_rewards

            next_action, next_log_prob = self.policy(sampled_batch['next_obs'], mode='all')[:2]
            q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target
        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1]))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.policy(sampled_batch['obs'], mode='all')[:2]
        q_pi = self.critic(sampled_batch['obs'], pi)
        q_pi_min = torch.min(q_pi, dim=-1, keepdim=True).values
        policy_loss = -(q_pi_min - self.alpha * log_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (-(log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        return {
            'critic_loss': critic_loss.item(),
            'max_critic_abs_err': abs_critic_error.max().item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'log_pi': torch.mean(log_pi).item(),
        }
