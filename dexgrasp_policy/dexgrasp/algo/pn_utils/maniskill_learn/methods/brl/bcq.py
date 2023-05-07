"""
Off-Policy Deep Reinforcement Learning without Exploration
    https://arxiv.org/abs/1812.02900
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent

from algo.pn_utils.maniskill_learn.networks import build_model, soft_update
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch, repeat_interleave, get_one_shape
from algo.pn_utils.maniskill_learn.utils.torch import run_with_mini_batch
from ..builder import BRL


@BRL.register_module()
class BCQ(BaseAgent):
    def __init__(self, value_cfg, policy_vae_cfg, obs_shape, action_shape, action_space,
                 batch_size=128, gamma=0.99, update_coeff=0.005, lmbda=0.75, num_random_action_train=10,
                 num_random_action_eval=100, target_update_interval=1):
        super(BCQ, self).__init__()
        assert policy_vae_cfg.nn_cfg.type == 'CVAE'

        value_optim_cfg = value_cfg.pop("optim_cfg")
        policy_vae_optim_cfg = policy_vae_cfg.pop("optim_cfg")

        policy_vae_cfg['obs_shape'] = obs_shape
        policy_vae_cfg['action_shape'] = action_shape
        policy_vae_cfg['action_space'] = action_space

        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.lmbda = lmbda
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.num_random_action_train = num_random_action_train
        self.num_random_action_eval = num_random_action_eval

        self.policy_vae = build_model(policy_vae_cfg)
        self.critic = build_model(value_cfg)

        self.target_critic = build_model(value_cfg)

        self.policy_vae_optim = build_optimizer(self.policy_vae, policy_vae_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)

    def forward(self, obs, **kwargs):
        # obs of size: [B, ...]
        obs = to_torch(obs, 'float32', self.device)
        batch_size = get_one_shape(obs)[0]
        with torch.no_grad():
            obs = repeat_interleave(obs, self.num_random_action_eval, 0)
            action = self.policy_vae(obs, decode=True)
            value = self.critic(obs, action)
            value = value.reshape(batch_size, self.num_random_action_eval, -1)[..., 0]
            action = action.reshape(batch_size, self.num_random_action_eval, -1)
            index = value.argmax(-1)[:, None, None]
            index = index.expand_as(action)[:, :1]
            action = torch.gather(action, 1, index)[:, 0]
        return action.cpu().data.numpy()

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)

        for key in sampled_batch:
           if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]

        recon, mean, std = self.policy_vae(sampled_batch['obs'], sampled_batch['actions'])
        recon_loss = F.mse_loss(recon, sampled_batch['actions'])
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.policy_vae_optim.zero_grad()
        vae_loss.backward()
        self.policy_vae_optim.step()

        # Critic Training
        with torch.no_grad():
            # Duplicate next obs self.num_random_action_train times
            next_obs = repeat_interleave(sampled_batch['next_obs'], self.num_random_action_train, 0)

            def get_next_critic(x):
                return self.target_critic(x, self.policy_vae(x, decode=True))

            q_next = run_with_mini_batch(get_next_critic, next_obs, self.batch_size)

            # q_next = self.target_critic(next_obs,
            #                             self.policy_vae.decode(next_obs))
            # Soft Clipped Double Q-learning
            q_next = self.lmbda * q_next.min(-1).values + (1. - self.lmbda) * q_next.max(-1).values
            q_next = q_next.reshape(self.batch_size, self.num_random_action_train).max(-1).values[:, None]
            # Take max over each action sampled from the VAE
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * q_next

        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1]))
        abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1]))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        return {
            'critic_loss': critic_loss.item(),
            'max_critic_abs_err': abs_critic_error.max().item(),
            # 'policy_loss': policy_loss.item(),
            'vae_loss': vae_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
        }
