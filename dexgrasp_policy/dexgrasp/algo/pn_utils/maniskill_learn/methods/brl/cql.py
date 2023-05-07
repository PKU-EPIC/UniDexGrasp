"""
Conservative Q-Learning for Offline Reinforcement Learning:
    https://arxiv.org/pdf/2006.04779
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from algo.pn_utils.maniskill_learn.networks import soft_update
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch, repeat_interleave
from math import log
from ..builder import BRL
from ..mfrl import SAC


@BRL.register_module()
class CQL(SAC):
    def __init__(self, num_action_sample=10, forward_block=2, automatic_regularization_tuning=True, lagrange_thresh=10.,
                 alpha_prime=5., temperature=1., min_q_weight=1., min_q_with_entropy=True, alpha_prime_optim_cfg=None,
                 target_q_with_entropy=True, reward_scale=1, **kwargs):
        super(CQL, self).__init__(**kwargs)
        self.temperature = temperature
        self.min_q_weight = min_q_weight
        self.min_q_with_entropy = min_q_with_entropy

        self.num_action_sample = num_action_sample
        self.automatic_regularization_tuning = automatic_regularization_tuning
        self.lagrange_thresh = lagrange_thresh
        self.alpha_prime = alpha_prime
        self.forward_block = forward_block
        self.reward_scale = reward_scale
        self.target_q_with_entropy = target_q_with_entropy

        if self.automatic_regularization_tuning:
            self.log_alpha_prime = Parameter(torch.zeros(1, requires_grad=True))
            self.alpha_prime = self.log_alpha_prime.exp().item()
            self.alpha_prime_optim = build_optimizer(self.log_alpha_prime, alpha_prime_optim_cfg)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        # print(sampled_batch['rewards'].mean())
        sampled_batch['rewards'] = self.reward_scale * sampled_batch['rewards']
        # print(sampled_batch['rewards'].mean())
        # exit(0)
        with torch.no_grad():
            next_action, next_log_prob = self.policy(sampled_batch['next_obs'], mode='all')[:2]
            q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            if self.target_q_with_entropy:
                min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            else:
                min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target
        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1]))

        """"                 Key difference from SAC                     """
        assert self.num_action_sample % self.forward_block == 0
        action_samples_per_block = self.num_action_sample // self.forward_block
        num_per_block = self.batch_size * action_samples_per_block
        # [batch_size * action_samples_per_block, ...]
        repeated_block_obs = repeat_interleave(sampled_batch['obs'], action_samples_per_block, 0)

        def policy_sample_and_split(obs, num_actions):
            sampled_actions, sampled_log_prob = self.policy(obs, num_actions, mode='all')[:2]
            sampled_actions = sampled_actions.reshape(num_per_block, self.forward_block, -1)
            sampled_log_prob = sampled_log_prob.reshape(num_per_block, self.forward_block)
            # [batch_size * action_samples_per_block, blocks, ...]
            return sampled_actions, sampled_log_prob.detach()

        cur_sampled_actions, cur_sampled_log_prob = policy_sample_and_split(sampled_batch['obs'], self.num_action_sample)
        next_sampled_actions, next_sampled_log_prob = policy_sample_and_split(sampled_batch['next_obs'],
                                                                              self.num_action_sample)
        unif_sampled_actions, unif_log_prob = self.policy['policy_head'].uniform(num_per_block * self.forward_block)
        unif_sampled_actions = unif_sampled_actions.reshape(num_per_block, self.forward_block, -1)
        unif_log_prob = unif_log_prob.reshape(num_per_block, self.forward_block)
        # print(cur_sampled_log_prob.shape, cur_sampled_actions.shape)

        def critic_compute_and_merge(actions, log_probs):
            qs = []
            for i in range(self.forward_block):
                if self.min_q_with_entropy:
                    tmp = self.critic(repeated_block_obs, actions[:, i]) - log_probs[:, i][..., None]
                else:
                    tmp = self.critic(repeated_block_obs, actions[:, i])
                qs.append(tmp)
                qs[i] = qs[i].reshape(self.batch_size, action_samples_per_block, -1)
            return torch.cat(qs, dim=1)

        q_cur_a = critic_compute_and_merge(cur_sampled_actions, cur_sampled_log_prob)
        q_next_a = critic_compute_and_merge(next_sampled_actions, next_sampled_log_prob)
        q_unif_a = critic_compute_and_merge(unif_sampled_actions, unif_log_prob)
        q_sample = torch.cat([q_cur_a, q_next_a, q_unif_a], dim=1)
        # lower bound of maximum of q  (log sum exp - log N <= max)
        q_sample = torch.logsumexp(q_sample / self.temperature, dim=1) * self.temperature - log(q_sample.shape[1])

        min_q_loss = (q_sample.mean(0) - q.mean(0)) * self.min_q_weight

        if self.automatic_regularization_tuning:
            self.alpha_prime_optim.zero_grad()
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1E6)
            alpha_prime_loss = alpha_prime * (-(min_q_loss - self.lagrange_thresh).detach()).mean()
            alpha_prime_loss.backward()
            self.alpha_prime_optim.step()
            self.alpha_prime = self.log_alpha_prime.exp().item()
        else:
            alpha_prime_loss = torch.tensor(0.).to(self.device)

        self.critic_optim.zero_grad()
        (critic_loss + min_q_loss.sum() * self.alpha_prime).backward()
        self.critic_optim.step()
        """" ************************************************************  """

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
            'min_q_loss': min_q_loss.mean().item(),
            'min_q_loss_minus_thresh': (min_q_loss - self.lagrange_thresh).mean().item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'alpha_prime': self.alpha_prime,
            'alpha_prime_loss': alpha_prime_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'log_pi': torch.mean(log_pi).item(),
        }
