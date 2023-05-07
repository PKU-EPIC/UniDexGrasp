"""
A Minimalist Approach toOffline Reinforcement Learning:
    https://arxiv.org/pdf/2106.06860.pdf
"""
import torch
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import soft_update
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from ..builder import BRL
from ..mfrl import TD3


@BRL.register_module()
class TD3_BC(TD3):
    def __init__(self, alpha=2.5, reward_scale=1, **kwargs):
        super(TD3_BC, self).__init__(**kwargs)
        self.reward_scale = reward_scale
        self.alpha = alpha

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        sampled_batch['rewards'] = self.reward_scale * sampled_batch['rewards']

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
            pred_action = self.policy(sampled_batch['obs'], mode='eval')
            q = self.critic(sampled_batch['obs'], pred_action)[..., 0]
            lmbda = self.alpha / (q.abs().mean().detach() + 1E-5)
            bc_loss = F.mse_loss(pred_action, sampled_batch['actions'])
            policy_loss = -lmbda * q.mean() + bc_loss
            bc_abs_error = torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            soft_update(self.target_critic, self.critic, self.update_coeff)
            soft_update(self.target_policy, self.policy, self.update_coeff)
        else:
            policy_loss = torch.zeros(1)
            bc_loss = torch.zeros(1)
            lmbda = torch.zeros(1)
            bc_abs_error = torch.zeros(1)

        return {
            'critic_loss': critic_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'policy_loss': policy_loss.item(),
            'bc_loss': bc_loss.item(),
            'bc_abs_error': bc_abs_error.item(),
            'lmbda': lmbda.item(),
        }
