from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import build_model, hard_update, soft_update
from algo.pn_utils.maniskill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from ..builder import MFRL
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent
import tools.data_augment as DA


@MFRL.register_module()
class CURL(BaseAgent):
    def __init__(self,encoder_cfg, policy_cfg, value_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99,
                 update_coeff=0.005, alpha=0.2, target_update_interval=1, automatic_alpha_tuning=True,
                 alpha_optim_cfg=None, feature_dim=192, encoder_optim_cfg = None):
        super(CURL, self).__init__()
        subpolicy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        encoder_cfg = replace_placeholder_with_args(encoder_cfg, **replaceable_kwargs)

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning


        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        value_cfg['obs_shape'] = obs_shape
        value_cfg['action_shape'] = action_shape


        self.subpolicy = build_model(policy_cfg)

        self.critic = build_model(value_cfg)
        self.encoderQ = build_model(encoder_cfg)#Contrastive init

        self.target_critic = build_model(value_cfg)
        self.encoderK = build_model(encoder_cfg)#Contrastive init
        hard_update(self.target_critic, self.critic)
        hard_update(self.encoderK, self.encoderQ)#Contrastive init
        

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -np.prod(action_shape)
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)
        self.subpolicy_optim = build_optimizer(self.subpolicy, subpolicy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)
        self.encoder_optim = build_optimizer(self.encoderQ, encoder_optim_cfg)#Contrastive init
        
        #Contrastive init
        self.feature_dim = feature_dim
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim, requires_grad=True))
        self.W_optim = build_optimizer(self.W, encoder_optim_cfg)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.policy = nn.Sequential(self.encoderQ, self.subpolicy)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        augment_batch = deepcopy(sampled_batch)
        augment_batch['obs']['pointcloud'] = DA.pcd_transform(augment_batch['obs']['pointcloud'])

        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        augment_batch = to_torch(augment_batch, dtype='float32', device=self.device, non_blocking=True)

        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]

        
        with torch.no_grad():
            next_featureQ = self.encoderK(sampled_batch['next_obs'])
            next_action, next_log_prob = self.subpolicy(next_featureQ, mode='all')[:2]
            q_next_target = self.target_critic(next_featureQ, next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            q_target = sampled_batch['rewards'] + (1 - sampled_batch['dones']) * self.gamma * min_q_next_target

        featureQ = self.encoderQ(sampled_batch['obs'])
        featureK = self.encoderK(augment_batch['obs'])
        featureK = featureK.detach()
        Wk = torch.matmul(self.W, featureK.T)
        logits = torch.matmul(featureQ, Wk) #[B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        encoder_loss = self.cross_entropy_loss(logits, labels).mean()
        
        q = self.critic(featureQ, sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1]))

        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.encoder_optim.step()

        pi, log_pi = self.subpolicy(featureQ.detach(), mode='all')[:2]
        q_pi = self.critic(featureQ.detach(), pi)
        q_pi_min = torch.min(q_pi, dim=-1, keepdim=True).values
        subpolicy_loss = -(q_pi_min - self.alpha * log_pi).mean()

        self.subpolicy_optim.zero_grad()
        subpolicy_loss.backward()
        self.subpolicy_optim.step()

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
            soft_update(self.encoderK, self.encoderQ, self.update_coeff)
        '''
        self.W_optim.zero_grad()    
        encoder_loss.backward()
        self.W_optim.step()
        '''
        
        hard_update(self.policy, nn.Sequential(self.encoderQ, self.subpolicy))
        return {
            'critic_loss': critic_loss.item(),
            'max_critic_abs_err': abs_critic_error.max().item(),
            'policy_loss': subpolicy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'log_pi': torch.mean(log_pi).item(),
            'encoder_loss': encoder_loss.item(),
        }
