from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.pn_utils.maniskill_learn.networks import build_model, hard_update, soft_update
from algo.pn_utils.maniskill_learn.optimizers import build_optimizer
from algo.pn_utils.maniskill_learn.utils.data import to_torch
from ..builder import MFRL
from algo.pn_utils.maniskill_learn.utils.torch import BaseAgent
import tools.data_augment as DA


@MFRL.register_module()
class CURL(BaseAgent):
    def __init__(self, policy_cfg, value_cfg, obs_shape, action_shape, action_space, batch_size=128, gamma=0.99,
                 update_coeff=0.005, alpha=0.2, target_update_interval=1, automatic_alpha_tuning=True,
                 alpha_optim_cfg=None, feature_dim=192,):
        super(CURL, self).__init__()
        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        value_optim_cfg = value_cfg.pop("optim_cfg")


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


        self.policy = build_model(policy_cfg)
        self.critic = build_model(value_cfg)
        self.target_critic = build_model(value_cfg)

        self.encoderQ = self.critic.encoder#Contrastive init
        self.encoderK = self.target_critic.encoder#Contrastive init

        hard_update(self.target_critic, self.critic)
        hard_update(self.policy.encoder, self.critic.encoder)
        
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -np.prod(action_shape)
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, value_optim_cfg)
        
        #Contrastive init
        self.feature_dim = feature_dim
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim, requires_grad=True))
        self.W_optim = build_optimizer(self.W, value_optim_cfg)
        self.encoder_optim = build_optimizer(self.critic.encoder, value_optim_cfg)#Contrastive init
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        augment_batch = deepcopy(sampled_batch)
        #augment_batch['obs']['pointcloud'] = DA.pcd_transform(augment_batch['obs']['pointcloud'])

        sampled_batch = to_torch(sampled_batch, dtype='float32', device=self.device, non_blocking=True)
        augment_batch = to_torch(augment_batch, dtype='float32', device=self.device, non_blocking=True)

        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        for key in augment_batch:
            if not isinstance(augment_batch[key], dict) and augment_batch[key].ndim == 1:
                augment_batch[key] = augment_batch[key][..., None]


        #critic_update
        with torch.no_grad():
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

        #policy_update
        feature = self.encoderQ(sampled_batch['obs'])
        pi, log_pi = self.policy.module.policy_head(self.policy.module.backbone(feature), num_actions=1)[:2]
        #pi, log_pi = self.policy(sampled_batch['obs'], mode='all', detach_encoder=True)[:2]
        q_pi = self.critic(sampled_batch['obs'], pi)
        q_pi_min = torch.min(q_pi, dim=-1, keepdim=True).values
        policy_loss = -(q_pi_min - self.alpha * log_pi).mean()

        self.policy_optim.zero_grad()
        #self.encoder_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        #self.encoder_optim.step()

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (-(log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        #encoder update
        """
        featureQ = self.encoderQ(sampled_batch['obs'])
        featureK = self.encoderK(augment_batch['obs'])
        featureK = featureK.detach()
        Wk = torch.matmul(self.W, featureK.T)
        logits = torch.matmul(featureQ, Wk) #[B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        encoder_loss = self.cross_entropy_loss(logits, labels).mean()

        self.encoder_optim.zero_grad()
        self.W_optim.zero_grad()    
        encoder_loss.backward()
        self.encoder_optim.step()
        self.W_optim.step()
        """

        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)


        hard_update(self.policy.module.encoder, self.critic.module.encoder)
        
        return {
            'critic_loss': critic_loss.item(),
            'max_critic_abs_err': abs_critic_error.max().item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'q': torch.min(q, dim=-1).values.mean().item(),
            'q_target': torch.mean(q_target).item(),
            'log_pi': torch.mean(log_pi).item(),
            #'encoder_loss': encoder_loss.item(),
        }
