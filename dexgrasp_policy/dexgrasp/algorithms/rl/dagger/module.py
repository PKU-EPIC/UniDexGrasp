import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
# import pytorch_lightning as pl
from typing import List, Optional, Tuple
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo




class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others



class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        state_dim: int = 191 + 29,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = 191 + 29
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        others = {}
        input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim = -1)
        return self.transpn(input_pc), others

class Actor(nn.Module): # student

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(Actor, self).__init__()

        self.asymmetric = asymmetric
        
        self.use_pc = use_pc
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # 0~191(robot state), 191-207(object state) 207~236(goal), 236~300(obj visual feat)
        self.num_obs = 300
        
        if self.use_pc:
            self.num_obs = 191 + 29 # robot state + robot state
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=8, feature_dim=128)
            elif self.backbone_type =="transpn":
                self.backbone = TransPointNetBackbone(pc_dim=6, feature_dim=128)
            else:
                print("no such backbone")
                exit(123)
            print(self.backbone)
            self.num_obs += 128

        actor_layers = []

        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print("student")
        print(self.actor)


        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="transpn":  
                pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 300:].reshape(-1, 1024, 8)
                pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 300:].reshape(-1, 1024, 8)
                    pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions = self.actor(observations)
        else:
            actions = self.actor(observations[:,:300])
        return actions

    def act_inference(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="transpn":
                pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 300:].reshape(-1, 1024, 8)
                pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 300:].reshape(-1, 1024, 8)
                    pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions = self.actor(observations)
        else:
            actions = self.actor(observations[:,:300])
        return actions.detach()

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric
        self.use_pc = use_pc
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_obs = 300
        
        if self.use_pc:
            self.num_obs = 191 + 29 # robot_state + goal_state
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=8, feature_dim=128)
            elif self.backbone_type =="transpn":
                self.backbone = TransPointNetBackbone(pc_dim=6, feature_dim=128)
            else:
                print("no such backbone")
                exit(123)
            print(self.backbone)
            self.num_obs += 128


        actor_layers = []
        critic_layers = []
            
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers.append(nn.Linear(self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):

        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="transpn":
                pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 300:].reshape(-1, 1024, 8)
                pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 300:].reshape(-1, 1024, 8)
                    pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.actor(observations[:,:300])
        # actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if not self.use_pc:
            observations = observations[:,:300]

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="transpn":
                pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 300:].reshape(-1, 1024, 8)
                pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 300:].reshape(-1, 1024, 8)
                    pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.actor(observations[:,:300])
        return actions_mean.detach()
        
    def evaluate(self, observations, states, actions):
        if self.use_pc and not self.freeze_backbone:
            if self.backbone_type =="transpn":
                pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 300:].reshape(-1, 1024, 8)
                pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 300:300+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 300+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 300:].reshape(-1, 1024, 8)
                    pc_feature = self.backbone(pc)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.actor(observations[:,:300])

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if not self.use_pc:
            observations = observations[:,:300]

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None