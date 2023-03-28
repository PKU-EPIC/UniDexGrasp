import torch.nn as nn
import torch
import os
import sys
from os.path import join as pjoin
from abc import abstractmethod

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, '..'))
sys.path.insert(0, pjoin(base_path, '..', '..'))

from network.models.graspipdf.ipdf_network import IPDFFullNet


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.device = cfg['device']
        self.loss_weights = cfg['model']['loss_weight']

        self.cfg = cfg
        self.feed_dict = {}
        self.pred_dict = {}
        self.save_dict = {}
        self.loss_dict = {}

    def summarize_losses(self, loss_dict):
        total_loss = 0
        for key, item in self.loss_weights.items():
            if key in loss_dict:
                total_loss += loss_dict[key] * item
        loss_dict['total_loss'] = total_loss
        self.loss_dict = loss_dict

    def update(self):
        self.pred_dict = self.net(self.feed_dict)
        self.compute_loss()
        self.loss_dict['total_loss'].backward()

    def set_data(self, data):
        self.feed_dict = {}
        for key, item in data.items():
            if key in [""]:
                continue
            item = item.float().to(self.device)
            self.feed_dict[key] = item

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def test(self, save=False, no_eval=False, epoch=0):
        pass


class IPDFModel(BaseModel):
    def __init__(self, cfg):
        super(IPDFModel, self).__init__(cfg)
        self.net = IPDFFullNet(cfg)

    def compute_loss(self):
        pred_dict = self.pred_dict
        loss_dict = {}

        probability = pred_dict["probability"]  # [B]
        loss = -torch.mean(torch.log(probability))

        loss_dict["log_prob"] = loss

        self.summarize_losses(loss_dict)

        # for logging
        loss_dict["mean_probability"] = torch.mean(pred_dict["probability"])
        if "mean_local_prob" in pred_dict:
            loss_dict["mean_local_prob"] = torch.mean(pred_dict["local_prob"])
        if "mean_confidence" in pred_dict:
            loss_dict["mean_confidence"] = torch.mean(pred_dict["confidence"])

    def test(self, save=False, no_eval=False, epoch=0):
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict)
            sampled_rotation = self.net.sample_rotations(self.feed_dict)  # [B, 3, 3]
            self.pred_dict["sampled_rotation"] = sampled_rotation  # [B, 3, 3]

            if not no_eval:
                self.compute_loss()
