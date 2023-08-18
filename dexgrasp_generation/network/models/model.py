import torch.nn as nn
import torch
import os
import sys
from os.path import join as pjoin
from abc import abstractmethod
from copy import deepcopy
from hydra import compose
from omegaconf.omegaconf import open_dict

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, '..'))
sys.path.insert(0, pjoin(base_path, '..', '..'))

from network.models.loss import discretize_gt_cm
from network.models.graspipdf.ipdf_network import IPDFFullNet
from network.models.graspglow.glow_network import DexGlowNet
from network.models.contactnet.contact_network import ContactMapNet


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
        # loss_dict['total_loss'] = total_loss
        total_loss = float(total_loss)
        loss_tensor = torch.tensor([total_loss], requires_grad=True)
        loss_dict['total_loss'] = loss_tensor
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
            if type(item) == torch.Tensor:
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
        self.net = IPDFFullNet(cfg).to(self.device)

    def compute_loss(self):
        pred_dict = self.pred_dict
        loss_dict = {}

        if "probability" in pred_dict:
            probability = pred_dict["probability"]  # [B]
            loss = -torch.mean(torch.log(probability))

            loss_dict["nll"] = loss

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

class GlowModel(BaseModel):
    def __init__(self, cfg):
        super(GlowModel, self).__init__(cfg)
        if cfg['model']['joint_training']:
            rotation_cfg = compose(f"{cfg['model']['rotation_net']['type']}_config")
            with open_dict(rotation_cfg):
                rotation_cfg['device'] = self.device
            self.rotation_net = IPDFFullNet(rotation_cfg).to(self.device)
            contact_cfg = compose(f"{cfg['model']['contact_net']['type']}_config")
            with open_dict(contact_cfg):
                contact_cfg['device'] = self.device
            self.contact_net = ContactMapNet(contact_cfg).to(self.device)
        else:
            self.rotation_net = None
            self.contact_net = None
        self.net = DexGlowNet(cfg, self.rotation_net, self.contact_net).to(self.device)

    def compute_loss(self):
        pred_dict = self.pred_dict
        loss_dict = {}

        if 'nll' in pred_dict:
            loss_dict['nll'] = torch.mean(pred_dict['nll'])
            loss_dict['cmap_loss'] = torch.mean(pred_dict['cmap_loss'])

            self.summarize_losses(loss_dict)

        # for logging
        for key in pred_dict.keys():
            if 'cmap_part' in key:
                loss_dict[key] = torch.mean(pred_dict[key])

    def test(self, save=False, no_eval=False, epoch=0):
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict)
            self.pred_dict.update(self.net.sample(self.feed_dict))

            if not no_eval:
                self.compute_loss()

class ContactModel(BaseModel):
    def __init__(self, cfg):
        super(ContactModel, self).__init__(cfg)
        self.net = ContactMapNet(cfg).to(self.device)

        self.cm_loss = nn.MSELoss()
        self.cm_bin_loss = nn.CrossEntropyLoss()

    def compute_loss(self):
        loss_dict = {}

        gt_contact_map = self.feed_dict['contact_map']
        pred_contact_map = self.pred_dict['contact_map']

        if len(pred_contact_map.shape) == 2:
            # pred_contact_map: [B, N]
            loss_dict["contact_map"] = self.cm_loss(pred_contact_map, gt_contact_map)
        elif len(pred_contact_map.shape) == 3:
            # pred_contact_map: [B, N, 10]
            gt_bins = discretize_gt_cm(gt_contact_map, num_bins=pred_contact_map.shape[-1])  # [B, N, 10]
            gt_bins_labels = torch.argmax(gt_bins, dim=-1)  # [B, N]
            pred_contact_map = pred_contact_map.transpose(2, 1)  # [B, 10, N]
            loss_dict["contact_map"] = self.cm_bin_loss(pred_contact_map, gt_bins_labels)

        self.summarize_losses(loss_dict)

    def test(self, save=False, no_eval=False, epoch=0):
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict)
            if not no_eval:
                self.compute_loss()
