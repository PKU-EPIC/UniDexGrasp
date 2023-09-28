import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

import math

from collections import OrderedDict

import os
from os.path import join as pjoin
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '..', '..'))

from network.models.model import IPDFModel, GlowModel, ContactModel
from utils.global_utils import update_dict


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

def get_scheduler(optimizer, cfg, it=-1):
    scheduler = None
    if optimizer is None:
        return scheduler
    if 'lr_policy' not in cfg or cfg['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif cfg['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=cfg['lr_step_size'],
                                        gamma=cfg['lr_gamma'],
                                        last_epoch=it)
    else:
        assert 0, '{} not implemented'.format(cfg['lr_policy'])
    return scheduler

def get_optimizer(params, cfg):
    if len(params) == 0:
        return None
    if cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg['learning_rate'],
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=cfg['learning_rate'],
            momentum=0.9)
    else:
        assert 0, "Unsupported optimizer type {}".format(cfg['optimizer'])
    return optimizer

def get_last_model(dirname, key=""):
    if not os.path.exists(dirname):
        return None
    models = [pjoin(dirname, f) for f in os.listdir(dirname) if
              os.path.isfile(pjoin(dirname, f)) and
              key in f and ".pt" in f]
    if models is None or len(models) == 0:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name


class Trainer(nn.Module):
    def __init__(self, cfg, logger):
        super(Trainer, self).__init__()

        self.cfg = cfg
        self.device = cfg["device"]
        self.ckpt_dir = pjoin(cfg["exp_dir"], "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if cfg["network_type"] == "ipdf":
            self.model = IPDFModel(cfg)
        elif cfg["network_type"] == "glow":
            self.model = GlowModel(cfg)
        elif cfg["network_type"] == "cm_net":
            self.model = ContactModel(cfg)

        self.optimizer = get_optimizer([p for p in self.model.parameters() if p.requires_grad], cfg)
        self.scheduler = get_scheduler(self.optimizer, cfg)

        self.apply(weights_init(cfg['weight_init']))

        self.epoch = 1
        self.iteration = 0
        self.loss_dict = {}

        self.logger = logger

    def log_string(self, out_str):
        self.logger.info(out_str)

    def step_epoch(self):
        self.epoch += 1

        cfg = self.cfg
        if self.scheduler is not None and self.scheduler.get_last_lr()[0] > cfg['lr_clip']:
            self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]

        self.log_string("Epoch %d/%d, learning rate = %f" % (
            self.epoch, cfg['total_epoch'], self.lr))

        momentum = cfg['momentum_original'] * (
                cfg['momentum_decay'] ** (self.epoch // cfg['momentum_step_size']))
        momentum = max(momentum, cfg['momentum_min'])
        self.log_string("BN momentum updated to %f" % momentum)
        self.momentum = momentum

    def resume(self):
        def get_model(dir, resume_epoch):
            last_model_name = get_last_model(dir)
            print('last model name', last_model_name)
            if resume_epoch is not None and resume_epoch > 0:
                specified_model = pjoin(dir, f"model_{resume_epoch:04d}.pt")
                if os.path.exists(specified_model):
                    last_model_name = specified_model
            return last_model_name

        ckpt = OrderedDict()

        model_name = get_model(self.ckpt_dir, self.cfg.get('resume_epoch', None))

        if model_name is None:
            self.log_string('Initialize from 0')
        else:
            state_dict = torch.load(model_name, map_location=self.device)
            self.epoch = state_dict['epoch']
            self.iteration = state_dict['iteration']
            ckpt.update(state_dict['model'])

            if self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state_dict['optimizer'])
                except (ValueError, KeyError):
                    pass  # when new params are added, just give up on the old states
                self.scheduler = get_scheduler(self.optimizer, self.cfg, self.epoch)

            self.log_string('Resume from epoch %d' % self.epoch)

        try:
            self.model.load_state_dict(ckpt)
        except:
            # load old version glow
            new_ckpt = OrderedDict()
            for name in ckpt.keys():
                new_ckpt[name.replace('backbone.', '')] = ckpt[name]
            self.model.load_state_dict(new_ckpt, strict=False)

        print(self.model)

        return self.epoch

    def save(self, name=None):
        epoch = self.epoch
        if name is None:
            name = f'model_{epoch:06d}'
        savepath = pjoin(self.ckpt_dir, "%s.pt" % name)
        state = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, savepath)
        self.log_string("Saving model at epoch {}, path {}".format(epoch, savepath))

    def update(self, data):
        self.optimizer.zero_grad()
        self.model.train()
        self.model.set_data(data)
        self.model.update()
        loss_dict = self.model.loss_dict
        update_dict(self.loss_dict, loss_dict)
        self.optimizer.step()
        self.iteration += 1

        return loss_dict

    def test(self, data, save=False, no_eval=False):
        self.model.eval()
        self.model.set_data(data)
        self.model.test()
        pred_dict = self.model.pred_dict
        loss_dict = self.model.loss_dict
        return pred_dict, loss_dict
