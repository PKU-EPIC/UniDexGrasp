import os
import sys
from os.path import join as pjoin
from network.models.backbones.pointnet_encoder import PointNetEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from nflows.flows.glow import ConditionalGlow # from ProHMR
from utils.hand_model import AdditionalLoss

class Glow(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        features = cfg['model']['flow']['points']+3
        context_features = cfg['model']['flow']['feature_dim']
        hidden_features = cfg['model']['flow']['hidden_dim']
        num_layers = cfg['model']['flow']['layer']
        num_blocks_per_layer = cfg['model']['flow']['block']
        self.flow = ConditionalGlow(features, hidden_features, num_layers, num_blocks_per_layer, context_features=context_features)
        self.register_buffer('initialized', torch.tensor(False))
    
    def initialize(self, batch, context):
        with torch.no_grad():
            self.initialized |= True
            _, _ = self.flow.log_prob(batch, context)
    
    def log_prob(self, batch, context):
        batch = self.to_flow(batch)
        if not self.initialized:
            self.initialize(batch, context)
        log_prob, _ = self.flow.log_prob(batch, context)
        return log_prob
    
    def sample_and_log_prob(self, num_samples, context):
        samples, log_prob, _ = self.flow.sample_and_log_prob(num_samples, context=context)
        samples = self.from_flow(samples)
        return samples, log_prob
    
    def to_flow(self, batch):
        gt_trans, gt_qpos = batch
        return torch.cat([gt_trans, gt_qpos], dim=-1)

    def from_flow(self, batch):
        gt_trans, gt_qpos = torch.split(batch, [3, 22], dim=-1)
        return gt_trans, gt_qpos

class DexGlowNet(nn.Module):
    """
    PointNet or Point Transformer all integrated in this class.
    """
    def __init__(self, cfg, rotation_net=None, contact_net=None):
        super(DexGlowNet, self).__init__()
        self.cfg = cfg
        self.sample_func = rotation_net.sample_rotations if rotation_net else None #sample_func
        self.cmap_func = AdditionalLoss(cfg['model']['tta'], cfg['device'], cfg['dataset']['num_obj_points'], cfg['dataset']['num_hand_points'], contact_net) if contact_net else None 

        # [B, 3, N] -> [B, 128, N]
        if cfg['model']['network']['type'] == 'pointnet':
            self.encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3, use_stn=True)
        else:
            raise NotImplementedError(f"backbone {cfg['model']['network']['type']} not implemented")
        
        self.sample_num = cfg["model"]["sample_num"]

        '''
        self.hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            n_surface_points=1000,
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device="cuda"
        )
        '''
        self.flow = Glow(cfg)

    def prepare_data(self, obj_pc):
        """
        Prepare inputs for PointNet pipeline.
        :param obj_pc: [B, NO, 3]
        :return x: [B * N, 1], all ones for initial features
        :return pos: [B * N, 3]
        :return batch: [B * N], indicating which batch the points in pos belong to.
        """
        batch_size, N, _ = obj_pc.shape
        device = obj_pc.device
        pos = obj_pc.contiguous().view(batch_size * N, 3)
        # assert pos.shape[1] == 3, f"pos.shape[1] should be 3, but got pos.shape: {pos.shape}"

        x = torch.ones((batch_size * N, 1), device=device, dtype=torch.float32)  # [B * N, 1]
        batch = torch.empty((batch_size, N), device=device, dtype=torch.long)  # [B, N]
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)  # [B * N]
        assert batch.shape[0] == batch_size * N,\
               f"batch should be [{batch_size} * {N}] = [{batch_size * N}], but got {batch.shape} instead."

        return x, pos, batch
    
    def sample(self, dic):
        ret_dict = {}

        if not 'canon_obj_pc' in dic.keys():
            dic['canon_obj_pc'] = torch.einsum('nab,ncb->nac', dic['obj_pc'], dic['sampled_rotation'])
            plane = dic['plane'].clone()
            plane[:, :3] = torch.einsum('nbc,nc->nb', dic['sampled_rotation'], plane[:, :3])
            ret_dict['canon_plane'] = plane
            
        pc = dic['canon_obj_pc']
        batch_size=pc.shape[0]
        pc_transformed = pc.transpose(1, 2).contiguous()  # [B, 3, N]
        feat, _, _ = self.encoder(pc_transformed)

        samples, log_prob = self.flow.sample_and_log_prob(self.sample_num, feat)
        trans_samples, qpos_samples = samples
        max_index = torch.argmax(log_prob, dim=-1)
        arange = torch.arange(0, batch_size, device=qpos_samples.device, dtype=torch.long)
        sel_qpos = qpos_samples[arange, max_index].reshape(batch_size, -1)
        sel_trans = trans_samples[arange, max_index].reshape(batch_size, 3)

        ret_dict['canon_translation'] = sel_trans
        if 'sampled_rotation' in dic.keys():
            ret_dict['translation'] = torch.einsum('na,nab->nb', sel_trans, dic['sampled_rotation'])
        else:
            ret_dict['translation'] = sel_trans
        ret_dict['hand_qpos'] = sel_qpos
        ret_dict['sampled_canon_translation'] = trans_samples
        if 'sampled_rotation' in dic.keys():
            ret_dict['sampled_translation'] = torch.einsum('nka,nab->nkb', trans_samples, dic['sampled_rotation'])
        else:
            ret_dict['sampled_translation'] = trans_samples
        ret_dict['sampled_hand_qpos'] = qpos_samples
        ret_dict['entropy'] = -log_prob.mean(dim=-1)    
        ret_dict['canon_obj_pc'] = dic['canon_obj_pc']

        return ret_dict

    def forward(self, dic): 
        """
        :param x: [B, N, 3]
        :return: ret_dict:
            {
                "rotation": [B, R],
                "translation": [B, 3],
                "hand_qpos": [B, H]
            }

        """
        if not 'canon_translation' in dic:
            return dict()
            
        raw_pc = dic['obj_pc']
        pc = dic['canon_obj_pc']
        gt = (dic['canon_translation'], dic['hand_qpos'])

        batch_size=pc.shape[0]
        pc_transformed = pc.transpose(1, 2).contiguous()  # [B, 3, N]
        feat, _, _ = self.encoder(pc_transformed)

        ret_dict = {}

        ret_dict['nll'] = -self.flow.log_prob(gt, feat)

        if self.sample_func is not None:
            raw_plane = dic['plane']
            sr_rotation = self.sample_func({"obj_pc":raw_pc}).detach()
            sr_pc = torch.einsum('nab,ncb->nac', raw_pc, sr_rotation)
            sr_plane = torch.empty_like(raw_plane)
            sr_plane[:, :3] = torch.einsum('nbc,nc->nb', sr_rotation, raw_plane[:, :3])
            sr_plane[:, 3] = raw_plane[:, 3]

            sr_pc_transformed = sr_pc.transpose(1, 2).contiguous()  # [B, 3, N]
            sr_feat, _, _ = self.encoder(sr_pc_transformed)
            sr_samples, sr_log_prob = self.flow.sample_and_log_prob(self.sample_num, sr_feat)
            sr_trans_samples, sr_qpos_samples = sr_samples
            sr_max_index = torch.argmax(sr_log_prob, dim=-1)
            arange = torch.arange(0, batch_size, device=sr_max_index.device, dtype=torch.long)
            sr_sel_qpos = sr_qpos_samples[arange, sr_max_index].reshape(batch_size, -1)
            sr_sel_trans = sr_trans_samples[arange, sr_max_index].reshape(batch_size, 3)
            ret_dict['sr_rotation'] = sr_rotation
            ret_dict['sr_hand_qpos'] = sr_sel_qpos
            ret_dict['sr_translation'] = sr_sel_trans
            ret_dict['sr_sampled_hand_qpos'] = sr_qpos_samples
            ret_dict['sr_sampled_translation'] = sr_trans_samples
            sr_pc = sr_pc.unsqueeze(1).repeat(1,self.sample_num,1,1).reshape(batch_size*self.sample_num,-1,3)
            sr_plane = sr_plane.unsqueeze(1).repeat(1,self.sample_num,1).reshape(batch_size*self.sample_num,4)
            cmap_loss, cmap_losses = self.cmap_func(sr_pc, sr_plane, sr_trans_samples.reshape(batch_size*self.sample_num, 3), sr_qpos_samples.reshape(batch_size*self.sample_num, 22))
            ret_dict['cmap_loss']=cmap_loss.mean()#.reshape(batch_size,-1).mean(dim=-1)
            for key in cmap_losses.keys():
                ret_dict[f'cmap_part_{key}']=cmap_losses[key]
        else:
            ret_dict['cmap_loss'] = torch.zeros((batch_size,), device=pc.device, dtype=pc.dtype)

        return ret_dict