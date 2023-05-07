

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
# import pytorch_lightning as pl
from typing import List, Optional, Tuple

        
from perception.structures.point_cloud import PointCloud
from einops import rearrange, repeat
from perception.datasets.gapartnet_new import apply_voxelization, apply_voxelization_batch

from epic_ops.voxelize import voxelize
from epic_ops.reduce import segmented_maxpool
import functools
from algo.ppo_utils.sparse_unet_backbone import SparseUNet
import copy
import spconv.pytorch as spconv
from algo.ppo_utils.pointgroup_utils import (apply_nms, cluster_proposals, compute_ap,
                            compute_npcs_loss, filter_invalid_proposals,
                            get_gt_scores, segmented_voxelize)

class SparseUnetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        channels: List[int] = [16, 64, 112],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        use_domain_discrimination: bool = False
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.in_channels = pc_dim
        self.channels = channels

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.unet = SparseUNet.build(self.in_channels, channels, block_repeat, norm_fn)
        self.pn = getPointNet({
                'input_feature_dim': 6 + channels[0],
                'feat_dim': feature_dim,
            })


        if pretrained_model_path != None and pretrained_model_path != "None":
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
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]

        pc = PointCloud(scene_id=["train"],points=input_pc)

        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()

        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features, _ = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]
        # sem_logits = self.seg_head(voxel_features.features)[pc_voxel_id]
        # sem_preds = torch.argmax(sem_logits.detach(), dim=-1)
        # import pdb
        # pdb.set_trace()
        # pt_features[:] = 0.
        res = self.pn(torch.cat((input_pc[..., :6], pt_features.reshape(batch_size, -1, pt_features.shape[-1])), dim = -1))
        # res = self.pn(input_pc[..., :6])

        others = {}
        # others["sem_logits"] = sem_logits
        # others["sem_preds"] = sem_preds
        
        
        # for i in range(batch_size):
        #     if (global_feats.indices[:,0] == i).sum() == 0:
        #         continue ## ? have some bug, maybe
        #     res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res, others

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4

        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, pc.points.shape[-1])),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id