from copy import deepcopy, copy
import torch
import torch.nn as nn
from algo.pn_utils.maniskill_learn.networks.modules.activation import build_activation_layer
import ipdb
from algo.pn_utils.maniskill_learn.utils.data import dict_to_seq
from algo.pn_utils.maniskill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone
import ipdb

class PointBackbone(nn.Module):
    def __init__(self):
        super(PointBackbone, self).__init__()

    def forward(self, pcd):
        pcd = copy(pcd)
        if isinstance(pcd, dict):
            if 'pointcloud' in pcd:
                pcd['pcd'] = pcd['pointcloud']
                del pcd['pointcloud']
            assert 'pcd' in pcd
            return self.forward_raw(**pcd)
        else:
            return self.forward_raw(pcd)

    def forward_raw(self, pcd, state=None):
        raise NotImplementedError("")

@BACKBONES.register_module()
class SimplePointNetV0(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False, with_activation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """
        super(SimplePointNetV0, self).__init__()
        conv_cfg = deepcopy(conv_cfg)
        conv_cfg['mlp_spec'][0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, mask=None):
        """
        :param pcd: point cloud with states
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """

        if self.subtract_mean_coords:
            # Use xyz - mean xyz instead of original xyz
            mask = torch.ones_like(pcd[..., :1])
            xyz = pcd[:, :, :3]
            mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
            pcd = torch.cat((mean_xyz.repeat(1, xyz.shape[1], 1), xyz-mean_xyz, pcd[:, :, 3:]), dim=2)

        B, N = pcd.shape[:2]
        point_feature = self.conv_mlp(pcd.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = torch.ones_like(pcd[..., :1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        
        global_feature = global_feature.reshape(B, -1)


        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)


@BACKBONES.register_module()
class NaivePointNetV0(PointBackbone):
    def __init__(self, conv_cfg, state_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False, with_activation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """
        super(NaivePointNetV0, self).__init__()
        conv_cfg = deepcopy(conv_cfg)
        conv_cfg['mlp_spec'][0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.state_dim = state_cfg["mlp_spec"][0]
        self.state_mlp = build_backbone(state_cfg)
        self.global_mlp = build_backbone(mlp_cfg)
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, mask=None):
        """
        :param pcd: point cloud with states
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        pcd: num_envs*N*28
        cp:6维卷积
        partial:5维卷积
        """

        if self.subtract_mean_coords:
            # Use xyz - mean xyz instead of original xyz
            mask = torch.ones_like(pcd[..., :1])
            xyz = pcd[:, :, :3]
            mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
            pcd = torch.cat((mean_xyz.repeat(1, xyz.shape[1], 1), xyz-mean_xyz, pcd[:, :, 3:]), dim=2)

        B, N = pcd.shape[:2]
        point_feature = self.conv_mlp(pcd.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = torch.ones_like(pcd[..., :1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        
        if self.max_mean_mix_aggregation:

            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
            # ipdb.set_trace()
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)

        state = pcd[:, 0, -self.state_dim:]
        state_feature = self.state_mlp(state)

        global_feature = torch.cat([global_feature, state_feature], dim=-1)

        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)


@BACKBONES.register_module()
class PointNetV0(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False, with_activation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """

        super(PointNetV0, self).__init__()
        conv_cfg = deepcopy(conv_cfg)
        conv_cfg['mlp_spec'][0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, state, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function)
        :param state: shape (l, state_shape) agent state and other information of robot
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """
        if isinstance(pcd, dict):
            pcd = pcd.copy()
            mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
            if self.subtract_mean_coords:
                # Use xyz - mean xyz instead of original xyz
                xyz = pcd['xyz']  # [B, N, 3]
                mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
                pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
                pcd['xyz'] = xyz - mean_xyz
            # Concat all elements like xyz, rgb, seg mask, mean_xyz
            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]

        B, N = pcd.shape[:2]
        state = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]
        point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)
        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)



@BACKBONES.register_module()
class SparseUnetV0(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False, with_activation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """

        super(SparseUnetV0, self).__init__()
        from algo.pn_utils.sparseunet import  SparseUnetBackbone
        self.sparse_unet_backbone = SparseUnetBackbone(pc_dim = 6, feature_dim = 128)
        # conv_cfg = deepcopy(conv_cfg)
        # conv_cfg['mlp_spec'][0] += int(subtract_mean_coords) * 3
        # self.conv_mlp = build_backbone(conv_cfg)
        # self.stack_frame = stack_frame
        # self.max_mean_mix_aggregation = max_mean_mix_aggregation
        # self.subtract_mean_coords = subtract_mean_coords
        # self.global_mlp = build_backbone(mlp_cfg)
        # self.with_activation = with_activation
        # if with_activation:
        #     self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, state, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function)
        :param state: shape (l, state_shape) agent state and other information of robot
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """
        if isinstance(pcd, dict):
            pcd = pcd.copy()
            mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
            if self.subtract_mean_coords:
                # Use xyz - mean xyz instead of original xyz
                xyz = pcd['xyz']  # [B, N, 3]
                mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
                pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
                pcd['xyz'] = xyz - mean_xyz
            # Concat all elements like xyz, rgb, seg mask, mean_xyz
            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]



        B, N = pcd.shape[:2]
        state = torch.cat([pcd, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]
        import pdb
        pdb.set_trace()
        
        point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        
        
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)
        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)


from algo.pn_utils.maniskill_learn.utils.meta import build_from_cfg

def getNaivePointNet(cfg) :

    stack_frame = 1

    nn_cfg=dict(
        type='NaivePointNetV0',
        conv_cfg=dict(
            type='ConvMLP',
            norm_cfg=None,
            mlp_spec=[cfg["input_feature_dim"], 256, 512],
            bias='auto',
            inactivated_output=False,
            conv_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        state_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], 1024, 256],
            bias='auto',
            inactivated_output=False,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[768 * stack_frame, 1024, cfg["feat_dim"]],
            bias='auto',
            inactivated_output=False,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        subtract_mean_coords=True,
        max_mean_mix_aggregation=True,
        stack_frame=stack_frame,
    )

    pointnet = build_from_cfg(nn_cfg, BACKBONES, None)

    return pointnet

def getPointNet(cfg) :
    stack_frame = 1
    nn_cfg=dict(
        type='SimplePointNetV0',
        conv_cfg=dict(
            type='ConvMLP',
            norm_cfg=None,
            mlp_spec=[cfg["input_feature_dim"], 128, 256],
            bias='auto',
            inactivated_output=False,
            conv_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[256 * stack_frame, 256, cfg["feat_dim"]],
            bias='auto',
            inactivated_output=False,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        subtract_mean_coords=True,
        max_mean_mix_aggregation=True,
        stack_frame=stack_frame,
    )

    pointnet = build_from_cfg(nn_cfg, BACKBONES, None)

    return pointnet

def getPointNet_(cfg, backbone_cfg=None) :
    stack_frame = 1
    if backbone_cfg is None:
        conv_layers = [128, 256]
        mlp_hidden_layers = [256]
    else:
        conv_layers = backbone_cfg['conv_layers']
        mlp_hidden_layers = backbone_cfg['mlp_hidden_layers']

    nn_cfg=dict(
        type='SimplePointNetV0',
        conv_cfg=dict(
            type='ConvMLP',
            norm_cfg=None,
            mlp_spec=[cfg["input_feature_dim"]] + conv_layers,
            bias='auto',
            inactivated_output=False,
            conv_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[conv_layers[-1] * stack_frame] + mlp_hidden_layers + [cfg["feat_dim"]],
            bias='auto',
            inactivated_output=backbone_cfg['mlp_inactivate_output'],
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        subtract_mean_coords=True,
        max_mean_mix_aggregation=True,
        stack_frame=stack_frame,
    )

    pointnet = build_from_cfg(nn_cfg, BACKBONES, None)

    return pointnet

def getNewPointNet(cfg) :

    stack_frame = 1
    nn_cfg=dict(
        type='NaivePointNetV0',
        conv_cfg=dict(
            type='ConvMLP',
            norm_cfg=None,
            mlp_spec=[cfg["input_feature_dim"], 128, 128],
            bias='auto',
            inactivated_output=False,
            conv_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        state_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], 128, 128],
            bias='auto',
            inactivated_output=False,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[384 * stack_frame, 128, cfg["feat_dim"]],
            bias='auto',
            inactivated_output=False,
            linear_init_cfg=dict(
                type='xavier_init',
                gain=1,
                bias=0,
            )
        ),
        subtract_mean_coords=True,
        max_mean_mix_aggregation=True,
        stack_frame=stack_frame,
    )

    pointnet = build_from_cfg(nn_cfg, BACKBONES, None)

    return pointnet

@BACKBONES.register_module()
class PointNetWithInstanceInfoV0(PointBackbone):
	def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None, with_activation=False,
				xyz_dim=3, mask_dim=0, state_dim=0):
		"""
		PointNet with instance segmentation masks.
		There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
		(where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

		For points of the same object, the same PointNet processes each frame and concatenates the
		representations from all frames to form the representation of that point type.

		Finally representations from the state and all types of points are passed through final attention
		to output a vector of representation.

		:param pcd_pn_cfg: configuration for building point feature extractor
		:param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
		:param stack_frame: num of the frame in the input
		:param num_objs: dimension of the segmentation mask
		:param transformer_cfg: if use transformer to aggregate the features from different objects
		"""
		super(PointNetWithInstanceInfoV0, self).__init__()

		self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
		self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
		self.state_mlp = build_backbone(state_mlp_cfg)
		self.global_mlp = build_backbone(final_mlp_cfg)

		self.stack_frame = stack_frame
		self.num_objs = num_objs
		assert self.num_objs > 0
		self.with_activation = with_activation
		if with_activation:
			self.activation = nn.Sigmoid()
		
		self.xyz_dim = xyz_dim
		self.mask_dim = mask_dim
		self.state_dim = state_dim

	def forward(self, data):
		"""
		:param data:
				consists of pcd(B, N, 3), mask (B, N, M), state (B, N, S)
		:param state: shape (l, state_shape) state and other information of robot
		:return: [B,F] [batch size, final output]
		"""


		pcd = {'xyz': data["pc"][:, :, :3], 'rgb': data["pc"][:, :, 3:6], 'seg': data["pc"][:, :, 6:6+self.mask_dim]}
		state = data["state"]

		assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
		pcd = pcd.copy()
		seg = pcd.pop('seg')  # [B, N, NO]
		xyz = pcd['xyz']  # [B, N, 3]
		obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
		for i in range(self.num_objs):
			obj_masks.append(seg[..., i])
		obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

		obj_features = [] 
		obj_features.append(self.state_mlp(state))
		for i in range(len(obj_masks)):
			obj_mask = obj_masks[i]
			obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
			# print('X', obj_features[-1].shape)
		if self.attn is not None:
			obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
			new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
			non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
			non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
			obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]		   
			global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
		else:
			global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
		# print('Y', global_feature.shape)
		x = self.global_mlp(global_feature)
		# print(x)
		if self.with_activation:
			return self.activation(x)
		return x

def getPointNetWithInstanceInfo(cfg):

    stack_frame = 1
    num_heads = 4
    F = 128

    nn_cfg=dict(
        type='PointNetWithInstanceInfoV0',
        stack_frame=stack_frame,
        num_objs=cfg["mask_dim"],
        xyz_dim=3,
        mask_dim=cfg["mask_dim"],
        state_dim=cfg["state_dim"],
        pcd_pn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=[cfg["state_dim"] + cfg["pc_dim"], 128, 128],
                bias='auto',
                inactivated_output=True,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[128 * stack_frame, 128, 128],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True
        ),
        state_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], 128, 128],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),                            
        transformer_cfg=dict(
            type='TransformerEncoder',
            block_cfg=dict(
                attention_cfg=dict(
                    type='MultiHeadSelfAttention',
                    embed_dim=128,
                    num_heads=num_heads,
                    latent_dim=32,
                    dropout=0.1,
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[128, 128], #768,
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                dropout=0.1,
            ),
            pooling_cfg=dict(
                embed_dim=128,
                num_heads=num_heads,
                latent_dim=32,
            ),
            mlp_cfg=None,
            num_blocks=2,
        ),
        final_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[128, cfg["output_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
    )

    PointNetWithInstanceInfo = build_from_cfg(nn_cfg, BACKBONES, None)

    return PointNetWithInstanceInfo

@BACKBONES.register_module()
class SparseUnetWithInstanceInfoV0(PointBackbone):
	def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None, with_activation=False,
				xyz_dim=3, mask_dim=0, state_dim=0):
		"""
		PointNet with instance segmentation masks.
		There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
		(where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

		For points of the same object, the same PointNet processes each frame and concatenates the
		representations from all frames to form the representation of that point type.

		Finally representations from the state and all types of points are passed through final attention
		to output a vector of representation.

		:param pcd_pn_cfg: configuration for building point feature extractor
		:param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
		:param stack_frame: num of the frame in the input
		:param num_objs: dimension of the segmentation mask
		:param transformer_cfg: if use transformer to aggregate the features from different objects
		"""
		super(SparseUnetWithInstanceInfoV0, self).__init__()

		self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
		self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
		self.state_mlp = build_backbone(state_mlp_cfg)
		self.global_mlp = build_backbone(final_mlp_cfg)

		self.stack_frame = stack_frame
		self.num_objs = num_objs
		assert self.num_objs > 0
		self.with_activation = with_activation
		if with_activation:
			self.activation = nn.Sigmoid()
		
		self.xyz_dim = xyz_dim
		self.mask_dim = mask_dim
		self.state_dim = state_dim

	def forward(self, data):
		"""
		:param data:
				consists of pcd(B, N, 3), mask (B, N, M), state (B, N, S)
		:param state: shape (l, state_shape) state and other information of robot
		:return: [B,F] [batch size, final output]
		"""


		pcd = {'xyz': data["pc"][:, :, :3], 'rgb': data["pc"][:, :, 3:6], 'seg': data["pc"][:, :, 6:6+self.mask_dim]}
		state = data["state"]

		assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
		pcd = pcd.copy()
		seg = pcd.pop('seg')  # [B, N, NO]
		xyz = pcd['xyz']  # [B, N, 3]
		obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
		for i in range(self.num_objs):
			obj_masks.append(seg[..., i])
		obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

		obj_features = [] 
		obj_features.append(self.state_mlp(state))
		for i in range(len(obj_masks)):
			obj_mask = obj_masks[i]
			obj_features.append(self.pcd_pns[i].forward_raw(pcd, state, obj_mask))  # [B, F]
			# print('X', obj_features[-1].shape)
		if self.attn is not None:
			obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
			new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
			non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
			non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
			obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]		   
			global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
		else:
			global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
		# print('Y', global_feature.shape)
		x = self.global_mlp(global_feature)
		# print(x)
		if self.with_activation:
			return self.activation(x)
		return x

def getSparseUnetWithInstanceInfo(cfg):

    stack_frame = 1
    num_heads = 4
    F = 128

    nn_cfg=dict(
        type='SparseUnetWithInstanceInfoV0',
        stack_frame=stack_frame,
        num_objs=cfg["mask_dim"],
        xyz_dim=3,
        mask_dim=cfg["mask_dim"],
        state_dim=cfg["state_dim"],
        pcd_pn_cfg=dict(
            type='SparseUnetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=[cfg["state_dim"] + cfg["pc_dim"], 128, 128],
                bias='auto',
                inactivated_output=True,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[128 * stack_frame, 128, 128],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True
        ),
        state_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], 128, 128],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),                            
        transformer_cfg=dict(
            type='TransformerEncoder',
            block_cfg=dict(
                attention_cfg=dict(
                    type='MultiHeadSelfAttention',
                    embed_dim=128,
                    num_heads=num_heads,
                    latent_dim=32,
                    dropout=0.1,
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[128, 768, 128],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                dropout=0.1,
            ),
            pooling_cfg=dict(
                embed_dim=128,
                num_heads=num_heads,
                latent_dim=32,
            ),
            mlp_cfg=None,
            num_blocks=2,
        ),
        final_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[128, cfg["output_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
    )

    PointNetWithInstanceInfo = build_from_cfg(nn_cfg, BACKBONES, None)

    return PointNetWithInstanceInfo

    


@BACKBONES.register_module()
class PointNetDex(PointBackbone):
    def __init__(self, conv_cfg, mlp_cfg, stack_frame=1, subtract_mean_coords=False, max_mean_mix_aggregation=False, with_activation=False):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick 
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """

        super(PointNetDex, self).__init__()
        conv_cfg = deepcopy(conv_cfg)
        conv_cfg['mlp_spec'][0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, mask=None):
        """
        :param pcd: point cloud
                xyz: shape (l, n_points, 3)
                rgb: shape (l, n_points, 3)
                seg: shape (l, n_points, n_seg) (unused in this function)
        :param state: shape (l, state_shape) agent state and other information of robot
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """
        if isinstance(pcd, dict):
            pcd = pcd.copy()
            mask = torch.ones_like(pcd['xyz'][..., :1]) if mask is None else mask[..., None]  # [B, N, 1]
            if self.subtract_mean_coords:
                # Use xyz - mean xyz instead of original xyz
                xyz = pcd['xyz']  # [B, N, 3]
                mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
                pcd['mean_xyz'] = mean_xyz.repeat(1, xyz.shape[1], 1)
                pcd['xyz'] = xyz - mean_xyz
            # Concat all elements like xyz, rgb, seg mask, mean_xyz
            pcd = torch.cat(dict_to_seq(pcd)[1], dim=-1)
        else:
            mask = torch.ones_like(pcd[..., :1]) if mask is None else mask[..., None]  # [B, N, 1]

        B, N = pcd.shape[:2]
        state = pcd  # [B, N, CS]
        point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]
        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]
        global_feature = global_feature.reshape(B, -1)
        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)



@BACKBONES.register_module()
class PointNetWithInstanceInfoDex(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None, with_activation=False,
                xyz_dim=6, mask_dim=0):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithInstanceInfoDex, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
        assert self.num_objs > 0
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()
        
        self.xyz_dim = xyz_dim
        self.mask_dim = mask_dim
        # self.state_dim = state_dim

    def forward_raw(self, data):
        """
        :param data:
                consists of pcd(B, N, 6), mask (B, N, M), state (B, N, S)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """

        import pdb
        pdb.set_trace()

        pcd = {'xyz': data[:, :, :3], 'rgb': data[:, :, 3:6], 'seg': data[:, :, 6:6+self.mask_dim]}
        # state = data[:, 0, 6+self.mask_dim:]

        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 6]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = [] 
        # obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)
        x = self.global_mlp(global_feature)
        # print(x)
        if self.with_activation:
            return self.activation(x)
        return x


def getPointNetWithInstanceInfoDex(cfg):

    stack_frame = 1
    num_heads = 4
    F = cfg["feature_dim"]

    # import pdb
    # pdb.set_trace()
    nn_cfg=dict(
        type='PointNetWithInstanceInfoDex',
        stack_frame=stack_frame,
        num_objs=cfg["mask_dim"],
        xyz_dim=3,
        mask_dim=cfg["mask_dim"],
        # state_dim=cfg["state_dim"],
        pcd_pn_cfg=dict(
            type='PointNetDex',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=[cfg["state_dim"] + cfg["pc_dim"], cfg["feature_dim"], cfg["feature_dim"]],
                bias='auto',
                inactivated_output=True,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[cfg["feature_dim"] * stack_frame, cfg["feature_dim"], cfg["feature_dim"]],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True
        ),
        state_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], cfg["feature_dim"], cfg["feature_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),                            
        transformer_cfg=dict(
            type='TransformerEncoder',
            block_cfg=dict(
                attention_cfg=dict(
                    type='MultiHeadSelfAttention',
                    embed_dim=cfg["feature_dim"],
                    num_heads=num_heads,
                    latent_dim=32,
                    dropout=0.1,
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[cfg["feature_dim"], 768, cfg["feature_dim"]],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                dropout=0.1,
            ),
            pooling_cfg=dict(
                embed_dim=cfg["feature_dim"],
                num_heads=num_heads,
                latent_dim=32,
            ),
            mlp_cfg=None,
            num_blocks=2,
        ),
        final_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["feature_dim"], cfg["output_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
    )

    PointNetWithInstanceInfo = build_from_cfg(nn_cfg, BACKBONES, None)

    return PointNetWithInstanceInfo

@BACKBONES.register_module()
class SparseUnetWithInstanceInfoDex(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None, with_activation=False,
                xyz_dim=6, mask_dim=0):
        """
        PointNet with instance segmentation masks.
        There is one MLP that processes the agent state, and (num_obj + 2) PointNets that process background points
        (where all masks = 0), points from some objects (where some mask = 1), and the entire point cloud, respectively.

        For points of the same object, the same PointNet processes each frame and concatenates the
        representations from all frames to form the representation of that point type.

        Finally representations from the state and all types of points are passed through final attention
        to output a vector of representation.

        :param pcd_pn_cfg: configuration for building point feature extractor
        :param state_mlp_cfg: configuration for building the MLP that processes the agent state vector
        :param stack_frame: num of the frame in the input
        :param num_objs: dimension of the segmentation mask
        :param transformer_cfg: if use transformer to aggregate the features from different objects
        """
        super(PointNetWithInstanceInfoDex, self).__init__()

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None
        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.stack_frame = stack_frame
        self.num_objs = num_objs
        assert self.num_objs > 0
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()
        
        self.xyz_dim = xyz_dim
        self.mask_dim = mask_dim
        # self.state_dim = state_dim

    def forward_raw(self, data):
        """
        :param data:
                consists of pcd(B, N, 6), mask (B, N, M), state (B, N, S)
        :param state: shape (l, state_shape) state and other information of robot
        :return: [B,F] [batch size, final output]
        """

        import pdb
        pdb.set_trace()

        pcd = {'xyz': data[:, :, :3], 'rgb': data[:, :, 3:6], 'seg': data[:, :, 6:6+self.mask_dim]}
        # state = data[:, 0, 6+self.mask_dim:]

        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 6]
        obj_masks = [1. - (torch.sum(seg, dim=-1) > 0.5).type(xyz.dtype)]  # [B, N], the background mask
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        obj_features = [] 
        # obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            obj_features.append(self.pcd_pns[i].forward_raw(pcd, obj_mask))  # [B, F]
            # print('X', obj_features[-1].shape)
        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]
        # print('Y', global_feature.shape)
        x = self.global_mlp(global_feature)
        # print(x)
        if self.with_activation:
            return self.activation(x)
        return x


def getSparseUnetWithInstanceInfoDex(cfg):

    stack_frame = 1
    num_heads = 4
    F = cfg["feature_dim"]

    # import pdb
    # pdb.set_trace()
    nn_cfg=dict(
        type='SparseUnetWithInstanceInfoDex',
        stack_frame=stack_frame,
        num_objs=cfg["mask_dim"],
        xyz_dim=3,
        mask_dim=cfg["mask_dim"],
        # state_dim=cfg["state_dim"],
        pcd_pn_cfg=dict(
            type='SparseUnetDex',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=[cfg["state_dim"] + cfg["pc_dim"], cfg["feature_dim"], cfg["feature_dim"]],
                bias='auto',
                inactivated_output=True,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[cfg["feature_dim"] * stack_frame, cfg["feature_dim"], cfg["feature_dim"]],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True
        ),
        state_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["state_dim"], cfg["feature_dim"], cfg["feature_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),                            
        transformer_cfg=dict(
            type='TransformerEncoder',
            block_cfg=dict(
                attention_cfg=dict(
                    type='MultiHeadSelfAttention',
                    embed_dim=cfg["feature_dim"],
                    num_heads=num_heads,
                    latent_dim=32,
                    dropout=0.1,
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[cfg["feature_dim"], 768, cfg["feature_dim"]],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                dropout=0.1,
            ),
            pooling_cfg=dict(
                embed_dim=cfg["feature_dim"],
                num_heads=num_heads,
                latent_dim=32,
            ),
            mlp_cfg=None,
            num_blocks=2,
        ),
        final_mlp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[cfg["feature_dim"], cfg["output_dim"]],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
    )

    PointNetWithInstanceInfo = build_from_cfg(nn_cfg, BACKBONES, None)

    return PointNetWithInstanceInfo
