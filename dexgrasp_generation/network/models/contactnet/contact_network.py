import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import os
from os.path import join as pjoin
from network.models.backbones.pointnet_encoder import PointNetEncoder


class ContactMapNet(nn.Module):
    def __init__(self, cfg):
        super(ContactMapNet, self).__init__()

        num_obj_pc = cfg["dataset"]["num_obj_points"]
        num_hand_pc = cfg["dataset"]["num_hand_points"]
        out_channel = cfg["model"]["network"]["out_channel"]  # 1 or 10

        channel = 3
        self.feat_o = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel, use_stn=False)  # feature trans True
        self.feat_h = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel, use_stn=False)  # feature trans True
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, out_channel, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.convfuse = nn.Conv1d(num_obj_pc + num_hand_pc, num_obj_pc, 1)
        self.bnfuse = nn.BatchNorm1d(num_obj_pc)

    def forward(self, dic): # obj_pc, hand_pc):
        """
        :param obj_pc: [B, NO, 3]
        :param hand_pc: [B, NH, 3]
        :return: [B, NO] or [B, NO, C]
        """
        obj_pc = dic['canon_obj_pc']
        hand_pc = dic['observed_hand_pc']

        x = obj_pc.transpose(-1, -2)  # [B, 3, NO]
        hand = hand_pc.transpose(-1, -2)  # [B, 3, NH]

        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # for obj
        x, obj_local_feat, trans_feat = self.feat_o(x)  # x: [B, 1024], local: [B, 64, NO]
        # for hand
        hand, hand_local_feat, trans_feat2 = self.feat_h(hand)  # hand: [B, 1024]
        # fuse feature of object and hand
        # [B, 1024] -> [B, 1024, 1] -> [B, 1024, NO]
        fused_global = (hand + x).unsqueeze(-1).expand(-1, -1, obj_local_feat.shape[-1])
        # ([B, 1024, NO], [B, 64, NO]) -> [B, 1088, NO]
        fused_feat = torch.cat((fused_global, obj_local_feat), dim=1)
        # x = torch.cat((x, hand), dim=2).permute(0,2,1).contiguous()  # [B, NO+NH, 1088]
        # x = F.relu(self.bnfuse(self.convfuse(x)))  # [B, N, 1088]
        # x = x.permute(0,2,1).contiguous()  # [B, 1088, N]
        # inference cmap
        x = fused_feat

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
        x = self.conv4(x)  # [B, 1, N] or [B, C, N]
        x = x.transpose(2,1).contiguous()  # [B, N, 1 or C]
        if x.shape[-1] == 1:
            x = torch.sigmoid(x)
            x = x.view(batchsize, n_pts)  # n_pts  [B, N]
        else:
            x = F.log_softmax(x, dim=-1)  # [B, N, C]

        return dict(contact_map=x)
