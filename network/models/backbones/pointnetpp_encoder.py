"""Pytorch-Geometric implementation of Pointnet++
Original source available at https://github.com/rusty1s/pytorch_geometric"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        """
        :param x: [B * N, N_feats]
        :param pos: [B * N, 3]
        :param batch: [B * N]
        :return:
        """
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class PointNetPPReg(torch.nn.Module):
    def __init__(self, cfg):
        super(PointNetPPReg, self).__init__()
        NUM_FEATS = 1  # original 25

        self.sa1_module = SAModule(0.5, 0.1, MLP([NUM_FEATS + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = torch.nn.Linear(1024, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 128)
        self.lin4 = torch.nn.Linear(128, 1)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)

    def forward(self, x, pos, batch):
        sa0_out = x, pos, batch
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, _, _ = self.sa3_module(*sa2_out)  # [B, 1024]

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        x = self.lin4(x)

        return x

class PointNetPPFeat(torch.nn.Module):
    def __init__(self, cfg):
        super(PointNetPPFeat, self).__init__()
        NUM_FEATS = 1  # original 25

        self.sa1_module = SAModule(0.5, 0.1, MLP([NUM_FEATS + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = torch.nn.Linear(1024, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 128)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)


    def forward(self, x, pos, batch):
        sa0_out = x, pos, batch
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, _, _ = self.sa3_module(*sa2_out)  # [B, 1024]

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.lin3(x)

        return x
