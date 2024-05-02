"""
Some code related to IPDF is credited to: https://github.com/google-research/google-research/tree/master/implicit_pdf
"""

import sys
import os
from os.path import join as pjoin

import torch.nn as nn
import torch
import numpy as np
from pytorch3d.transforms import random_rotations
from scipy.spatial.transform import Rotation as R

from network.models.backbones.pointnetpp_encoder import PointNetPPFeat


class PointNetPP(nn.Module):
    def __init__(self, cfg):
        super(PointNetPP, self).__init__()
        self.cfg = cfg

        self.backbone = PointNetPPFeat(cfg)

    def forward(self, obj_pc):
        """
        :param obj_pc: [B, NO, 3]
        :return: [B], alpha * ln(probability), for rotation_net
                or [B, 128] feat for pn_rotation_net
        """
        x, pos, batch = self.prepare_data(obj_pc)

        x = self.backbone(x, pos, batch)

        return x

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

        x = torch.ones((batch_size * N, 1), device=device, dtype=torch.float32)  # [B * N, 1]
        batch = torch.empty((batch_size, N), device=device, dtype=torch.long)  # [B, N]
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)  # [B * N]
        assert batch.shape[0] == batch_size * N,\
               f"batch should be [{batch_size} * {N}] = [{batch_size * N}], but got {batch.shape} instead."

        return x, pos, batch


class IPDFFullNet(nn.Module):
    def __init__(self, cfg):
        super(IPDFFullNet, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.grids = {}

        self.backbone = PointNetPP(cfg).to(self.device)

        number_fourier_components = cfg["model"]["number_fourier_components"]
        self.frequencies = torch.arange(number_fourier_components, dtype=torch.float32)  # [0]
        self.frequencies = torch.pow(2., self.frequencies).to(self.device)  # [1]
        self.len_rotation = 9

        if number_fourier_components == 0:
            self.len_query = self.len_rotation
        else:
            self.len_query = self.len_rotation * number_fourier_components * 2  # 18?

        self.implicit_model = ImplicitModel(self.len_query, cfg)

        self.num_train_queries = cfg["model"]["num_train_queries"]
        self.pre_generated_grid_queries = self.generate_queries(self.num_train_queries, mode="grid")  # [num, 3 ,3]
        self.pre_generated_grid_queries = torch.stack([
            matrix
            for matrix in self.pre_generated_grid_queries
        ])

    def forward(self, inputs):
        """
        Predict the probability for a rotation
        :param inputs:
            "obj_pc": [B, NO, 3]
            "world_frame_hand_rotation_mat": [B, 3, 3], the rotations to be queried, during training it is the gt_R
        :return: probability p(R|obj_pc)
        """
        if not 'world_frame_hand_rotation_mat' in inputs:
            return dict()

        obj_pc = inputs["obj_pc"]  # [B, NO, 3]

        feat = self.backbone(obj_pc)  # [B, 128]
        assert feat.shape[1] == 128, f"Incorrect feat.shape: {feat.shape}"

        gt_r = inputs["world_frame_hand_rotation_mat"]  # [B, 3, 3]
        query_rotations = self.generate_queries(self.num_train_queries)  # [num, 3, 3]
        delta_rot = torch.einsum("ij,bjk->bik", query_rotations[-1].T, gt_r)  # [B, 3, 3]
        query_rotations = torch.einsum('aij,bjk->baik', query_rotations, delta_rot)  # [B, num, 3, 3]

        query_rotations = query_rotations.reshape(-1, self.num_train_queries, self.len_rotation)  # [B, num, 9]
        query_rotations = self.positional_encoding(query_rotations)  # [B, num, 18]

        probs = self.implicit_model(feat, query_rotations)  # [B, num]
        probs = torch.nn.functional.softmax(probs, dim=1)  # [B, num]
        probs = probs * (query_rotations.shape[1] / np.pi ** 2)  # [B, num]
        prob = probs[:, -1]  # [B], -1 is our rotation anchor

        pred_dict = {"probability": prob}

        return pred_dict

    def predict_rotation(self, inputs):
        """
        Predict a rotation that has the highest prob.
        :param inputs:
            "obj_pc": [B, NO, 3]
        :return: R, s.t. R @ obj_pc has the highest probability
        """
        obj_pc = inputs["obj_pc"]
        feat = self.backbone(obj_pc).reshape(len(obj_pc), -1)   # [B, 128]
        assert feat.shape[1] == 128, f"Incorrect feat.shape: {feat.shape}"

        batch_size = feat.shape[0]

        query_rotations = self.pre_generated_grid_queries  # [num, 3, 3]
        query_rotations = query_rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, num, 3, 3]

        query_rotations_embedded = query_rotations.reshape(-1, query_rotations.shape[1],
                                                           self.len_rotation)  # [B, num, 9]
        query_rotations_embedded = self.positional_encoding(query_rotations_embedded)  # [B, num, 18]

        probs = self.implicit_model(feat, query_rotations_embedded)  # [B, num]
        probs = torch.nn.functional.softmax(probs, dim=1)  # [B, num]
        probs = probs * (query_rotations.shape[1] / np.pi ** 2)  # [B, num]

        sample_idx = probs.argmax(dim=1)  # [B]
        sample_idx = sample_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        sample_idx = sample_idx.expand(-1, -1, 3, 3)  # [B, 1, 3, 3]

        sampled_rotations = torch.gather(query_rotations, dim=1, index=sample_idx)  # [B, 1, 3, 3]
        sampled_rotations = sampled_rotations.squeeze(1)  # [B, 3, 3]

        return sampled_rotations

    def sample_rotations(self, inputs, sample_times=None):
        """
        :return: [B, 3, 3]
        """
        obj_pc = inputs["obj_pc"]  # [B, NO, 3]
        feat = self.backbone(obj_pc).reshape(len(obj_pc), -1)   # [B, 128]
        assert feat.shape[1] == 128, f"Incorrect feat.shape: {feat.shape}"

        if sample_times is None:
            batch_size = feat.shape[0]
        elif isinstance(sample_times, int):
            batch_size = sample_times
            feat = feat.expand(sample_times, -1)
        else:
            raise ValueError(f"Invalid sample_times: {sample_times}")

        query_rotations = self.pre_generated_grid_queries  # [num, 3, 3]
        query_rotations = query_rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, num, 3, 3]

        query_rotations_embedded = query_rotations.reshape(-1, query_rotations.shape[1],
                                                           self.len_rotation)  # [B, num, 9]
        query_rotations_embedded = self.positional_encoding(query_rotations_embedded)  # [B, num, 18]

        probs = self.implicit_model(feat, query_rotations_embedded)  # [B, num]
        probs = torch.nn.functional.softmax(probs, dim=1)  # [B, num]
        probs = probs * (query_rotations.shape[1] / np.pi ** 2)  # [B, num]

        sample_idx = uniform_sample(probs, device=probs.device)  # [B]
        sample_idx = sample_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        sample_idx = sample_idx.expand(-1, -1, 3, 3)  # [B, 1, 3, 3]

        sampled_rotations = torch.gather(query_rotations, dim=1, index=sample_idx)  # [B, 1, 3, 3]
        sampled_rotations = sampled_rotations.squeeze(1)  # [B, 3, 3]

        return sampled_rotations

    def generate_queries(self, number_queries, mode="random"):
        """Generate query rotations from SO(3).
        Args:
          number_queries: The number of queries.
          mode: 'random' or 'grid'; determines whether to generate rotations from
            the uniform distribution over SO(3), or use an equivolumetric grid.
        Returns:
          A tensor of rotation matrices, shape [num_queries, 3, 3].
        """
        if mode == 'random':
            return self.generate_queries_random(number_queries)
        elif mode == 'grid':
            return torch.tensor(self.get_closest_available_grid(number_queries), device=self.device)

    def generate_queries_random(self, number_queries):
        """Generate rotation matrices from SO(3) uniformly at random.
        Args:
          number_queries: The number of queries.
        Returns:
          A tensor of shape [number_queries, 3, 3].
        """
        random_rots = random_rotations(number_queries, torch.float32).to(self.device)
        return random_rots

    def get_closest_available_grid(self, number_queries=None):
        if not number_queries:
            number_queries = self.number_eval_queries
        # HEALPix-SO(3) is defined only on 72 * 8^N points; we find the closest
        # valid grid size (in log space) to the requested size.
        # The largest grid size we consider has 19M points.
        grid_sizes = 72 * 8 ** np.arange(7)
        size = grid_sizes[
            np.argmin(np.abs(np.log(number_queries) - np.log(grid_sizes)))
        ]
        if self.grids.get(size) is not None:
            return self.grids[size]
        else:
            print('Using grid of size %d. Requested was %d.', size,
                  number_queries)
            grid_created = False

            if not grid_created:
                self.grids[size] = np.float32(generate_healpix_grid(size=size))

            return self.grids[size]

    def positional_encoding(self, query_rotations):
        """This handles the positional encoding.
        Args:
          query_rotations: tensor of shape [N, len_rotation] or
            [bs, N, len_rotation].
        Returns:
          Tensor of shape [N, len_query] or [bs, N, len_query].
        """
        if self.frequencies.shape[0] == 0:
            return query_rotations

        query_rotations = torch.cat(
            [torch.sin(query_rotations * self.frequencies),
             torch.cos(query_rotations * self.frequencies)],
            dim=-1)  # [B, N, 18] WARNING: only applicable if self.frequencies == [1.]
        query_rotations = query_rotations.unsqueeze(0)  # [1, B, N, 18] WARNING: only applicable as above
        query_shape = query_rotations.shape
        if len(query_shape) == 4:
            query_rotations = query_rotations.permute((1, 2, 0, 3))  # [B, N, 1, 18]
            shape = list(query_shape[1:3]) + [self.len_query]  # [B, N, 18]
            query_rotations = query_rotations.reshape(shape)  # [B, N, 18]
        else:  # WARNING: it should not go into here
            query_rotations = query_rotations.permute((1, 0, 2))  # [N, B, 18]
            query_rotations = query_rotations.reshape(-1, self.len_query)  # [N * B, 18] why?
        return query_rotations

    def group_rotations(self, query_rotations, gt_r, num_neighbors):
        """
        :param query_rotations: [B, num, 3, 3]
        :param gt_r: [B, 3, 3]
        :param num_neighbors: nn: int, e.g. 64
        :return: [B, nn, 3, 3]
        """
        gt_r_ = gt_r.unsqueeze(1)  # [B, 1, 3, 3]
        _, rotation_err = self.r_delta_loss_func(query_rotations, gt_r_)  # [B, num]
        selection_idx = torch.argsort(rotation_err, dim=1)  # [B, num]
        selection_idx = selection_idx[:, :num_neighbors]  # [B, nn]

        selection_idx = selection_idx.unsqueeze(-1).unsqueeze(-1)  # [B, nn, 1, 1]
        selection_idx = selection_idx.expand(-1, -1, 3, 3)
        grouped_rotations = torch.gather(query_rotations, dim=1, index=selection_idx)  # [B, nn, 3, 3]

        return grouped_rotations


def generate_healpix_grid(recursion_level=None, size=None):
    """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).
    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.
    Args:
    recursion_level: An integer which determines the level of resolution of the
      grid.  The final number of points will be 72*8**recursion_level.  A
      recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
      for evaluation.
    size: A number of rotations to be included in the grid.  The nearest grid
      size in log space is returned.
    Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    """
    import healpy as hp  # pylint: disable=g-import-not-at-top

    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.) / np.log(8.))), 0)
    number_per_side = 2 ** recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    # Take these points on the sphere and
    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2 ** recursion_level, endpoint=False)
    polars = np.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        # Build up the rotations from Euler angles, zyz format
        rot_mats = R.from_euler("xyz",
                                np.stack([azimuths,
                                          np.zeros(number_pix),
                                          np.zeros(number_pix)], 1)
                                ).as_matrix()
        rot_mats = rot_mats @ R.from_euler("xyz",
                                           np.stack([np.zeros(number_pix),
                                                     np.zeros(number_pix),
                                                     polars], 1)
                                           ).as_matrix()
        rot_mats = rot_mats @ np.expand_dims(R.from_euler("xyz",
                                                          [tilt, 0., 0.]
                                                          ).as_matrix(), axis=0)
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = np.concatenate(grid_rots_mats, 0)
    return grid_rots_mats


def uniform_sample(probabilities, device=torch.device("cuda:0")):
    """
    :param probabilities: [..., N]
    :return: idx in range(N), shape: [...]
    """
    histo = torch.nn.functional.softmax(probabilities, dim=-1)
    original_size = histo.shape
    histo = torch.cumsum(histo, dim=-1)
    histo = histo.reshape(-1, original_size[-1])  # [?, 1]
    num_bins = histo.shape[-1]
    sample_point = torch.rand(histo.shape[:-1]).to(device)  # [?]

    hot_vector = [torch.logical_and(sample_point >= 0,
                                    sample_point < histo[..., 0])]
    for b in range(num_bins - 1):
        hot_vector.append(torch.logical_and(sample_point >= histo[..., b],
                                            sample_point < histo[..., b + 1]))  # [?]
    hot_vector = torch.stack(hot_vector, dim=-1).float()
    hot_vector = hot_vector.reshape(original_size)

    idx = torch.argmax(hot_vector, dim=-1)
    return idx


class ImplicitModel(nn.Module):
    def __init__(self, len_query, cfg):
        super(ImplicitModel, self).__init__()

        self.mlp_layer_sizes = cfg["model"]["mlp_layer_sizes"]  # [256, 256, 256]
        if cfg["model"]["network"]["type"] == "epn_net":
            self.input_feat_dim = cfg["model"]["network"]["equi_feat_mlps"][-1]  # 64
        elif cfg["model"]["network"]["type"] in ["pn_rotation_net", "kpconv_rot_net"]:
            self.input_feat_dim = 128
        self.query_dim = len_query  # 18
        self.out_dim = 1  # since we just want a prob

        mlp_dim = self.mlp_layer_sizes[0]
        self.feat_mlp = nn.Linear(self.input_feat_dim, mlp_dim)  # [B, C] -> [B, 256]
        self.query_mlp = nn.Conv1d(self.query_dim, mlp_dim, 1)  # [B, 18, nn] -> [B, 256, nn]

        self.mlps = []
        for i in range(1, len(self.mlp_layer_sizes)):
            self.mlps.append(nn.Conv1d(mlp_dim, self.mlp_layer_sizes[i], 1))
            mlp_dim = self.mlp_layer_sizes[i]
            self.mlps.append(nn.BatchNorm1d(mlp_dim))
            self.mlps.append(nn.ReLU())
        self.mlps.append(nn.Conv1d(mlp_dim, 1, 1))  # [B, 1, nn]

        self.mlps = nn.Sequential(*self.mlps)

    def forward(self, feat, query_embedded):
        """
        :param feat: [B, C]
        :param query_embedded: [B, nn, 18]
        :return: [B, nn]
        """
        query_embedded = query_embedded.transpose(-1, -2).contiguous()  # [B, 18, nn]

        feat = self.feat_mlp(feat)  # [B, 256]
        query_embedded = self.query_mlp(query_embedded)  # [B, 256, nn]

        # ([B, 256] -> [B, 256, 1]) + [B, 256, nn]
        x = feat.unsqueeze(-1) + query_embedded  # [B, 256, nn]
        x = nn.functional.relu(x)

        x = self.mlps(x)  # [B, 1, nn]
        x = x.squeeze(1)  # [B, nn]

        #assert x.shape[1] == 64 or x.shape[1] == 4096, f"Incorrect ImplicitModel dim: x.shape: {x.shape}"
        return x
