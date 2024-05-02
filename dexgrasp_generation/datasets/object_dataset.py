"""
Last modified date: 2022.08.07
Author: mzhmxzh
Description: dataset
"""

import os
import json
import numpy as np
import torch
import trimesh as tm
import transforms3d
from torch.utils.data import Dataset
import pytorch3d.ops


class Meshdata(Dataset):
    def __init__(self, cfg, mode):
        self.mode = mode
        self.data_root_path = os.path.join(cfg['dataset']['root_path'], 'DFCData', 'meshes')
        self.splits_path = os.path.join(cfg['dataset']['root_path'], 'DFCData', 'splits')

        self.object_code_list = []
        for splits_file_name in os.listdir(self.splits_path):
            with open(os.path.join(self.splits_path, splits_file_name), 'r') as f:
                splits_map = json.load(f)
            self.object_code_list += [os.path.join(splits_file_name[:-5], object_code) for object_code in splits_map[mode]]
        
        self.object_list = []
        for object_code in self.object_code_list:
            pose_matrices = np.load(os.path.join(self.data_root_path, object_code, 'poses.npy'))
            pcs_table = np.load(os.path.join(self.data_root_path, object_code, 'pcs_table.npy'))
            for scale in [0.06, 0.08, 0.1, 0.12, 0.15]:
                indices = np.random.permutation(len(pose_matrices))[:cfg['n_samples']]
                for index in indices:
                    pose_matrix = pose_matrices[index]
                    pose_matrix[:2, 3] = 0
                    self.object_list.append((object_code, pcs_table[index], scale, pose_matrix))
    
    
    def __len__(self):
        return len(self.object_list)
    
    def __getitem__(self, idx):
        object_code, pcs_table, scale, pose_matrix = self.object_list[idx]
        object_pc = torch.from_numpy(scale * (pcs_table @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]))
        only_object_pc = object_pc[:3000]
        table_pc = object_pc[3000:]
        max_diameter = 0.2
        n_samples_table_extra = 2000
        min_diameter = (only_object_pc[:, 0] ** 2 + only_object_pc[:, 1] ** 2).max()
        distances = min_diameter + (max_diameter - min_diameter) * torch.rand(n_samples_table_extra, dtype=torch.float) ** 0.5
        theta = 2 * np.pi * torch.rand(n_samples_table_extra, dtype=torch.float)
        table_pc_extra = torch.stack([distances * torch.cos(theta), distances * torch.sin(theta), torch.zeros_like(distances)], dim=1)
        table_pc = torch.cat([table_pc, table_pc_extra])
        table_pc_cropped = table_pc[table_pc[:, 0] ** 2 + table_pc[:, 1] ** 2 < max_diameter]
        table_pc_cropped_sampled = pytorch3d.ops.sample_farthest_points(table_pc_cropped.unsqueeze(0), K=1000)[0][0]
        object_pc = torch.cat([only_object_pc, table_pc_cropped_sampled])
        plane = torch.zeros_like(torch.from_numpy(pose_matrix[2]))
        plane[2] = 1
        #plane = pose_matrix[2].copy()
        #plane[3] *= scale
        ret_dict = {
            "object_code": object_code, 
            "obj_pc": object_pc,
            "plane": plane,
            "scale": scale,
        }
        return ret_dict

