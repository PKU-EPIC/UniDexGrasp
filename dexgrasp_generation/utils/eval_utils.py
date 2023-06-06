"""
Last modified date: 2022.09.09
Author: mzhmxzh
Description: cal q1 for datasetv3f
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
import argparse
import numpy as np
import scipy.spatial
import trimesh as tm
import transforms3d
import torch
import pytorch3d.structures
import pytorch3d.ops
import csdf
from csdf import index_vertices_by_faces, compute_sdf
from tqdm import tqdm
from utils.hand_model import HandModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]


class KaolinModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_verts_list = None
        self.object_faces_list = None
        self.object_face_verts_list = None
        self.scale_choice = torch.tensor(
            [0.06, 0.08, 0.1, 0.12, 0.15])
        # self.model = None

    def initialize(self, object_code_list):
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_verts_list = []
        self.object_faces_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        self.object_scale_list = []
        model_params_list = []
        for object_code in object_code_list:
            if object_code[:5] == 'ddgo/':
                object_code = object_code[5:]
            # info = json.load(
            #     open(
            #         os.path.join(self.data_root_path, object_code,
            #                      "info.json")))
            self.object_scale_tensor.append(
                1 / (self.scale_choice[torch.randint(
                    0, self.scale_choice.shape[0],
                    (self.batch_size_each, ))].to(self.device)))
            # self.object_scale_tensor.append(
            #     1 / (1 / float(info["scale"]) * (torch.rand(self.batch_size_each, dtype=torch.float, device=self.device) * 0.4 + 0.8)))
            self.object_mesh_list.append(
                tm.load(
                    os.path.join(self.data_root_path, object_code, 'coacd', 'decomposed.obj'), force="mesh", process=False))
            self.object_verts_list.append(
                torch.Tensor(self.object_mesh_list[-1].vertices).to(
                    self.device, torch.float32))
            self.object_faces_list.append(
                torch.Tensor(self.object_mesh_list[-1].faces).long().to(
                    self.device, torch.long))
            self.object_face_verts_list.append(
                index_vertices_by_faces(
                    self.object_verts_list[-1],
                    self.object_faces_list[-1]))
            vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float32, device=self.device)
            faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.long, device=self.device)
            mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
            surface_points.to(dtype=float, device=self.device)
            self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x * scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, normal, dis_signs, _, _ = csdf.compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis+1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance / scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) / scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals


def cal_q1(cfg, hand_model, object_model, object_code, scale, hand_pose, device):
    # load data
    object_model.initialize([object_code])
    object_model.object_scale_tensor = torch.tensor(1 / scale, dtype=torch.float, device=device).reshape(1, 1)
    object_model.batch_size_each = 1
    # cal hand
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])
    # cal contact points and contact normals
    contact_points_object = []
    contact_normals = []
    for link_name in hand_model.mesh:
        if len(hand_model.mesh[link_name]['surface_points']) == 0:
            continue
        surface_points = current_status[link_name].transform_points(hand_model.mesh[link_name]['surface_points'])
        surface_points = surface_points @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(1)
        distances, normals, closest_points = object_model.cal_distance(surface_points, with_closest_points=True)
        if cfg['nms']:
            nearest_point_index = distances.argmax()
            if -distances[0, nearest_point_index] < cfg['thres_contact']:
                contact_points_object.append(closest_points[0, nearest_point_index])
                contact_normals.append(normals[0, nearest_point_index])
        else:
            contact_idx = (-distances < cfg['thres_contact']).nonzero().reshape(-1)
            if len(contact_idx) != 0:
                for idx in contact_idx:
                    contact_points_object.append(closest_points[0, idx])
                    contact_normals.append(normals[0, idx])
    if len(contact_points_object) == 0:
        contact_points_object.append(torch.tensor([0, 0, 0], dtype=torch.float, device=device))
        contact_normals.append(torch.tensor([1, 0, 0], dtype=torch.float, device=device))
    
    contact_points_object = torch.stack(contact_points_object).cpu().numpy()
    contact_normals = torch.stack(contact_normals).cpu().numpy()

    n_contact = len(contact_points_object)

    if np.isnan(contact_points_object).any() or np.isnan(contact_normals).any():
        return 0

    # cal contact forces
    u1 = np.stack([-contact_normals[:, 1], contact_normals[:, 0], np.zeros([n_contact], dtype=np.float32)], axis=1)
    u2 = np.stack([np.ones([n_contact], dtype=np.float32), np.zeros([n_contact], dtype=np.float32), np.zeros([n_contact], dtype=np.float32)], axis=1)
    u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(u, contact_normals)
    theta = np.linspace(0, 2 * np.pi, cfg['m'], endpoint=False).reshape(cfg['m'], 1, 1)
    contact_forces = (contact_normals + cfg['mu'] * (np.cos(theta) * u + np.sin(theta) * v)).reshape(-1, 3)

    # cal wrench space and q1
    origin = np.array([0, 0, 0], dtype=np.float32)
    wrenches = np.concatenate([np.concatenate([contact_forces, cfg['lambda_torque'] * np.cross(np.tile(contact_points_object - origin, (cfg['m'], 1)), contact_forces)], axis=1), np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32)], axis=0)
    try:
        wrench_space = scipy.spatial.ConvexHull(wrenches)
    except scipy.spatial._qhull.QhullError:
        return 0
    q1 = np.array([1], dtype=np.float32)
    for equation in wrench_space.equations:
        q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))

    return q1.item()


def cal_pen(hand_model, object_model, object_code, scale, hand_pose, device):
    # load data
    object_model.initialize([object_code])
    object_model.object_scale_tensor = torch.tensor(1 / scale, dtype=torch.float, device=device).reshape(1, 1)
    object_model.batch_size_each = 1
    # cal pen
    object_surface_points = object_model.surface_points_tensor * scale
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])
    distances = []
    x = (object_surface_points - global_translation.unsqueeze(1)) @ global_rotation
    for link_name in hand_model.mesh:
        if link_name in ['robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child', 'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 'robot0:thbase_child', 'robot0:thhub_child']:
            continue
        matrix = current_status[link_name].get_matrix()
        x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
        x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
        if 'geom_param' not in hand_model.mesh[link_name]:
            face_verts = hand_model.mesh[link_name]['face_verts']
            dis_local, _, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)
        else:
            height = hand_model.mesh[link_name]['geom_param'][1] * 2
            radius = hand_model.mesh[link_name]['geom_param'][0]
            nearest_point = x_local.detach().clone()
            nearest_point[:, :2] = 0
            nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            dis_local = radius - (x_local - nearest_point).norm(dim=1)
        distances.append(dis_local.reshape(x.shape[0], x.shape[1]))
    distances = torch.max(torch.stack(distances, dim=0), dim=0)[0]

    return max(distances.max().item(), 0)


def cal_tpen(hand_model, hand_pose, plane_parameters, device):
    # plane_parameters: (total_batch_size, 4)
    dis = []
    hand_pose = hand_pose.unsqueeze(0)
    plane_parameters = torch.tensor(plane_parameters, dtype=torch.float, device=device).unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])
    for link_name in hand_model.mesh:
        if link_name in ['robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child', 'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 'robot0:thbase_child', 'robot0:thhub_child']:
            continue
        if 'geom_param' not in hand_model.mesh[link_name]:
            verts = current_status[link_name].transform_points(hand_model.mesh[link_name]['vertices'])
            if len(verts.shape) == 2:
                verts = verts.unsqueeze(0).repeat(plane_parameters.shape[0], 1, 1)
            verts = torch.bmm(verts, global_rotation.transpose(1, 2)) + global_translation.unsqueeze(1)
            dis_tmp = (plane_parameters[:, :3].unsqueeze(1) * verts).sum(2) + plane_parameters[:, 3].unsqueeze(1)
            dis_tmp = dis_tmp.min(1)[0]
        else:
            height = hand_model.mesh[link_name]['geom_param'][1] * 2
            radius = hand_model.mesh[link_name]['geom_param'][0]
            verts = torch.tensor([[0, 0, 0], [0, 0, height]], dtype=torch.float32, device=device)
            verts = verts.unsqueeze(0).repeat(plane_parameters.shape[0], 1, 1)
            matrix = current_status[link_name].get_matrix()
            verts = torch.bmm(verts, matrix[:, :3, :3].transpose(1, 2)) + matrix[:, :3, 3].unsqueeze(1)
            verts = torch.bmm(verts, global_rotation.transpose(1, 2)) + global_translation.unsqueeze(1)
            dis_tmp = (plane_parameters[:, :3].unsqueeze(1) * verts).sum(2) + plane_parameters[:, 3].unsqueeze(1)
            dis_tmp -= radius
            dis_tmp = dis_tmp.min(1)[0]
        dis.append(dis_tmp)
    dis = torch.stack(dis, dim=1)  # [B, n_links]
    dis[dis > 0] = 0

    return max(-dis.min().item(), 0)


def eval_result(q1_cfg, result, hand_model, object_model, device):
    result = {k: v.cuda() if type(v) == torch.Tensor else v for k, v in result.items()}
    result_dict = {}
    for has_tta in ['', 'tta_']:
        result_dict[has_tta+'q1'] = cal_q1(q1_cfg, hand_model, object_model, result['object_code'], result['scale'], result[has_tta+'hand_pose'], device)
        result_dict[has_tta+'pen'] = cal_pen(hand_model, object_model, result['object_code'], result['scale'], result[has_tta+'hand_pose'], device)
        result_dict[has_tta+'tpen'] = cal_tpen(hand_model, result[has_tta+'hand_pose'], result['canon_plane'], device)
        valid = (result_dict[has_tta+'pen'] < q1_cfg['thres_pen']) and (result_dict[has_tta+'tpen'] < q1_cfg['thres_tpen'])
        result_dict[has_tta+'valid_q1'] = result_dict[has_tta+'q1'] if valid else 0
    return {k: [v] for k, v in result_dict.items()}