"""
Last modified date: 2022.08.07
Author: mzhmxzh
Description: class HandModel
"""

import os
import json
import torch
from torch import nn
import pytorch_kinematics as pk
import trimesh
import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.transforms
from csdf import index_vertices_by_faces, compute_sdf

class AdditionalLoss(nn.Module):
    def __init__(self, tta_cfg, device, num_obj_points, num_hand_points, cmap_net):
        super().__init__()
        self.num_obj_points = num_obj_points
        self.hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            n_surface_points=num_hand_points,
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device=device,
        )
        self.cmap_func = cmap_net.forward
        self.normalize_factor=tta_cfg['normalize_factor']
        self.weights = dict(
            weight_cmap=tta_cfg['weight_cmap'],
            weight_pen=tta_cfg['weight_pen'],
            weight_dis=tta_cfg['weight_dis'],
            weight_spen=tta_cfg['weight_spen'],
            weight_tpen=tta_cfg['weight_tpen'],
            thres_dis=tta_cfg['thres_dis'],
        )
    
    def forward(self, points, plane_parameters, translation, hand_qpos):
        hand_pose = torch.cat([translation,torch.zeros_like(translation), hand_qpos],dim=-1)
        hand = self.hand_model(hand_pose, points, with_penetration=True, with_surface_points=True, with_contact_candidates=True, with_penetration_keypoints=True)
        discretized_cmap_pred = self.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))['contact_map'].detach().exp()# [B, N, 10]
        arange = (torch.arange(0, discretized_cmap_pred.shape[-1], dtype=discretized_cmap_pred.dtype, device=discretized_cmap_pred.device)+0.5)
        cmap_pred = torch.mean(discretized_cmap_pred * arange, dim=-1).detach()
        cmap = 2 - 2 * torch.sigmoid(self.normalize_factor * (hand['distances'].abs() + 1e-8).sqrt())  # calculate pseudo contactmap: 0~3cm mapped into value 1~0
        loss, losses = cal_loss(hand, cmap, cmap_pred, points, plane_parameters, self.num_obj_points, **self.weights, verbose=True)
        return loss, losses
    
    def tta_loss(self, hand_pose, points, cmap_pred, plane_parameters):
        hand = self.hand_model(hand_pose, points, with_penetration=True, with_surface_points=True, with_contact_candidates=True, with_penetration_keypoints=True)
        cmap = 2 - 2 * torch.sigmoid(self.normalize_factor * (hand['distances'].abs() + 1e-8).sqrt())
        loss = cal_loss(hand, cmap, cmap_pred, points, plane_parameters, self.num_obj_points, **self.weights)
        return loss

def cal_loss(hand, cmap_labels, cmap_pred, object_pc, plane_parameters, num_obj_points, verbose=False, weight_cmap=1., weight_pen=1., weight_dis=1., weight_spen=1., weight_tpen=1., thres_dis=0.02):

    batch_size = len(cmap_labels)
    
    distances = hand['penetration']  # signed squared distances from object_pc to hand, inside positive, outside negative
    contact_candidates = hand['contact_candidates']
    penetration_keypoints = hand['penetration_keypoints']
    # plane_distances = hand['plane_distances']

    # loss_cmap
    loss_cmap = torch.nn.functional.mse_loss(cmap_pred[:, :num_obj_points], cmap_labels[:, :num_obj_points], reduction='sum') / batch_size

    # loss_pen
    loss_pen = distances[distances > 0].sum() / batch_size

    # loss_dis
    dis_pred = pytorch3d.ops.knn_points(object_pc, contact_candidates).dists[:, :, 0]  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2
    loss_dis = dis_pred[small_dis_pred].sqrt().sum()# / (small_dis_pred.sum() + 1e-4)

    # loss_spen
    dis_spen = (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    loss_spen = dis_spen.sum() / batch_size

    # loss_tpen
    # plane_distances[plane_distances > 0] = 0
    # loss_tpen = - weight_tpen * plane_distances.sum() / batch_size
    dis_tpen = (penetration_keypoints * plane_parameters[:, :3].unsqueeze(1)).sum(2) + plane_parameters[:, 3].unsqueeze(1) - 0.01
    dis_tpen[dis_tpen > 0] = 0
    loss_tpen = - dis_tpen.sum() / batch_size


    # loss
    loss = weight_cmap * loss_cmap + weight_pen * loss_pen + weight_dis * loss_dis + weight_spen * loss_spen + weight_tpen * loss_tpen
    if verbose:
        return loss, dict(loss=loss, loss_cmap=loss_cmap, loss_pen=loss_pen, loss_dis=loss_dis, loss_spen=loss_spen, loss_tpen=loss_tpen)
    else:
        return loss
class HandModel:
    def __init__(self, mjcf_path, mesh_path, n_surface_points=3000, contact_points_path=None, penetration_points_path=None, device='cpu'):
        self.device = device
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None
        contact_points = json.load(open(contact_points_path, 'r')) if contact_points_path is not None else None
        
        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, 'box.obj'), process=False)
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if contact_points is not None else None
                penetration_keypoints = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None
                link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                self.mesh[link_name] = dict(
                    vertices=link_vertices,
                    faces=link_faces,
                    contact_candidates=contact_candidates,
                    penetration_keypoints=penetration_keypoints,
                    face_verts=link_face_verts
                )
                if link_name not in ['robot0:palm', 'robot0:lfmetacarpal_child']:
                    self.mesh[link_name]['geom_param'] = body.link.visuals[0].geom_param
                areas[link_name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)

        build_mesh_recurse(self.chain._root)
                
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples['robot0:palm'] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=torch.float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points
    
    def __call__(self, hand_pose, object_pc=None, plane_parameters=None, with_meshes=False, with_surface_points=False, with_contact_candidates=False, with_penetration_keypoints=False, with_penetration=False):
        batch_size = len(hand_pose)
        global_translation = hand_pose[:, 0:3]
        global_rotation = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
        current_status = self.chain.forward_kinematics(hand_pose[:, 6:])

        hand = {}

        if object_pc is not None:
            distances = []
            penetration = []
            x = (object_pc - global_translation.unsqueeze(1)) @ global_rotation  # (batch_size, num_samples, 3)
            for link_name in self.mesh:
                if link_name in ['robot0:ffknuckle_child', 'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 'robot0:thbase_child', 'robot0:thhub_child']:
                    continue
                matrix = current_status[link_name].get_matrix()
                x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
                x_local = x_local.reshape(-1, 3)  # (batch_size * num_samples, 3)
                if 'geom_param' not in self.mesh[link_name]:
                    face_verts = self.mesh[link_name]['face_verts']
                    dis_local, _, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                    dis_local = dis_local * (-dis_signs)
                    if with_penetration:
                        penetration_local = dis_local
                else:
                    height = self.mesh[link_name]['geom_param'][1] * 2
                    radius = self.mesh[link_name]['geom_param'][0]
                    projected_point = x_local.detach().clone()
                    projected_point[:, :2] = 0
                    projected_point[:, 2] = torch.clamp(projected_point[:, 2], 0, height)
                    direction = torch.nn.functional.normalize(x_local.detach() - projected_point)
                    direction = torch.where(direction.norm(dim=1, keepdim=True) < 0.9, torch.tensor([1, 0, 0], dtype=torch.float, device=self.device), direction)
                    nearest_point = projected_point + radius * direction
                    dis_local = (x_local - nearest_point).square().sum(dim=1)  # (batch_size * num_samples)
                    mask = (x_local.detach() - projected_point).norm(dim=1) < radius
                    dis_local = torch.where(mask, dis_local, -dis_local)
                    if with_penetration:
                        if link_name not in ['robot0:thmiddle_child', 'robot0:thdistal_child', 'robot0:thproximal_child']:
                            nearest_point = projected_point.clone()
                            nearest_point[:, 1] = -radius
                            penetration_local = (x_local - nearest_point).square().sum(dim=1)  # (batch_size * num_samples)
                            penetration_local = torch.where(mask, penetration_local, -penetration_local)
                        else:
                            nearest_point = projected_point.clone()
                            nearest_point[:, 0] = -radius
                            penetration_local = (x_local - nearest_point).square().sum(dim=1)  # (batch_size * num_samples)
                            penetration_local = torch.where(mask, penetration_local, -penetration_local)
                            # penetration_local = dis_local
                distances.append(dis_local.reshape(x.shape[0], x.shape[1]))  # (batch_size, num_samples)
                if with_penetration:
                    penetration.append(penetration_local.reshape(x.shape[0], x.shape[1]))
            distances = torch.max(torch.stack(distances), dim=0)[0]
            hand['distances'] = distances
            if with_penetration:
                penetration = torch.max(torch.stack(penetration), dim=0)[0]
                hand['penetration'] = penetration
        
        if plane_parameters is not None:
            # plane_parameters: (total_batch_size, 4)
            dis = []
            for link_name in self.mesh:
                if link_name in ['robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child', 'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 'robot0:thbase_child', 'robot0:thhub_child']:
                    continue
                if 'geom_param' not in self.mesh[link_name]:
                    verts = current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
                    if len(verts.shape) == 2:
                        verts = verts.unsqueeze(0).repeat(plane_parameters.shape[0], 1, 1)
                    verts = torch.bmm(verts, global_rotation.transpose(1, 2)) + global_translation.unsqueeze(1)
                    dis_tmp = (plane_parameters[:, :3].unsqueeze(1) * verts).sum(2) + plane_parameters[:, 3].unsqueeze(1)
                    dis_tmp = dis_tmp.min(1)[0]
                else:
                    height = self.mesh[link_name]['geom_param'][1] * 2
                    radius = self.mesh[link_name]['geom_param'][0]
                    verts = torch.tensor([[0, 0, 0], [0, 0, height]], dtype=torch.float32, device=self.device)
                    verts = verts.unsqueeze(0).repeat(plane_parameters.shape[0], 1, 1)
                    matrix = current_status[link_name].get_matrix()
                    verts = torch.bmm(verts, matrix[:, :3, :3].transpose(1, 2)) + matrix[:, :3, 3].unsqueeze(1)
                    verts = torch.bmm(verts, global_rotation.transpose(1, 2)) + global_translation.unsqueeze(1)
                    dis_tmp = (plane_parameters[:, :3].unsqueeze(1) * verts).sum(2) + plane_parameters[:, 3].unsqueeze(1)
                    dis_tmp -= radius
                    dis_tmp = dis_tmp.min(1)[0]
                dis.append(dis_tmp)
            dis = torch.stack(dis, dim=1)  # [B, n_links]
            hand['plane_distances'] = dis

        def get_points(key):
            points = [current_status[link_name].transform_points(self.mesh[link_name][key]).expand(batch_size, -1, -1) for link_name in self.mesh]
            points = torch.concat(points, dim=1) @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(1)
            return points

        if with_meshes:
            hand['vertices'] = get_points('vertices')
            n_vertices = 0
            faces = []
            for link_name in self.mesh:
                faces.append(self.mesh[link_name]['faces'] + n_vertices)
                n_vertices += self.mesh[link_name]['vertices'].shape[0]
            hand['faces'] = torch.concat(faces)
        
        if with_surface_points:
            hand['surface_points'] = get_points('surface_points')
        
        if with_contact_candidates:
            hand['contact_candidates'] = get_points('contact_candidates')

        if with_penetration_keypoints:
            hand['penetration_keypoints'] = get_points('penetration_keypoints')

        return hand

def add_rotation_to_hand_pose(hand_pose, rotation):
    translation = hand_pose[..., :3]
    added_rot_aa = hand_pose[..., 3:6]
    added_rot_mat = pytorch3d.transforms.axis_angle_to_matrix(added_rot_aa)
    hand_qpos = hand_pose[..., 6:]

    new_translation = torch.einsum('na,nab->nb', translation, rotation)
    new_rotation_mat = torch.einsum('nba,nbc->nac', rotation, added_rot_mat)
    new_rotation_aa = pytorch3d.transforms.matrix_to_axis_angle(new_rotation_mat)

    return torch.cat([new_translation, new_rotation_aa, hand_qpos], dim=-1)