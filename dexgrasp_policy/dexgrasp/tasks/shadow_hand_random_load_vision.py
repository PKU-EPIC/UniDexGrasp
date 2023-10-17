# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import random
import transforms3d

from pyparsing import And
import torch
import cv2
import os.path as osp

from utils.torch_jit_utils import *
from utils.data_info import plane2euler
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi



class ShadowHandRandomLoadVision(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.has_sensor = True
        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips)  ##
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.cfg["env"]["numStates"] = 0
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 24
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # Vision
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.segmentation_id = {
            'hand': 2,
            'object': 3,
            'goal': 4,
            'table': 1,
        }
        self.num_state_obs = self.cfg['env']['numObservations']
        self.camera_depth_tensor_list = []
        self.camera_rgb_tensor_list = []
        self.camera_seg_tensor_list = []
        self.camera_vinv_mat_list = []
        self.camera_proj_mat_list = []
        self.camera_handles = []
        self.num_cameras = len(self.cfg['env']['vision']['camera']['eye'])
        self._cfg_camera_props()
        self._cfg_camera_pose()

        camera_u = torch.arange(0, self.camera_props.width)
        camera_v = torch.arange(0, self.camera_props.height)
        self.camera_v2, self.camera_u2 = torch.meshgrid(
            camera_v, camera_u, indexing='ij')

        self.num_envs = self.cfg['env']['numEnvs']
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float)

        self.x_n_bar = self.cfg['env']['vision']['bar']['x_n']
        self.x_p_bar = self.cfg['env']['vision']['bar']['x_p']
        self.y_n_bar = self.cfg['env']['vision']['bar']['y_n']
        self.y_p_bar = self.cfg['env']['vision']['bar']['y_p']
        self.z_n_bar = self.cfg['env']['vision']['bar']['z_n']
        self.z_p_bar = self.cfg['env']['vision']['bar']['z_p']
        self.depth_bar = self.cfg['env']['vision']['bar']['depth']
        self.num_pc_downsample = self.cfg['env']['vision']['pointclouds']['numDownsample']
        self.num_pc_presample = self.cfg['env']['vision']['pointclouds']['numPresample']
        self.num_each_pt = self.cfg['env']['vision']['pointclouds']['numEachPoint']
        self.num_pc_flatten = self.num_pc_downsample * self.num_each_pt
        self.cfg['env']['numObservations'] += self.num_pc_flatten
        self.cfg['env']['numObservations'] += 2*self.num_pc_downsample

        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        # Vision
        self.camera_u2 = to_torch(self.camera_u2, device=self.device)
        self.camera_v2 = to_torch(self.camera_v2, device=self.device)
        self.env_origin = to_torch(self.env_origin, device=self.device)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if self.has_sensor:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 1 + self.num_object_dofs * 1)  ##
            self.dof_force_tensor = self.dof_force_tensor[:, :24]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0
        self.step_counter = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = list(object_scale_dict.keys())
        all_scales = set()
        for object_scales in object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales):
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)

        self.repose_z = self.cfg['env']['repose_z']

        self.grasp_data = {}
        assets_path = '../assets'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        for object_code in self.object_code_list:
            data_per_object = {}
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            for num in data_num_list:
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                qpos = data_dict['qpos'].item() #goal
                scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
                scale = round(1 / scale_inverse, 2)
                assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.tensor(list(qpos.values())[:22], dtype=torch.float, device=self.device)
                target_hand_rot_xyz = torch.tensor(list(qpos.values())[22:25], dtype=torch.float, device=self.device)  # 3
                target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                translation, euler = plane2euler(plane, axes='sxyz')  # object
                object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)

                if object_init_z > 0.05:
                    continue

                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                    data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                    data_per_object[scale]['object_init_z'].append(object_init_z)
                else:
                    data_per_object[scale] = {}
                    data_per_object[scale]['target_qpos'] = [target_qpos]
                    data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                    data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                    data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                    data_per_object[scale]['object_init_z'] = [object_init_z]
            self.grasp_data[object_code] = data_per_object

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg['env']['random_time']
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping

        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        object_asset_list = []
        goal_asset_list = []

        # RandomLoad
        num_object_bodies_list = []
        num_object_shapes_list = []

        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }

        assets_path = '../assets'
        visual_feat_root = osp.join(assets_path, 'meshdatav3_pc_feat')
        object_scale_idx_pairs = []
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64))

        for object_id in range(len(self.object_code_list)):
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    object_scale_idx_pairs.append([object_id, scale_id])
                else:
                    print(f'prior not found: {object_code}/{scale}')
                file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                with open(file_dir, 'rb') as f:
                    feat = np.load(f)
                self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)


        object_asset_dict = {}
        goal_asset_dict = {}


        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None

            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    scaled_object_asset = self.gym.load_asset(self.sim, asset_root+"/meshdatav3_scaled", scaled_object_asset_file, object_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset
                    if object_asset is None:
                        object_asset = scaled_object_asset
                                    
            assert object_asset is not None
            object_asset_options.disable_gravity = True
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

            num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)

            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)


        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z,gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        object_start_pose = gymapi.Transform()
        self.object_rise = 0.1
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + self.object_rise)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.3 - self.object_rise)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.p.z -= 0.0
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.table_indices = []

        # RandomLoad
        self.num_obj_per_env = self.cfg['env']['random_load']['num_obj_per_env']
        self.num_actors_per_env = 2 + self.num_obj_per_env * 2
        self.init_object_waiting_pose = []
        self.init_goal_waiting_pose = []
        self._init_waiting_pose()
        self.all_object_indices = []
        self.all_goal_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        body_names = {
            'wrist': 'robot0:wrist',
            'palm': 'robot0:palm',
            'thumb': 'robot0:thdistal',
            'index': 'robot0:ffdistal',
            'middle': 'robot0:mfdistal',
            'ring': 'robot0:rfdistal',
            'little': 'robot0:lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)
        
        # create fingertip force sensors, if needed
        if self.has_sensor:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        self.object_id_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_scale_buf = {}

        # RandomLoad
        self.object_scale_idx_pairs = to_torch(object_scale_idx_pairs, dtype=torch.int32, device=self.device)

        def sample_idx(total, num_envs, num_per_env):
            num_total = num_envs * num_per_env
            div = num_total // total
            res = num_total % total
            assert div >= 1
            idx = list(torch.arange(total)) * div + list(torch.arange(res))
            idx = torch.tensor(idx).view(num_envs, num_per_env)
            return idx

        self.idx_tensor = sample_idx(self.object_scale_idx_pairs.shape[0], self.num_envs, self.num_obj_per_env)

        self.obj_actors = []
        for i in range(self.num_envs):
            self.obj_actors.append([])

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # RandomLoad
            idx_this_env = self.idx_tensor[i]
            object_idx_this_env = [idx.item() for idx in self.object_scale_idx_pairs[idx_this_env][:, 0]]
            object_scale_idx_this_env = [idx.item() for idx in self.object_scale_idx_pairs[idx_this_env][:, 1]]

            object_asset_this_env = [
                object_asset_dict[obj_idx][scale_idx]
                for obj_idx, scale_idx in zip(object_idx_this_env, object_scale_idx_this_env)
            ]

            goal_asset_this_env = [goal_asset] * self.num_obj_per_env

            if self.aggregate_mode >= 1:
                num_object_bodies = 0
                num_object_shapes = 0
                for obj_idx in object_idx_this_env:
                    num_object_bodies += num_object_bodies_list[obj_idx]
                    num_object_shapes += num_object_shapes_list[obj_idx]
                max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * num_object_bodies + 1
                max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * num_object_shapes + 1
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, shadow_hand_asset, shadow_hand_dof_props, shadow_hand_start_pose)

            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            for j in range(self.num_obj_per_env):
                if j == 0:
                    object_pose = object_start_pose
                else:
                    object_pose = self.init_object_waiting_pose[j - 1]
                object_actor = self._load_object(env_ptr, i, object_asset_this_env[j], object_pose, 1.0)
                # set friction
                object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
                object_shape_props[0].friction = 1
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)

                # set color
                object_color = self.cfg['env']['vision']['color']['object']
                self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
                self.obj_actors[i].append(object_actor)

            for j in range(self.num_obj_per_env):
                if j == 0:
                    goal_pose = goal_start_pose
                else:
                    goal_pose = self.init_goal_waiting_pose[j - 1]
                goal_actor = self._load_goal(env_ptr, i, goal_asset_this_env[j], goal_pose, 1.0)
                # set color
                goal_color = self.cfg['env']['vision']['color']['goal']
                self.gym.set_rigid_body_color(env_ptr, goal_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*goal_color))

            # add object
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # Vision
            self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs,13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        # RandomLoad
        self.all_object_indices = to_torch(self.all_object_indices, device=self.device, dtype=torch.long).view(self.num_envs, self.num_obj_per_env)
        self.all_goal_indices = to_torch(self.all_goal_indices, device=self.device, dtype=torch.long).view(self.num_envs, self.num_obj_per_env)
        self.object_indices = self.all_object_indices[:, 0].clone()
        self.goal_object_indices = self.all_goal_indices[:, 0].clone()
        self.base_object_indices = self.object_indices.clone()
        self.base_goal_indices = self.goal_object_indices.clone()
        self.active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if not self.sequential_load:
            self.object_waiting_tensor = to_torch(self.object_waiting_tensor, device=self.device)
            self.goal_waiting_tensor = to_torch(self.goal_waiting_tensor, device=self.device)

    def _init_waiting_pose(self):
        self.sequential_load = self.cfg['env']['random_load']['sequential']
        a = self.cfg['env']['random_load']['a']
        b = self.cfg['env']['random_load']['b']
        assert self.num_obj_per_env <= a * b * 4 + 1
        width = self.cfg['env']['random_load']['obj_width']
        half_width = width * 0.5
        half_table = max(self.table_dims.x, self.table_dims.y) * 0.5
        env_spacing = self.cfg["env"]['envSpacing']
        effective_env_spacing_row = half_table + width * a * 2
        effective_env_spacing_col = width * b - half_table
        assert env_spacing > max(effective_env_spacing_row, effective_env_spacing_col)
        waiting_pose = np.zeros((a * b * 4, 6))
        row = half_table + half_width + np.arange(a * 2) * width
        pos_slice = -effective_env_spacing_col + half_width + np.arange(b) * width
        neg_slice = effective_env_spacing_col - half_width - np.arange(b) * width
        waiting_pose[:, 2] = self.object_rise
        waiting_pose[:, 5] = self.object_rise

        for i in range(a):
            j = i
            waiting_pose[b * j:b * (j + 1), 0] = row[i * 2]
            waiting_pose[b * j:b * (j + 1), 1] = pos_slice
            waiting_pose[b * j:b * (j + 1), 3] = row[i * 2 + 1]
            waiting_pose[b * j:b * (j + 1), 4] = pos_slice
            j += a
            waiting_pose[b * j:b * (j + 1), 1] = row[i * 2]
            waiting_pose[b * j:b * (j + 1), 0] = neg_slice
            waiting_pose[b * j:b * (j + 1), 4] = row[i * 2 + 1]
            waiting_pose[b * j:b * (j + 1), 3] = neg_slice
            j += a
            waiting_pose[b * j:b * (j + 1), 0] = -row[i * 2]
            waiting_pose[b * j:b * (j + 1), 1] = neg_slice
            waiting_pose[b * j:b * (j + 1), 3] = -row[i * 2 + 1]
            waiting_pose[b * j:b * (j + 1), 4] = neg_slice
            j += a
            waiting_pose[b * j:b * (j + 1), 1] = -row[i * 2]
            waiting_pose[b * j:b * (j + 1), 0] = pos_slice
            waiting_pose[b * j:b * (j + 1), 4] = -row[i * 2 + 1]
            waiting_pose[b * j:b * (j + 1), 3] = pos_slice

        for i in range(a * b * 4):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*list(waiting_pose[i, 0:3]))
            self.init_object_waiting_pose.append(pose)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*list(waiting_pose[i, 3:6]))
            self.init_goal_waiting_pose.append(pose)

        if not self.sequential_load:
            self.object_waiting_tensor = torch.zeros((a * b * 4, 13))
            self.object_waiting_tensor[:, 0:3] = to_torch(waiting_pose[:, 0:3])
            self.object_waiting_tensor[:, 6] = 1
            self.goal_waiting_tensor = torch.zeros((a * b * 4, 13))
            self.goal_waiting_tensor[:, 0:3] = to_torch(waiting_pose[:, 3:6])
            self.goal_waiting_tensor[:, 6] = 1

        return

    def _load_shadow_hand(self, env_ptr, env_id, shadow_hand_asset, hand_dof_props, init_hand_actor_pose):

        hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, init_hand_actor_pose, "hand", env_id, -1, self.segmentation_id['hand'])

        if self.has_sensor:
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

        self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)

        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)

        hand_rigid_body_index = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ]

        agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
        hand_color = self.cfg['env']['vision']['color']['hand']
        for n in agent_index[0]:
            for m in n:
                for o in hand_rigid_body_index[m]:
                    self.gym.set_rigid_body_color(env_ptr, hand_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color))

        return hand_actor

    def _load_object(self, env_ptr, env_id, object_asset, init_object_pose, scale=1.0):

        object_actor = self.gym.create_actor(env_ptr, object_asset, init_object_pose, "object", env_id, 0, self.segmentation_id['object'])
        object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
        self.all_object_indices.append(object_idx)
        self.gym.set_actor_scale(env_ptr, object_actor, scale)

        return object_actor

    def _load_goal(self, env_ptr, env_id, goal_asset, init_goal_pose, scale=1.0):

        goal_actor = self.gym.create_actor(env_ptr, goal_asset, init_goal_pose, "goal_object", env_id + self.num_envs, 0, self.segmentation_id['goal'])
        goal_idx = self.gym.get_actor_index(env_ptr, goal_actor, gymapi.DOMAIN_SIM)
        self.all_goal_indices.append(goal_idx)
        self.gym.set_actor_scale(env_ptr, goal_actor, scale)

        return goal_actor

    def _cfg_camera_props(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True
        return

    def _cfg_camera_pose(self):
        self.camera_eye_list = []
        self.camera_lookat_list = []
        camera_eye_list = self.cfg['env']['vision']['camera']['eye']
        camera_lookat_list = self.cfg['env']['vision']['camera']['lookat']
        table_centor = np.array([0.0, 0.0, self.table_dims.z])
        for i in range(self.num_cameras):
            camera_eye = np.array(camera_eye_list[i]) + table_centor
            camera_lookat = np.array(camera_lookat_list[i]) + table_centor
            self.camera_eye_list.append(gymapi.Vec3(*list(camera_eye)))
            self.camera_lookat_list.append(gymapi.Vec3(*list(camera_lookat)))
        return

    def _load_cameras(self, env_ptr, env_id, camera_props, camera_eye_list, camera_lookat_list):
        camera_handles = []
        depth_tensors = []
        rgb_tensors = []
        seg_tensors = []
        vinv_mats = []
        proj_mats = []

        origin = self.gym.get_env_origin(env_ptr)
        self.env_origin[env_id][0] = origin.x
        self.env_origin[env_id][1] = origin.y
        self.env_origin[env_id][2] = origin.z

        for i in range(self.num_cameras):
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            camera_eye = camera_eye_list[i]
            camera_lookat = camera_lookat_list[i]
            self.gym.set_camera_location(camera_handle, env_ptr, camera_eye, camera_lookat)
            raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
            depth_tensors.append(depth_tensor)

            raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
            rgb_tensors.append(rgb_tensor)

            raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
            seg_tensors.append(seg_tensor)

            vinv_mat = torch.inverse((to_torch(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle), device=self.device)))
            vinv_mats.append(vinv_mat)

            proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            proj_mats.append(proj_mat)

            camera_handles.append(camera_handle)

        self.camera_depth_tensor_list.append(depth_tensors)
        self.camera_rgb_tensor_list.append(rgb_tensors)
        self.camera_seg_tensor_list.append(seg_tensors)
        self.camera_vinv_mat_list.append(vinv_mats)
        self.camera_proj_mat_list.append(proj_mats)

        return

    def compute_reward(self, actions, id=-1):
        self.dof_pos = self.shadow_hand_dof_pos
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_id_buf, self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, self.goal_cond
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.has_sensor:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['little']
        self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # for DexGraspNet
        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos

        self.compute_full_state()

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def unpose_pc(self, pc):
        if self.repose_z:
            num_pts = pc.shape[1]
            return quat_apply(self.unpose_z_theta_quat.view(-1, 1, 4).expand(-1, num_pts, 4), pc)
        return pc

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()
        
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0:66
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 66:131: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux
        # forces and torques

        # 131:161: ft sensors: do not need repose
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:,:30]

        hand_pose_start = fingertip_obs_start + 95
        # 161:167: hand_pose
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        # 167:191: action
        aux = self.actions[:, :24]
        aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + 24] = aux

        obj_obs_start = action_obs_start + 24
        # 191:207 object_pose, goal_pos    note: only use for teacher!
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(
            self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

        # 207:236 goal
        hand_goal_start = obj_obs_start + 16
        self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] =  self.delta_target_hand_rot
        self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] =  self.delta_qpos

        # 236:300 visual feature        note: only use for teacher!
        visual_feat_start = hand_goal_start + 29
        self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf

        # Vision
        point_cloud_start = visual_feat_start + 64
        points_fps, others = self._collect_pointclouds()
        
        
        self.obs_buf[:, point_cloud_start:point_cloud_start + self.num_pc_flatten].copy_(points_fps.reshape(self.num_envs, self.num_pc_flatten))
        
        mask_hand = others["mask_hand"]
        mask_object = others["mask_object"]
        self.obs_buf[:, point_cloud_start+self.num_pc_flatten:point_cloud_start+self.num_pc_flatten+self.num_pc_downsample].copy_(mask_hand)
        self.obs_buf[:, point_cloud_start+self.num_pc_flatten+self.num_pc_downsample:point_cloud_start+self.num_pc_flatten+2*self.num_pc_downsample].copy_(mask_object)

        self.step_counter += 1

    def _collect_pointclouds(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        depth_tensor = torch.stack([torch.stack(i) for i in self.camera_depth_tensor_list])
        rgb_tensor = torch.stack([torch.stack(i) for i in self.camera_rgb_tensor_list])
        seg_tensor = torch.stack([torch.stack(i) for i in self.camera_seg_tensor_list])
        vinv_mat = torch.stack([torch.stack(i) for i in self.camera_vinv_mat_list])
        proj_matrix = torch.stack([torch.stack(i) for i in self.camera_proj_mat_list])

        point_list = []
        valid_list = []
        for i in range(self.num_cameras):
            # (num_envs, num_pts, 7) (num_envs, num_pts)
            point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], rgb_tensor[:, i], seg_tensor[:, i],
                                                                vinv_mat[:, i], proj_matrix[:, i],
                                                                self.camera_u2, self.camera_v2, self.camera_props.width,
                                                                self.camera_props.height, self.depth_bar, self.device,
                                                                # self.z_p_bar, self.z_n_bar
                                                                )
            point_list.append(point)
            valid_list.append(valid)

            # print(f'camera {i}, ', valid.sum(dim=1))

        # (num_envs, 65536 * num_cameras, 7)
        points = torch.cat(point_list, dim=1)
        depth_mask = torch.cat(valid_list, dim=1)
        points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3)
        # if self.headless:
        #     points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3) * 2
        # else:
        #     points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3)

        x_mask = (points[:, :, 0] > self.x_n_bar) * (points[:, :, 0] < self.x_p_bar)
        y_mask = (points[:, :, 1] > self.y_n_bar) * (points[:, :, 1] < self.y_p_bar)
        z_mask = (points[:, :, 2] > self.z_n_bar) * (points[:, :, 2] < self.z_p_bar)
        # (num_envs, 65536 * 3)
        valid = depth_mask * x_mask * y_mask * z_mask

        # (num_envs,)
        point_nums = valid.sum(dim=1)
        now = 0
        points_list = []
        # (num_valid_pts_total, 7)
        valid_points = points[valid]

        # presample, make num_pts equal for each env
        for env_id, point_num in enumerate(point_nums):
            if point_num == 0:
                print(f'env{env_id}_____point_num = 0_____')
                continue
            points_all = valid_points[now: now + point_num]
            random_ids = torch.randint(0, points_all.shape[0], (self.num_pc_presample,), device=self.device,
                                       dtype=torch.long)
            points_all_rnd = points_all[random_ids]
            points_list.append(points_all_rnd)
            now += point_num
        
        assert len(points_list) == self.num_envs, f'{self.num_envs - len(points_list)} envs have 0 point'
        # (num_envs, num_pc_presample)
        points_batch = torch.stack(points_list)

        num_sample_dict = self.cfg['env']['vision']['pointclouds']
        if 'numSample' not in num_sample_dict:
            points_fps = sample_points(points_batch, sample_num=self.num_pc_downsample,
                                       sample_method='furthest_batch', device=self.device)
        else:
            num_sample_dict = num_sample_dict['numSample']
            assert num_sample_dict['hand'] + num_sample_dict['object'] + num_sample_dict[
                'goal'] == self.num_pc_downsample

            zeros = torch.zeros((self.num_envs, self.num_pc_presample), device=self.device).to(torch.long)
            idx = torch.arange(self.num_envs * self.num_pc_presample, device=self.device).view(self.num_envs, self.num_pc_presample).to(torch.long)
            hand_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['hand'], idx, zeros)
            hand_pc = points_batch.view(-1, 7)[hand_idx]
            object_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['object'], idx, zeros)
            object_pc = points_batch.view(-1, 7)[object_idx]
            goal_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['goal'], idx, zeros)
            goal_pc = points_batch.view(-1, 7)[goal_idx]
            hand_fps = sample_points(hand_pc, sample_num=num_sample_dict['hand'],
                                     sample_method='furthest_batch', device=self.device)
            object_fps = sample_points(object_pc, sample_num=num_sample_dict['object'],
                                       sample_method='furthest_batch', device=self.device)
            goal_fps = sample_points(goal_pc, sample_num=num_sample_dict['goal'],
                                     sample_method='furthest_batch', device=self.device)
            points_fps = torch.cat([hand_fps, object_fps, goal_fps], dim=1)
        
        mask_hand = (points_fps[:,:,6] == self.segmentation_id["hand"])
        mask_object = (points_fps[:,:,6] == self.segmentation_id["object"])
        others = {}
        others["mask_hand"] = mask_hand
        others["mask_object"] = mask_object        
        
        if self.repose_z:
            pc_xyz = self.unpose_pc(points_fps[:, :, 0:3])
            pc_rgb = points_fps[:, :, 3:6]

            points_fps = torch.cat([pc_xyz, pc_rgb], dim=-1)
        else:
            points_fps = points_fps[:, :, 0:6]

        self.gym.end_access_image_tensors(self.sim)
        return points_fps, others

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        rand_length = torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        rand_angle = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor

        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        # self.reset_target_pose(env_ids)

        # RandomLoad
        self._switch_active(env_ids, self.step_counter == 0)
        if self.random_prior:
            # object_euler_list = []
            for env_id in env_ids:
                i = env_id.item()
                object_code = self.object_code_list[self.object_id_buf[i]]
                scale = self.object_scale_buf[i]

                data = self.grasp_data[object_code][scale]
                buf = data['object_euler_xy']
                prior_idx = random.randint(0, len(buf) - 1)
                # prior_idx = 0 ## use only one data

                self.target_qpos[i:i + 1] = data['target_qpos'][prior_idx]
                self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
                self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
                self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
                self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        dof_pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        dof_vel = self.shadow_hand_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta) 
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z
        prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)

        # RandomLoad
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self._reset_hand(
            env_ids, dof_pos, dof_vel, dof_pos, dof_pos,
            self.saved_root_tensor[hand_indices.to(torch.long), 0:3],
            prior_rot_quat,
            0,
            0
        )
        object_indices = self.object_indices[env_ids]
        goal_indices = self.goal_object_indices[env_ids]
        self._reset_object(
            env_ids,
            self.object_init_state[env_ids, 0:3],
            new_object_rot,
            0,
            0,
            self.goal_init_state[env_ids, 0:3],
            # set goal pose same to object pose, else vision input is not z-rotation invariant
            self.goal_init_state[env_ids, 3:7],  # self.goal_init_state[env_ids, 3:7], new_object_rot
            0,
            0,
        )

        table_indices = self.table_indices[env_ids]
        self._reset_table(env_ids)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        all_indices = torch.unique(torch.cat([hand_indices,
                                              self.all_object_indices[env_ids].view(-1),
                                              self.all_goal_indices[env_ids].view(-1),
                                              table_indices]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    # RandomLoad
    def _switch_active(self, env_ids, first_frame=False):
        if first_frame:
            self.active[env_ids] = 0
        elif self.sequential_load:
            self.active[env_ids] = (self.active[env_ids] + 1) % self.num_obj_per_env
        else:
            self.active[env_ids] = torch.randint(0, self.num_obj_per_env, env_ids.shape, device=self.device)

        self.object_indices[env_ids] = self.base_object_indices[env_ids] + self.active[env_ids]
        self.goal_object_indices[env_ids] = self.base_goal_indices[env_ids] + self.active[env_ids]

        for env_id in env_ids:
            i = env_id.item()
            object_id = self.object_scale_idx_pairs[self.idx_tensor[i, self.active[i]]][0].item()
            scale_id = self.object_scale_idx_pairs[self.idx_tensor[i, self.active[i]]][1].item()
            self.object_id_buf[i] = object_id
            self.object_scale_buf[i] = self.id2scale[scale_id]
            self.visual_feat_buf[i] = self.visual_feat_data[object_id][scale_id]

        return

    def _reset_hand(self, env_ids, dof_pos, dof_vel, prev_targets, cur_targets,
                    hand_positions, hand_orientations, hand_linvels, hand_angvels):

        self.shadow_hand_dof_pos[env_ids, :] = dof_pos
        self.shadow_hand_dof_vel[env_ids, :] = dof_vel

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = prev_targets
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = cur_targets

        indices = self.hand_indices[env_ids]
        self.hand_positions[indices, :] = hand_positions
        self.hand_orientations[indices, :] = hand_orientations
        self.hand_linvels[indices, :] = hand_linvels
        self.hand_angvels[indices, :] = hand_angvels

        return

    def _reset_object(self, env_ids, object_positions, object_orientations, object_linvels, object_angvels,
                      goal_positions, goal_orientations, goal_linvels, goal_angvels):
        if self.sequential_load:
            indices = self.base_object_indices[env_ids] + self.active[env_ids]
            self.hand_positions[indices, :] = object_positions
            self.hand_orientations[indices, :] = object_orientations
            self.hand_linvels[indices, :] = object_linvels
            self.hand_angvels[indices, :] = object_angvels

            indices = self.base_goal_indices[env_ids] + self.active[env_ids]
            self.hand_positions[indices, :] = goal_positions
            self.hand_orientations[indices, :] = goal_orientations
            self.hand_linvels[indices, :] = goal_linvels
            self.hand_angvels[indices, :] = goal_angvels

            for i in range(1, self.num_obj_per_env):
                indices = self.base_object_indices[env_ids] + (self.active[env_ids] + i) % self.num_obj_per_env
                self.root_state_tensor[indices, :] = self.saved_root_tensor[self.base_object_indices[env_ids] + i]
                indices = self.base_goal_indices[env_ids] + (self.active[env_ids] + i) % self.num_obj_per_env
                self.root_state_tensor[indices, :] = self.saved_root_tensor[self.base_goal_indices[env_ids] + i]
        else:
            for i in range(self.num_obj_per_env):
                indices = self.base_object_indices[env_ids] + i
                self.root_state_tensor[indices] = self.object_waiting_tensor[i]
                indices = self.base_goal_indices[env_ids] + i
                self.root_state_tensor[indices] = self.goal_waiting_tensor[i]

            indices = self.base_object_indices[env_ids] + self.active[env_ids]
            self.hand_positions[indices, :] = object_positions
            self.hand_orientations[indices, :] = object_orientations
            self.hand_linvels[indices, :] = object_linvels
            self.hand_angvels[indices, :] = object_angvels

            indices = self.base_goal_indices[env_ids] + self.active[env_ids]
            self.hand_positions[indices, :] = goal_positions
            self.hand_orientations[indices, :] = goal_orientations
            self.hand_linvels[indices, :] = goal_linvels
            self.hand_angvels[indices, :] = goal_angvels
        return

    def _reset_table(self, env_ids):
        indices = self.table_indices[env_ids]
        self.root_state_tensor[indices, :] = self.saved_root_tensor[indices]
        return

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.get_pose_quat()
        actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone()

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:24],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + (
                                                                1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000


            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions, self.id)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                ##self.add_debug_lines(self.envs[i], self.block_left_handle_pos[i], self.block_left_handle_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
        object_id_buf, object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                              + torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - right_hand_lf_pos, p=2, dim=-1)
                              + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,
                                         right_hand_finger_dist)
    right_hand_dist_rew = right_hand_dist
    right_hand_finger_dist_rew = right_hand_finger_dist

    action_penalty = torch.sum(actions ** 2, dim=-1)

    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(
        torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)
    delta_value = 0.3 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.02 * delta_qpos_value
    target_flag = (delta_hand_pos_value <= 0.6).int() + (delta_hand_rot_value <= 1.8).int() + (delta_qpos_value <= 9.0).int()
    
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    lowest = object_pos[:, 2]
    lift_z = object_init_z[:, 0] + 0.6 +0.003

    if goal_cond:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
        
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= lift_z, torch.where(flag == 5, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag == 5, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()  #+ target_flag 
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5*delta_value

    else:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= lift_z, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus

    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)

# modify: now only checks for depth_bar, moves z_p z_n check out
def depth_image_to_point_cloud_GPU_batch(
        camera_depth_tensor_batch, camera_rgb_tensor_batch,
        camera_seg_tensor_batch, camera_view_matrix_inv_batch,
        camera_proj_matrix_batch, u, v, width: float, height: float,
        depth_bar: float, device: torch.device,
        # z_p_bar: float = 3.0,
        # z_n_bar: float = 0.3,
):
    
    batch_num = camera_depth_tensor_batch.shape[0]

    depth_buffer_batch = mov(camera_depth_tensor_batch, device)
    rgb_buffer_batch = mov(camera_rgb_tensor_batch, device) / 255.0
    seg_buffer_batch = mov(camera_seg_tensor_batch, device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj_batch = camera_proj_matrix_batch
    fu_batch = 2 / proj_batch[:, 0, 0]
    fv_batch = 2 / proj_batch[:, 1, 1]

    centerU = width / 2
    centerV = height / 2

    Z_batch = depth_buffer_batch

    Z_batch = torch.nan_to_num(Z_batch, posinf=1e10, neginf=-1e10)

    X_batch = -(u.view(1, u.shape[-2], u.shape[-1]) - centerU) / width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1]) - centerV) / height * Z_batch * fv_batch.view(-1, 1, 1)

    R_batch = rgb_buffer_batch[..., 0].view(batch_num, 1, -1)
    G_batch = rgb_buffer_batch[..., 1].view(batch_num, 1, -1)
    B_batch = rgb_buffer_batch[..., 2].view(batch_num, 1, -1)
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)

    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar

    Z_batch = Z_batch.view(batch_num, 1, -1)
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)

    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    # (b, N, 8)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[..., 0:4] = position_batch[..., 0:4] @ vinv_batch

    points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
    valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch

    return points_batch, valid_batch


import sys
sys.path.append(osp.realpath(osp.join(osp.realpath(__file__), '../../../..')))
from pointnet2_ops import pointnet2_utils


def sample_points(points, sample_num, sample_method, device):
    if sample_method == 'random':
        raise NotImplementedError

    elif sample_method == "furthest_batch":
        idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous(), sample_num).long()
        idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
        sampled_points = torch.gather(points, dim=1, index=idx)

    elif sample_method == 'furthest':
        eff_points = points[points[:, 2] > 0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num:
            eff_points = points[:, 0:3].contiguous()
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points_xyz.reshape(1, *eff_points_xyz.shape),
                                                                  sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
    else:
        assert False
    return sampled_points
