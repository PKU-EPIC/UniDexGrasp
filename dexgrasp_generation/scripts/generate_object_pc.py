"""
Last modified date: 2023.06.06
Author: Jialiang Zhang
Description: Generate object point clouds
"""

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import transforms3d
import torch
import pytorch3d.io
import pytorch3d.ops
import pytorch3d.structures
import sapien.core as sapien
from multiprocessing import Pool, current_process
from tqdm import tqdm


def sample_projected(_):
    args, object_code, idx = _

    worker = current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list[(worker - 1) % len(args.gpu_list)]
    print(idx)

    object_path = os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj')

    # set simulator

    engine = sapien.Engine()
    engine.set_log_level('critical')
    renderer = sapien.VulkanRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene = engine.create_scene()

    rscene = scene.get_renderer_scene()
    rscene.set_ambient_light([0.5, 0.5, 0.5])
    rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(object_path, scale=[args.scale, args.scale, args.scale])
    object_actor = builder.build_kinematic()

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=args.width,
        height=args.height,
        fovx=0,
        fovy=np.deg2rad(35),
        near=args.near,
        far=args.far,
    )
    # print('Intrinsic matrix\n', camera.get_camera_matrix())


    # camera_eye = np.array([[0.2, -0.5, 1.0], [1.0, 0.2, 1.0], [0.2, 0.2, 1.4]])
    # camera_forward = np.array([[0, 4.0, 0], [-4.0, 0, 0], [0, -0.01, -2.0]]
    camera_forward = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    camera_eye = np.array([0, 0, args.camera_height]) - args.camera_distance * camera_forward
    # angles = np.linspace(0, 2 * np.pi, args.n_cameras, endpoint=False)
    # cam_pos_array = np.stack([
    #     args.camera_distance * np.sin(args.theta) * np.cos(angles),
    #     args.camera_distance * np.sin(args.theta) * np.sin(angles),
    #     args.camera_distance * np.cos(args.theta).repeat(args.n_cameras)
    # ], axis=1)
    # cam_pos_array = np.stack(np.meshgrid([-args.camera_distance, args.camera_distance], [-args.camera_distance, args.camera_distance], [-args.camera_distance, args.camera_distance]), axis=-1).reshape(-1, 3)
    # print(f'n_camera: {len(cam_pos_array)}')

    # load poses

    pose_matrices = np.load(os.path.join(args.data_root_path, object_code, 'poses.npy'))
    pose_matrices = pose_matrices if len(pose_matrices) <= args.n_poses else pose_matrices[:args.n_poses]

    # sample pc

    pcs = []

    if os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy')):
        pcs_old = np.load(os.path.join(args.data_root_path, object_code, 'pcs.npy'))
        for pc in pcs_old:
            pcs.append(pc)
        if len(pcs) >= args.n_poses:
            return

    for pose_matrix in pose_matrices[len(pcs):]:

        pc = []

        translation = pose_matrix[:3, 3]
        translation *= args.scale
        translation[:2] = 0
        rotation_matrix = pose_matrix[:3, :3]
        rotation_quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

        # for cam_pos in cam_pos_array:
        for idx_camera in range(len(camera_eye)):

            # Compute the camera pose by specifying forward(x), left(y) and up(z)
            cam_pos = camera_eye[idx_camera]
            forward = camera_forward[idx_camera]
            # forward = -cam_pos / np.linalg.norm(cam_pos)
            left = np.cross([0, 0, 1], forward)
            left = np.cross([0, 1, 0], forward) if np.linalg.norm(left) < 0.01 else left
            left = left / np.linalg.norm(left)
            up = np.cross(forward, left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([forward, left, up], axis=1)
            mat44[:3, 3] = cam_pos
            camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

            # render

            object_actor.set_pose(sapien.Pose(translation, rotation_quaternion))
            scene.step()  # make everything set
            scene.update_render()
            camera.take_picture()

            # Each pixel is (x, y, z, is_valid) in camera space (OpenGL/Blender)
            position = camera.get_float_texture('Position')  # [H, W, 4]
            # OpenGL/Blender: y up and -z forward
            points_opengl = position[..., :3][position[..., 2] != 0]
            # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
            # camera.get_model_matrix() must be called after scene.update_render()!
            model_matrix = camera.get_model_matrix()
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            pc.append(points_world)

        pc = np.concatenate(pc)
        pc_down_sampled = pc[np.random.choice(len(pc), args.max_n_points, replace=False)] if len(pc) > args.max_n_points else pc
        pc_sampled = pytorch3d.ops.sample_farthest_points(torch.tensor(pc_down_sampled).unsqueeze(0), K=args.num_samples)[0][0]
        pc_sampled = (pc_sampled - translation) @ rotation_matrix / args.scale
        pcs.append(pc_sampled)
        # print(f'n_points: {len(pc)}')

    pcs = np.stack(pcs)

    np.save(os.path.join(args.data_root_path, object_code, 'pcs.npy'), pcs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiments settings
    parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--n_poses', type=int, default=100)
    parser.add_argument('--max_n_points', type=int, default=9000)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--n_cpu', type=int, default=8)
    # parser.add_argument('--n_cameras', type=int, default=6)
    # parser.add_argument('--theta', type=float, default=np.pi / 4)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--gpu_list', type=str, nargs='*', default=['0', '1', '2', '3'])
    # camera settings
    parser.add_argument('--camera_distance', type=float, default=0.5)
    parser.add_argument('--camera_height', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100)
    args = parser.parse_args()

    object_category_list = os.listdir(args.data_root_path)
    object_code_list = []
    for object_category in object_category_list:
        object_code_list += [os.path.join(object_category, object_code) for object_code in sorted(os.listdir(os.path.join(args.data_root_path, object_category)))]
    # object_code_list = [object_code for object_code in object_code_list if not os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy'))]

    # object_code_list = object_code_list[:1]

    parameters = []
    for idx, object_code in enumerate(object_code_list):
        parameters.append((args, object_code, idx))
    
    with Pool(args.n_cpu) as p:
        it = tqdm(p.imap(sample_projected, parameters), desc='sampling', total=len(parameters))
        list(it)
