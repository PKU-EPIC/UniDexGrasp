from algo.pn_utils.maniskill_learn.utils.meta import ConfigDict
from numbers import Number
from torch.nn.parameter import Parameter
import torch


def combine_obs_with_action(obs, action=None):
    if isinstance(obs, dict):
        # For point cloud input
        assert 'state' in obs and 'pointcloud' in obs
        inputs = obs.copy()
        inputs['state'] = torch.cat([obs["state"], action], dim=-1) if action is not None else obs["state"]
    else:
        inputs = torch.cat([obs, action], dim=-1) if action is not None else obs
    return inputs


def get_kwargs_from_shape(obs_shape, action_shape):
    replaceable_kwargs = {}
    if action_shape is not None:
        replaceable_kwargs['action_shape'] = action_shape
    if isinstance(obs_shape, dict):
        if 'pointcloud' in obs_shape.keys():
            # For mani_skill point cloud input
            replaceable_kwargs['pcd_all_channel'] = (
                obs_shape['pointcloud']['xyz'][-1] +
                obs_shape['pointcloud']['rgb'][-1] +
                obs_shape['pointcloud']['seg'][-1]
            )
            replaceable_kwargs['num_objs'] = obs_shape['pointcloud']['seg'][-1]
            replaceable_kwargs['pcd_xyz_rgb_channel'] = (
                obs_shape['pointcloud']['xyz'][-1] +
                obs_shape['pointcloud']['rgb'][-1]
            )
        if 'rgbd' in obs_shape.keys():
            # For mani_skill point rgbd input
            mode = list(obs_shape['rgbd'].keys())[0]
            # image format is H, W, C
            replaceable_kwargs['rgbd_channel'] = (
                obs_shape['rgbd'][mode]['rgb'][-1] +
                obs_shape['rgbd'][mode]['depth'][-1] +
                obs_shape['rgbd'][mode]['seg'][-1]
            )
        replaceable_kwargs['agent_shape'] = obs_shape['state']
    else:
        replaceable_kwargs['obs_shape'] = obs_shape
    return replaceable_kwargs


def replace_placeholder_with_args(parameters, **kwargs):
    if isinstance(parameters, ConfigDict):
        for key, v in parameters.items():
            parameters[key] = replace_placeholder_with_args(v, **kwargs)
        return parameters
    elif isinstance(parameters, (tuple, list)):
        type_of_parameters = type(parameters)
        parameters = list(parameters)
        for i, parameter in enumerate(parameters):
            parameters[i] = replace_placeholder_with_args(parameter, **kwargs)
        return type_of_parameters(parameters)
    elif isinstance(parameters, Number):
        return parameters
    elif isinstance(parameters, str):
        for key in kwargs:
            if key in parameters:
                parameters = parameters.replace(key, str(kwargs[key]))
        try:
            return eval(parameters)
        except:
            return parameters
    elif parameters is None:
        return None
    else:
        print(f'Strange type!! {parameters}, type of parameters {type(parameters)}')


def soft_update(target, source, tau):
    if isinstance(target, Parameter):
        target.data.copy_(target.data * (1.0 - tau) + source.data * tau)
    else:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    if isinstance(target, Parameter):
        target.data.copy_(source.data)
    else:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

