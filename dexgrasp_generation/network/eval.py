from hydra import compose, initialize
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_mesh_dataloader
from trainer import Trainer
from utils.global_utils import result_to_loader, flatten_result
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin
import json
from tqdm import tqdm, trange

import argparse

from train import process_config
from utils.interrupt_handler import InterruptHandler
from network.models.contactnet.contact_network import ContactMapNet
from utils.hand_model import AdditionalLoss, add_rotation_to_hand_pose
from utils.eval_utils import KaolinModel, eval_result
from utils.visualize import visualize

from pytorch3d import transforms as pttf



def main(cfg):
    cfg = process_config(cfg)

    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EvalModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



    """ DataLoaders """
    test_loader = get_mesh_dataloader(cfg, "test")

    """ Trainer """
    trainers = []
    for key in cfg['models'].keys():
        net_cfg = compose(f"{cfg['models'][key]['type']}_config")
        with open_dict(net_cfg):
            net_cfg['device'] = cfg['device']
        trainer = Trainer(net_cfg, logger)
        trainer.resume()
        trainers.append(trainer)

    contact_cfg = compose(f"{cfg['tta']['contact_net']['type']}_config")
    with open_dict(contact_cfg):
        contact_cfg['device'] = cfg['device']
    contact_net = ContactMapNet(contact_cfg).to(cfg['device'])
    contact_net.eval()
    tta_loss = AdditionalLoss(cfg['tta'], 
                              cfg['device'], 
                              cfg['dataset']['num_obj_points'], 
                              cfg['dataset']['num_hand_points'], contact_net)

    """ Test """
    result = None
    # sample
    for key, trainer in zip(cfg['models'].keys(), trainers):
        loader = result_to_loader(result, cfg) if result else test_loader
        result = []
        for _, data in enumerate(tqdm(loader)):
            for i in range(cfg['models'][key]['sample_num']):
                pred_dict, _ = trainer.test(data)
                data.update(pred_dict)
                result.append({k: v.cpu() if type(v) == torch.Tensor else v for k, v in data.items()})
    
    # tta
    loader = result_to_loader(result, cfg, cfg['tta']['batch_size'])
    result = []
    for i, data in enumerate(tqdm(loader)):
        points = data['canon_obj_pc'].cuda()
        hand_pose = torch.cat([data['canon_translation'], torch.zeros_like(data['canon_translation']), data['hand_qpos']], dim=-1)
        old_hand_pose = hand_pose.clone()
        data['hand_pose'] = add_rotation_to_hand_pose(old_hand_pose, data['sampled_rotation'])
        plane_parameters = data['canon_plane'].cuda()

        hand_pose = hand_pose.cuda()
        hand = tta_loss.hand_model(hand_pose, with_surface_points=True)
        discretized_cmap_pred = tta_loss.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))['contact_map'].exp()
        cmap_pred = (torch.argmax(discretized_cmap_pred, dim=-1) + 0.5) / discretized_cmap_pred.shape[-1]
        
        hand_pose.requires_grad_()
        optimizer = torch.optim.Adam([hand_pose], lr=cfg['tta']['lr'])
        for t in range(cfg['tta']['iterations']):
            optimizer.zero_grad()
            loss = tta_loss.tta_loss(hand_pose, points, cmap_pred, plane_parameters)
            loss.backward()
            optimizer.step()

        data['tta_hand_pose'] = add_rotation_to_hand_pose(hand_pose.detach().cpu(), data['sampled_rotation'])
        result.append(data)

    result = flatten_result(result)

    hand_model = tta_loss.hand_model
    object_model = KaolinModel(
        data_root_path='data/DFCData/meshes',
        batch_size_each=1,
        device=cfg['device']
    )

    final_results = []
    for i in trange(len(result['hand_pose'])):
        final_results.append(eval_result(cfg['q1'], {k: result[k][i] for k in result.keys()}, hand_model, object_model, cfg['device']))
    result.update(flatten_result(final_results))
    
    seen_result, unseen_result = divide(result)

    output_result(seen_result, 'seen')
    output_result(unseen_result, 'unseen')


def divide(data):
    seen_data = []
    unseen_data = []

    object_code_list_train = []
    for splits_file_name in os.listdir('data/DFCData/splits'):
        with open(os.path.join('data/DFCData/splits', splits_file_name), 'r') as f:
            splits_map = json.load(f)
        object_code_list_train += [os.path.join(splits_file_name[:-5], object_code) for object_code in splits_map['train']]

    category_set_train = set()
    for object_code in object_code_list_train:
        category_set_train.add(object_code.split('-')[0])
    
    for i in range(len(data['object_code'])):
        object_code = data['object_code'][i]
        if 'ddg/' in object_code or 'mujoco/' in object_code or object_code.split('-')[0] in category_set_train:
            seen_data.append({k: v[i] for k, v in data.items()})
        else:
        #if 'ddg/' not in object_code and 'mujoco/' not in object_code and object_code.split('-')[0] not in category_set_train:
            unseen_data.append({k: v[i] for k, v in data.items()})
    
    return seen_data, unseen_data

def output_result(data, name):
    if len(data) == 0:
        return
    for has_tta in ['', 'tta_']:
        for label in ['q1', 'pen', 'tpen', 'valid_q1']:
            result = [dic[has_tta+label] for dic in data]
            print(f'category {name}, {has_tta+label}: {sum(result)/len(result)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './eval_result'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    main(cfg)
