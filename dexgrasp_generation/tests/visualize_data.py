import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.realpath('.'))

from hydra import compose, initialize
import pytorch3d.transforms
import argparse
import random
import numpy as np
import torch
import plotly.graph_objects as go
from datasets.dex_dataset import DFCDataset
from utils.hand_model import HandModel


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num', type=int, default=100)
args = parser.parse_args()

initialize(version_base=None, config_path="../configs", job_name="train")
cfg = compose(config_name='glow_config.yaml')

# seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
random.seed(args.seed)

# dataset
dataset = DFCDataset(cfg, mode=args.mode)
print(len(dataset))
data_dict = dataset[args.num]
hand_pose = torch.cat([
    torch.tensor(data_dict['canon_translation'], dtype=torch.float), 
    pytorch3d.transforms.matrix_to_axis_angle(torch.tensor(data_dict['canon_rotation'], dtype=torch.float)), 
    torch.tensor(data_dict['hand_qpos'], dtype=torch.float)
])

# hand model
hand_model = HandModel(
    mjcf_path='data/mjcf/shadow_hand.xml',
    mesh_path='data/mjcf/meshes',
)
hand = hand_model(hand_pose.unsqueeze(0), with_meshes=True)

# visualize
object_pc = data_dict['canon_obj_pc'][:3000]
table_pc = data_dict['canon_obj_pc'][3000:]
object_plotly = go.Scatter3d(x=object_pc[:, 0], y=object_pc[:, 1], z=object_pc[:, 2], mode='markers', marker=dict(size=2, color='lightgreen'))
table_pc = go.Scatter3d(x=table_pc[:, 0], y=table_pc[:, 1], z=table_pc[:, 2], mode='markers', marker=dict(size=2, color='lightgrey'))
hand_plotly = go.Mesh3d(x=hand['vertices'][0, :, 0], y=hand['vertices'][0, :, 1], z=hand['vertices'][0, :, 2], i=hand['faces'][:, 0], j=hand['faces'][:, 1], k=hand['faces'][:, 2], color='lightblue', opacity=1)
fig = go.Figure([object_plotly, table_pc, hand_plotly])
fig.show()
