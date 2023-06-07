import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='data/DFCData/meshes')
    parser.add_argument('--object_code', type=str, default='core/bottle-1a7ba1f4c892e2da30711cdbdbc73924')
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()
    
    # load data
    pcs_table = np.load(os.path.join(args.data_root_path, args.object_code, 'pcs_table.npy'))
    poses = np.load(os.path.join(args.data_root_path, args.object_code, 'poses.npy'))
    pc_table = pcs_table[args.num]
    pose = poses[args.num]
    
    # visualize
    v = pc_table[:3000]
    object_plotly = go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightgreen'))
    v = pc_table[3000:]
    table_plotly = go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightblue'))
    fig = go.Figure([object_plotly, table_plotly])
    fig.show()

    v = pc_table[:3000] @ pose[:3, :3].T + pose[:3, 3]
    object_plotly = go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightgreen'))
    v = pc_table[3000:] @ pose[:3, :3].T + pose[:3, 3]
    table_plotly = go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightblue'))
    fig = go.Figure([object_plotly, table_plotly])
    fig.show()
    
