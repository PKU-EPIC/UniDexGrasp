from torch.utils.data import DataLoader

import os
from os.path import join as pjoin
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))  # data -> model -> root, to import data_proc
sys.path.append(pjoin(base_dir, '..', '..'))  # data -> model -> root, to import data_proc

from datasets.dex_dataset import DFCDataset
from datasets.object_dataset import Meshdata


def get_dex_dataloader(cfg, mode="train", shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = DFCDataset(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"])

def get_mesh_dataloader(cfg, mode="train"):
    dataset = Meshdata(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
