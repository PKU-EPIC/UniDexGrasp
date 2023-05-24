from hydra import compose, initialize
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dex_dataloader
from trainer import Trainer
from utils.global_utils import log_loss_summary, add_dict
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin
from tqdm import tqdm

import argparse

from utils.interrupt_handler import InterruptHandler


def process_config(cfg, save=True):
    root_dir = cfg["exp_dir"]
    os.makedirs(root_dir, exist_ok=True)

    with open_dict(cfg):
        cfg["device"] = f'cuda:{cfg["cuda_id"]}' if torch.cuda.is_available() else "cpu"

    if save:
        yaml_path = pjoin(root_dir, "config.yaml")
        print(f"Saving config to {yaml_path}")
        with open(yaml_path, 'w') as f:
            print(OmegaConf.to_yaml(cfg), file=f)

    return cfg


def log_tensorboard(writer, mode, loss_dict, cnt, epoch):
    for key, value in loss_dict.items():
        writer.add_scalar(mode + "/" + key, value / cnt, epoch)
    writer.flush()


def main(cfg):
    cfg = process_config(cfg)

    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("TrainModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    """ Tensorboard """
    writer = SummaryWriter(pjoin(log_dir, "tensorboard"))

    """ DataLoaders """
    train_loader = get_dex_dataloader(cfg, "train")
    test_loader = get_dex_dataloader(cfg, "test")

    """ Trainer """
    trainer = Trainer(cfg, logger)
    start_epoch = trainer.resume()

    """ Test """
    def test_all(dataloader, mode, iteration):
        test_loss = {}
        for _, data in enumerate(tqdm(dataloader)):
            _, loss_dict = trainer.test(data)
            loss_dict["cnt"] = 1
            add_dict(test_loss, loss_dict)

        cnt = test_loss.pop("cnt")
        log_loss_summary(test_loss, cnt,
                         lambda x, y: logger.info(f'{mode} {x} is {y}'))
        log_tensorboard(writer, mode, test_loss, cnt, iteration)

    """ Train """
    # Upon SIGINT, it will save the current model before exiting
    with InterruptHandler() as h:
        train_loss = {}
        for epoch in range(start_epoch, cfg["total_epoch"]):
            for _, data in enumerate(tqdm(train_loader)):
                loss_dict = trainer.update(data)
                loss_dict["cnt"] = 1
                add_dict(train_loss, loss_dict)

                if trainer.iteration % cfg["freq"]["plot"] == 0:
                    cnt = train_loss.pop("cnt")
                    log_loss_summary(train_loss, cnt,
                                     lambda x, y: logger.info(f"Train {x} is {y}"))
                    log_tensorboard(writer, "train", train_loss, cnt, trainer.iteration)

                    train_loss = {}

                if trainer.iteration % cfg["freq"]["step_epoch"] == 0:
                    trainer.step_epoch()

                if trainer.iteration % cfg["freq"]["test"] == 0:
                    test_all(test_loader, "test", trainer.iteration)

                if trainer.iteration % cfg["freq"]["save"] == 0:
                    trainer.save()

                if h.interrupted:
                    break

            if h.interrupted:
                break

    trainer.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="ipdf_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './ipdf_train'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    main(cfg)
