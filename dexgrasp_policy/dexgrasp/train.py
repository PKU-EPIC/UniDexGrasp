# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex

def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["ppo", "dagger", "dagger_value"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    else:
        print("Unrecognized algorithm!")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
