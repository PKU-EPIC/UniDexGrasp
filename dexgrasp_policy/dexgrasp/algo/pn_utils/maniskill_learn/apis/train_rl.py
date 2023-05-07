import itertools
import os
import os.path as osp
import time
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from mani_skill_learn.env import ReplayMemory
from mani_skill_learn.env import save_eval_statistics
from mani_skill_learn.utils.data import dict_to_str, get_shape, is_seq_of
from mani_skill_learn.utils.meta import get_logger, get_total_memory, td_format
from mani_skill_learn.utils.torch import TensorboardLogger, save_checkpoint
from mani_skill_learn.utils.math import split_num
from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch


class EpisodicStatistics:
    def __init__(self, num_procs):
        self.num_procs = num_procs
        self.current_lens = np.zeros(num_procs)
        self.current_rewards = np.zeros(num_procs)
        self.history_rewards = np.zeros(num_procs)
        self.history_lens = np.zeros(num_procs)
        self.history_counts = np.zeros(num_procs)

    def push(self, rewards, dones):
        n, running_steps = split_num(len(dones), self.num_procs)
        j = 0
        for i in range(n):
            for _ in range(running_steps[i]):
                self.current_lens[i] += 1
                self.current_rewards[i] += rewards[j]
                if dones[j]:
                    self.history_rewards[i] += self.current_rewards[i]
                    self.history_lens[i] += self.current_lens[i]
                    self.history_counts[i] += 1
                    self.current_rewards[i] = 0
                    self.current_lens[i] = 0
                j += 1

    def reset_history(self):
        self.history_lens *= 0
        self.history_rewards *= 0
        self.history_counts *= 0

    def reset_current(self):
        self.current_rewards *= 0
        self.current_lens *= 0

    def get_mean(self):
        num_episode = np.clip(np.sum(self.history_counts), a_min=1E-5, a_max=1E10)
        return np.sum(self.history_lens) / num_episode, np.sum(self.history_rewards) / num_episode

    def print_current(self):
        print(self.current_lens, self.current_rewards)

    def print_history(self):
        print(self.history_lens, self.history_rewards, self.history_counts)


class EveryNSteps:
    def __init__(self, interval=None):
        self.interval = interval
        self.next_value = interval

    def reset(self):
        self.next_value = self.interval

    def check(self, x):
        if self.interval is None:
            return False
        sign = False
        while x >= self.next_value:
            self.next_value += self.interval
            sign = True
        return sign

    def standard(self, x):
        return int(x // self.interval) * self.interval


def train_rl(agent, rollout, evaluator, env_cfg, replay, on_policy, work_dir, total_steps=1000000, warm_steps=10000,
             n_steps=1, n_updates=1, n_checkpoint=None, n_eval=None, init_replay_buffers=None, expert_replay_buffers = None, expert_replay = None, tmp_replay = None,
             init_replay_with_split=None, eval_cfg=None, replicate_init_buffer=1, num_trajs_per_demo_file=-1, m_steps = 1, 
              discrim_steps = 1, rl_steps = 1, is_GAIL = False, is_SAC_BC = False):
    logger = get_logger(env_cfg.env_name)

    import torch
    from mani_skill_learn.utils.torch import get_cuda_info

    replay.reset()

    if init_replay_buffers is not None and init_replay_buffers != '':
        replay.restore(init_replay_buffers, replicate_init_buffer, num_trajs_per_demo_file)
        logger.info(f'Initialize buffer with {len(replay)} samples')

    if (expert_replay is not None): 
        expert_replay.reset()
        assert expert_replay_buffers is not None and expert_replay_buffers != '' ,'missing expert trajectories'

        expert_replay.restore(expert_replay_buffers, replicate_init_buffer, num_trajs_per_demo_file)
        logger.info(f'Initialize expert_buffer with {len(expert_replay)} samples')


    if init_replay_with_split is not None:
        assert is_seq_of(init_replay_with_split) and len(init_replay_with_split) == 2
        # For mani skill only
        from mani_skill.utils.misc import get_model_ids_from_yaml
        folder_root = init_replay_with_split[0]
        model_split_file = get_model_ids_from_yaml(init_replay_with_split[1])
        if init_replay_with_split[1] is None:
            files = [str(_) for _ in Path(folder_root).glob('*.h5')]
        else:
            files = [str(_) for _ in Path(folder_root).glob('*.h5') if re.split('[_-]', _.name)[1] in model_split_file]
        replay.restore(files, replicate_init_buffer, num_trajs_per_demo_file)

    tf_logs = ReplayMemory(total_steps)
    tf_logs.reset()
    tf_logger = TensorboardLogger(work_dir)

    checkpoint_dir = osp.join(work_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(agent)
    if rollout is not None:
        logger.info(f'Rollout state dim: {get_shape(rollout.recent_obs)}, action dim: {len(rollout.random_action())}')
        rollout.reset()
        episode_statistics = EpisodicStatistics(rollout.n)
        episode_statistics2 = EpisodicStatistics(rollout.n)
        
        total_episodes = 0
    else:
        # Batch RL
        if 'obs' not in replay.memory:
            logger.error('Empty replay buffer for Batch RL!')
            exit(0)
        logger.info(f'State dim: {get_shape(replay["obs"])}, action dim: {replay["actions"].shape[-1]}')

    check_eval = EveryNSteps(n_eval)
    check_checkpoint = EveryNSteps(n_checkpoint)
    check_tf_log = EveryNSteps(1000)

    if warm_steps > 0:
        assert not on_policy
        assert rollout is not None
        if (agent.exp == False):
            trajectories = rollout.forward_with_policy(None, warm_steps)[0]
        else:
            trajectories = rollout.forward_with_policy(agent.exp_agent.policy, warm_steps)[0]
        episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])
        replay.push_batch(**trajectories)
        rollout.reset()
        episode_statistics.reset_current()
        check_eval.check(warm_steps)
        check_checkpoint.check(warm_steps)
        check_tf_log.check(warm_steps)
        logger.info(f"Finish {warm_steps} warm-up steps!")
    steps = warm_steps
    total_updates = 0
    begin_time = datetime.now()
    max_ETA_len = None

    for iteration_id in itertools.count(1):
        tf_logs.reset()
        if rollout:
            episode_statistics.reset_history()
            episode_statistics2.reset_history()

        if on_policy:
            replay.reset()
        train_dict = {}
        print_dict = OrderedDict()

        update_time = 0
        time_begin_episode = time.time()
        
        if n_steps > 0:
            if is_GAIL==False:
                # For online RL
                collect_sample_time = 0
                cnt_episodes = 0
                num_done = 0
                """
                For on-policy algorithm, we will print training infos for every gradient batch.
                For off-policy algorithm, we will print training infos for every n_steps epochs.
                """
                while num_done < n_steps and not (on_policy and num_done > 0):
                    for ___ in range(m_steps):
                        tmp_time = time.time()
                        trajectories, infos = rollout.forward_with_policy(agent.policy, n_steps, whole_episode=on_policy)
                        episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])
                        collect_sample_time += time.time() - tmp_time

                        num_done += np.sum(trajectories['episode_dones'])
                        cnt_episodes += np.sum(trajectories['episode_dones'].astype(np.int32))
                        replay.push_batch(**trajectories)
                        steps += n_steps

                    for i in range(n_updates):
                        total_updates += 1
                        tmp_time = time.time()
                        if is_SAC_BC:
                            tf_logs.push(**agent.update_parameters(replay, updates=total_updates, expert_replay=expert_replay))
                        else:
                            tf_logs.push(**agent.update_parameters(replay, updates=total_updates))
                        update_time += time.time() - tmp_time

                total_episodes += cnt_episodes
                train_dict['num_episode'] = int(cnt_episodes)
                train_dict['total_episode'] = int(total_episodes)
                train_dict['episode_time'] = time.time() - time_begin_episode
                train_dict['collect_sample_time'] = collect_sample_time

                print_dict['episode_length'], print_dict['episode_reward'] = episode_statistics.get_mean()
            else:
                # For GAIL
                collect_sample_time = 0
                cnt_episodes = 0
                num_done = 0

                tmp_replay.reset() 
                for _i_ in range(rl_steps):
                    while num_done < n_steps and not (on_policy and num_done > 0):
                        for ___ in range(m_steps):
                            tmp_time = time.time()
                            trajectories, infos = rollout.forward_with_policy(agent.policy, n_steps, whole_episode=on_policy)
                            expert_rewards = agent.expert_reward(trajectories['obs'], trajectories['actions'])
                            episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])
                            episode_statistics2.push(expert_rewards, trajectories['episode_dones'])
                            collect_sample_time += time.time() - tmp_time

                            num_done += np.sum(trajectories['episode_dones'])
                            cnt_episodes += np.sum(trajectories['episode_dones'].astype(np.int32))
                            replay.push_batch(**trajectories)
                            tmp_replay.push_batch(**trajectories)
                            steps += n_steps

                        for i in range(n_updates):
                            total_updates += 1
                            tmp_time = time.time()
                            tf_logs.push(**agent.update_parameters(replay, updates=total_updates))
                            update_time += time.time() - tmp_time
                
                tmploss = 0
                exploss = 0
                for _i_ in range(discrim_steps):
                    tmp_time = time.time()
                    el, tl = agent.update_discriminator(expert_replay, tmp_replay)
                    tmploss += tl
                    exploss += el
                    update_time += time.time() - tmp_time
                tmploss /= discrim_steps
                exploss /= discrim_steps

                total_episodes += cnt_episodes
                train_dict['num_episode'] = int(cnt_episodes)
                train_dict['total_episode'] = int(total_episodes)
                train_dict['episode_time'] = time.time() - time_begin_episode
                train_dict['collect_sample_time'] = collect_sample_time

                print_dict['episode_length'], print_dict['episode_reward'] = episode_statistics.get_mean()
                print_dict['episode_length'], print_dict['expert_reward'] = episode_statistics2.get_mean()
                print_dict['fake_sample_loss'] = tmploss
                print_dict['expert_sample_loss'] = exploss
        else:
            # For offline RL
            tf_logs.reset()
            for i in range(n_updates):
                steps += 1
                total_updates += 1
                tmp_time = time.time()
                tf_logs.push(**agent.update_parameters(replay, updates=total_updates))
                update_time += time.time() - tmp_time
        train_dict['update_time'] = update_time
        train_dict['total_updates'] = int(total_updates)
        train_dict['buffer_size'] = len(replay)
        train_dict['memory'] = get_total_memory('G', True)
        train_dict['cuda_mem'] = get_total_memory('G', True)

        train_dict.update(get_cuda_info(device=torch.cuda.current_device()))

        print_dict.update(tf_logs.tail_mean(n_updates))
        print_dict['memory'] = get_total_memory('G', False)
        print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))

        print_info = dict_to_str(print_dict)

        percentage = f'{(steps / total_steps) * 100:.0f}%'
        passed_time = td_format(datetime.now() - begin_time)
        ETA = td_format((datetime.now() - begin_time) * (total_steps / (steps - warm_steps) - 1))
        if max_ETA_len is None:
            max_ETA_len = len(ETA)

        logger.info(f'{steps}/{total_steps}({percentage}) Passed time:{passed_time} ETA:{ETA} {print_info}')
        if check_tf_log.check(steps):
            train_dict.update(dict(print_dict))
            tf_logger.log(train_dict, n_iter=steps, eval=False)

        if check_checkpoint.check(steps):
            standardized_ckpt_step = check_checkpoint.standard(steps)
            model_path = osp.join(checkpoint_dir, f'model_{standardized_ckpt_step}.ckpt')
            logger.info(f'Save model at step: {steps}.The model will be saved at {model_path}')
            agent.to_normal()
            save_checkpoint(agent, model_path)
            agent.recover_data_parallel()
        if check_eval.check(steps):
            standardized_eval_step = check_eval.standard(steps)
            logger.info(f'Begin to evaluate at step: {steps}. '
                        f'The evaluation info will be saved at eval_{standardized_eval_step}')
            eval_dir = osp.join(work_dir, f'eval_{standardized_eval_step}')
            agent.eval()
            torch.cuda.empty_cache()
            lens, rewards, finishes = evaluator.run(agent, **eval_cfg, work_dir=eval_dir)
            torch.cuda.empty_cache()
            save_eval_statistics(eval_dir, lens, rewards, finishes, logger)
            agent.train()

            eval_dict = {}
            eval_dict['mean_length'] = np.mean(lens)
            eval_dict['std_length'] = np.std(lens)
            eval_dict['mean_reward'] = np.mean(rewards)
            eval_dict['std_reward'] = np.std(rewards)
            tf_logger.log(eval_dict, n_iter=steps, eval=True)

        if steps >= total_steps:
            break
    if n_checkpoint:
        print(f'Save checkpoint at final step {total_steps}')
        agent.to_normal()
        save_checkpoint(agent, osp.join(checkpoint_dir, f'model_{total_steps}.ckpt'))
