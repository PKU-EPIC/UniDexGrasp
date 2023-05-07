from datetime import datetime
from importlib.resources import path
import os
import os.path as osp
import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time

from matplotlib.patches import FancyArrow
from gym import spaces

from gym.spaces import Space

import numpy as np
import statistics
import copy
import yaml
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algorithms.rl.dagger_value.storage import RolloutStorage, PPORolloutStorage, PERBuffer



class DAGGERVALUE:

    def __init__(self,
                 vec_env,
                 actor_class,
                 actor_critic_class,
                 actor_critic_class_expert,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 buffer_size,
                 init_noise_std=1.0,
                 learning_rate=1e-3,
                 schedule="fixed",
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 expert_chkpt_path = "",
                 is_vision = False
                 ):

        # cfg_path = 'cfg/dagger_value/config.yaml'
        # with open(cfg_path, 'r') as f:
        #     cfg = yaml.safe_load(f)
        self.is_vision = is_vision

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.schedule = schedule
        self.step_size = learning_rate

        # DAGGER components
        self.buffer_size = buffer_size
        self.vec_env = vec_env
        
        # value_net
        # multi_expert
        self.is_vision_expert = False
        if self.is_vision_expert:
            self.expert_observation_space = self.observation_space
        else:
            self.expert_observation_space = spaces.Box(np.ones(300) * -np.Inf, np.ones(300))

        self.storage = RolloutStorage(self.vec_env.num_envs, self.buffer_size, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        
        # DAGGER parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset
        
        # value_net 
        self.value_loss_cfg = cfg['learn']['value_loss']
        self.apply_value_net = self.value_loss_cfg['apply']
        if self.apply_value_net:
            self.actor = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                            init_noise_std, model_cfg, asymmetric=asymmetric, use_pc = self.is_vision)
            
            self.optimizer = optim.Adam(self.actor.actor.parameters(), lr=learning_rate)
            self.optimizer_value = optim.Adam(self.actor.critic.parameters(), lr=learning_rate)

            self.ppo_buffer = PPORolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                    self.state_space.shape, self.action_space.shape, self.device, sampler)
         
        else:
            self.actor = actor_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                        init_noise_std, model_cfg, asymmetric=asymmetric, use_pc = self.is_vision)
            self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.actor.to(self.device)

        # multi_expert
        #if not self.is_testing:
        self.expert_cfg_list = cfg['learn']['expert']
        self.expert_list = []
        for expert_cfg in self.expert_cfg_list:
            expert = actor_critic_class_expert(self.expert_observation_space.shape, self.state_space.shape, self.action_space.shape,
                                            init_noise_std, model_cfg, asymmetric=asymmetric, use_pc = self.is_vision_expert)
            expert.to(self.device)
            expert.load_state_dict(torch.load(expert_cfg['path'], map_location=self.device))
            self.expert_list.append(expert)
                
                
            self.task = self.vec_env.task
            id2expert_id = []
            for obj_id, obj_code in enumerate(self.task.object_code_list):
                result_id = -1
                for expert_id, expert_cfg in enumerate(self.expert_cfg_list):
                    if obj_code in expert_cfg['object_code_dict']:
                        result_id = expert_id
                        break
                if result_id == -1:
                    result_id = 0
                    print(f'{obj_code} not covered by all experts')
                id2expert_id.append(result_id)
            self.id2expert_id = torch.tensor(id2expert_id, dtype=torch.int64, device=self.device)
            expert_id_buf = self.id2expert_id[self.task.object_id_buf]
            for expert_id, expert_cfg in enumerate(self.expert_cfg_list):
                expert_indices = torch.where(expert_id_buf == expert_id)[0]
                expert_cfg['indices'] = expert_indices
                print(f"Expert {expert_cfg['name']} covers indices {expert_indices}")

    def get_all_checkpoints_in_dir(self, workdir: str):
        model_dir_list = []
        for file in os.listdir(workdir):
            if file.endswith('.pt'):
                model_dir_list.append(osp.join(workdir, file))
        return model_dir_list

    def test(self, path):
        a = path.rfind('/')
        buffer_path = path[:a]+'/buffer.yaml'
        self.eval_result_savedir = buffer_path

        self.actor.load_state_dict(torch.load(path,map_location=self.device))
        self.actor.eval()

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = 0 #int(path.split("_")[-1].split(".")[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    # multi_expert
    def expert_inference(self, current_obs):
        action = torch.zeros((current_obs.shape[0], self.action_space.shape[0]), device=self.device)
        for expert, expert_cfg in zip(self.expert_list, self.expert_cfg_list):
            indices = expert_cfg['indices']
            action[indices] = expert.act_inference(current_obs[indices])
        return action

    def run(self, num_learning_iterations, log_interval=1):
        id = -1
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            length1 = self.vec_env.task.max_episode_length
            for _ in range(length1):
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        # value_net
                        if self.apply_value_net:
                            current_states = self.vec_env.get_state()
                    # Compute the action
                    id = (id+1)%self.vec_env.task.max_episode_length
                    if self.apply_value_net:
                        actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs, current_states)
                    else:
                        actions = self.actor.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions,id)
                    current_obs.copy_(next_obs)
                if _ == length1-2:
                    success_rate=self.vec_env.task.successes.sum()/self.vec_env.num_envs
            print("success_rate:",success_rate)

            import yaml
            eval_results = {'success_rate': success_rate.item()}
            with open(self.eval_result_savedir, 'w') as f:
                yaml.dump(eval_results, f)
            
        else:
            #expert
            # multi_expert
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        # value_net
                        if self.apply_value_net:
                            current_states = self.vec_env.get_state()
                    id = (id+1)%self.vec_env.task.max_episode_length

                    # for random_load, expert should change
                    if id == 0:
                        for expert_cfg in self.expert_cfg_list:
                            self.task = self.vec_env.task
                            expert_id_buf = self.id2expert_id[self.task.object_id_buf]
                            for expert_id, expert_cfg in enumerate(self.expert_cfg_list):
                                expert_indices = torch.where(expert_id_buf == expert_id)[0]
                                expert_cfg['indices'] = expert_indices
                                print(f"Expert {expert_cfg['name']} covers indices {expert_indices}")

                    # Compute the action
                    # value_net
                    if self.apply_value_net:
                        actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs, current_states)
                    else:
                        actions = self.actor.act_inference(current_obs)

                    # multi_expert
                    actions_expert = self.expert_inference(current_obs)

                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, actions_expert, rews, dones)
                    current_obs.copy_(next_obs)

                    # value_net
                    if self.apply_value_net:
                        self.ppo_buffer.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                        current_states.copy_(next_states)
                    
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                
                # value_net
                if self.apply_value_net:
                    actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs, current_states)
                else:
                    _ = self.actor.act_inference(current_obs)

                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                if self.apply_value_net:
                    self.ppo_buffer.compute_returns(values, self.value_loss_cfg['gamma'], self.value_loss_cfg['lam'])
                    mean_policy_loss, mean_value_loss = self.update()
                    self.ppo_buffer.clear()
                else:
                    mean_policy_loss = self.update()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_policy_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        if self.apply_value_net:
            mean_value_loss = 0
            batch_value = self.ppo_buffer.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]

                actions_expert_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]

                # value_net
                if self.apply_value_net:
                    actions_batch = self.actor.act_withgrad(obs_batch)
                else:
                    actions_batch = self.actor.act(obs_batch)

                # Policy loss
                dagger_loss = F.mse_loss(actions_batch, actions_expert_batch)

                # Gradient step
                self.optimizer.zero_grad()
                dagger_loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_policy_loss += dagger_loss.item()
        
            if self.apply_value_net:
                for indices in batch_value:
                    obs_batch = self.ppo_buffer.observations.view(-1, *self.ppo_buffer.observations.size()[2:])[indices]
                    if self.asymmetric:
                        states_batch = self.ppo_buffer.states.view(-1, *self.ppo_buffer.states.size()[2:])[indices]
                    else:
                        states_batch = None
                    actions_batch = self.ppo_buffer.actions.view(-1, self.ppo_buffer.actions.size(-1))[indices]
                    target_values_batch = self.ppo_buffer.values.view(-1, 1)[indices]
                    returns_batch = self.ppo_buffer.returns.view(-1, 1)[indices]

                    actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor.evaluate(
                        obs_batch,
                        states_batch,
                        actions_batch)

                    if self.value_loss_cfg['use_clipped_value_loss']:
                        clip_range = self.value_loss_cfg['clip_range']
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_range, clip_range)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                    value_loss = value_loss * self.value_loss_cfg['value_loss_coef']

                    # Gradient step
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer_value.step()

                    mean_value_loss += value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates

        if self.apply_value_net:
            mean_value_loss /= num_updates
        
            return mean_policy_loss, mean_value_loss
        else:
            return mean_policy_loss
