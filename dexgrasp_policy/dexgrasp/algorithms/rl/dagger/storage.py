import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:

    def __init__(self, num_envs, buffer_size, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(buffer_size, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.actions = torch.zeros(buffer_size, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()

        self.buffer_size = buffer_size
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, observations, actions, rewards, dones):
        self.step = self.step % self.buffer_size
        if self.step >= self.buffer_size:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1

    def clear(self):
        self.step = 0

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.buffer_size
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
