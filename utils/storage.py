# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import gym


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space):

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.actions = torch.zeros(num_steps, num_processes, action_space.n)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):


        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'actions': self.actions.view(-1)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
            }

    def recurrent_generator(self, advantages, num_mini_batch):


        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space, rec_state_size=0):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        
        # 初始化 rec_states
        if rec_state_size > 0:
            self.rec_states = torch.zeros(num_steps + 1, num_processes, rec_state_size)
        else:
        # 提供一个默认的零张量
            self.rec_states = torch.zeros(num_steps + 1, num_processes, 1)

        # 根据动作空间类型初始化动作存储
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = 1
        elif isinstance(action_space, gym.spaces.Box):
            action_shape = action_space.shape[0]
        else:
            raise NotImplementedError("不支持的动作空间类型")
        
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.extras = torch.zeros(num_steps + 1, num_processes, 2)  # Adjust the size as needed
        



        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        # 如果 rec_states 为 None，使用全零张量
        if rec_states is None:
            rec_states = torch.zeros_like(self.rec_states[self.step + 1])

        self.rec_states[self.step + 1].copy_(rec_states)
        
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)
