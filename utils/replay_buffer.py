import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, args):
        self.max_size = args.buffer_size
        self.batch_size = args.batch_size
        self.device = args.device

        # 初始化存储空间
        self.states = np.zeros((self.max_size, args.state_dim))
        self.actions = np.zeros((self.max_size, 1))
        self.action_log_probs = np.zeros((self.max_size, 1))
        self.rewards = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, args.state_dim))
        self.dones = np.zeros((self.max_size, 1))

        self.ptr = 0
        self.size = 0

    def add(self, state, action, action_log_prob, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.action_log_probs[self.ptr] = action_log_prob
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        # 随机采样
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        action_log_probs = torch.FloatTensor(self.action_log_probs[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, action_log_probs, rewards, next_states, dones

    def clear(self):
        self.ptr = 0
        self.size = 0 