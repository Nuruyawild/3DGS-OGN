import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, args):
        # 打印调试信息
        print("Debug: Initializing ReplayBuffer")
        print(f"  args.action_space: {args.action_space}")
        print(f"  args.observation_space: {args.observation_space}")

        # 使用 getattr 提供默认值
        self.max_size = getattr(args, 'buffer_size', args.num_global_steps * args.num_processes)
        self.batch_size = getattr(args, 'mini_batch_size', 64)
        self.device = getattr(args, 'device', torch.device('cpu'))
        
        # 计算状态维度
        if hasattr(args.observation_space, 'shape'):
            obs_shape = args.observation_space.shape
            print(f" Debug:  Observation space shape: {obs_shape}")
            # 如果是图像类型的观察空间，我们可能需要调整维度
            if len(obs_shape) == 3:  # [channels, height, width]
                self.state_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
            else:
                self.state_dim = int(np.prod(obs_shape))
            print(f" Debug:  Calculated state_dim: {self.state_dim}")
        else:
            raise ValueError("Error: observation_space does not have shape attribute")
        
        # 检查 action_space 的类型并相应地设置 action_dim
        if hasattr(args.action_space, 'n'):  # Discrete action space
            self.action_dim = 2  # 根据实际动作维度设置
            print(f" Debug: Discrete action space with {args.action_space.n} possible actions")
        elif hasattr(args.action_space, 'shape'):  # Continuous action space
            self.action_dim = args.action_space.shape[0]
            print(f" Debug: Continuous action space with dimension {self.action_dim}")
        else:
            raise ValueError("Error: args.action_space is neither Discrete nor Box")

        # 初始化存储空间
        print(f"Initializing storage with shapes:")
        print(f"  states: ({self.max_size}, {self.state_dim})")
        print(f"  actions: ({self.max_size}, {self.action_dim})")
        
        self.states = np.zeros((self.max_size, self.state_dim))
        self.actions = np.zeros((self.max_size, self.action_dim))
        self.action_log_probs = np.zeros((self.max_size, self.action_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, self.state_dim))
        self.dones = np.zeros((self.max_size, 1))
        self.values = np.zeros((self.max_size, 1))

        self.ptr = 0
        self.size = 0
        self.count = 0

        # 打印初始化完成的信息
        print(f"ReplayBuffer initialized with max_size={self.max_size}, state_dim={self.state_dim}, action_dim={self.action_dim}")

    def add(self, state, action, action_log_prob, reward, next_state, done, value):
        try:
            # 确保输入是 numpy 数组并且维度正确
            state = np.array(state).flatten()
            next_state = np.array(next_state).flatten()
            
            # 检查并处理状态维度
            if state.size > self.state_dim:
                print(f"Debug: Warning: Truncating state from {state.size} to {self.state_dim}")
                state = state[:self.state_dim]
            elif state.size < self.state_dim:
                raise ValueError(f"Debug: State dimension {state.size} is smaller than expected {self.state_dim}")
                
            if next_state.size > self.state_dim:
                print(f"Debug: Warning: Truncating next_state from {next_state.size} to {self.state_dim}")
                next_state = next_state[:self.state_dim]
            elif next_state.size < self.state_dim:
                raise ValueError(f"Debug: Next state dimension {next_state.size} is smaller than expected {self.state_dim}")

            # 处理其他输入
            action = np.array(action).flatten()
            action_log_prob = np.array(action_log_prob).flatten()
            reward = np.array(reward).reshape(1)
            done = np.array(done).reshape(1)
            value = np.array(value).reshape(1)

            # 存储数据
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.action_log_probs[self.ptr] = action_log_prob
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done
            self.values[self.ptr] = value

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            self.count += 1

        except Exception as e:
            print(f"Debug: Error during add operation: {e}")
            print(f"Debug: Input shapes:")
            print(f"  state: {state.shape if hasattr(state, 'shape') else 'N/A'}")
            print(f"  action: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            print(f"  next_state: {next_state.shape if hasattr(next_state, 'shape') else 'N/A'}")
            print(f"Debug: Buffer shapes:")
            print(f"  states: {self.states.shape}")
            print(f"  actions: {self.actions.shape}")
            raise

    def sample(self):
        # 随机采样
        indices = np.random.choice(self.size, min(self.batch_size, self.size), replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        action_log_probs = torch.FloatTensor(self.action_log_probs[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        values = torch.FloatTensor(self.values[indices]).to(self.device)

        return states, actions, action_log_probs, rewards, next_states, dones, values

    def clear(self):
        # 重置所有缓冲区
        self.ptr = 0
        self.size = 0
        self.count = 0
        
        self.states.fill(0)
        self.actions.fill(0)
        self.action_log_probs.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
        self.values.fill(0)
    
    def store(self, state, action, action_log_prob, reward, next_state, done, value, terminal=None):
        # 兼容 PPO 实现的方法
        self.add(state, action, action_log_prob, reward, next_state, done, value)

    def numpy_to_tensor(self):
        # 兼容 PPO 实现的方法
        states, actions, action_log_probs, rewards, next_states, dones, values = self.sample()
        
        # 创建 done 和 dw 张量
        done = dones.clone().detach()  # 使用 clone().detach() 而不是 torch.tensor()
        dw = torch.zeros_like(done)  # 假设没有特殊的死亡或获胜状态
        
        # 只返回需要的 7 个值
        return states, actions, action_log_probs, rewards, next_states, dw, done