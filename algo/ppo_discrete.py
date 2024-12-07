import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from .normalization import Normalization, RewardScaling
from .replaybuffer import ReplayBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta, Normal


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        
        # 获取输入维度
        if hasattr(args, 'observation_space'):
            obs_shape = args.observation_space.shape
            self.state_dim = int(np.prod(obs_shape))
        else:
            self.state_dim = args.state_dim
            
        print(f"Actor_Gaussian initialized with:")
        print(f"  state_dim: {self.state_dim}")
        print(f"  hidden_width: {args.hidden_width}")
        print(f"  action_dim: {args.action_dim}")
        
        # 网络结构
        self.fc1 = nn.Linear(self.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # 正交初始化
        if args.use_orthogonal_init:
            print("Using orthogonal initialization")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)


    def forward(self, s):
        try:
            # 确保输入是正确的类型和形状
            if not isinstance(s, torch.Tensor):
                s = torch.FloatTensor(s)
            if s.dim() == 1:
                s = s.unsqueeze(0)
                
            print(f"Forward pass - Input shape: {s.shape}")
            
            # 检查输入维度
            if s.shape[-1] != self.state_dim:
                raise ValueError(f"Input dimension {s.shape[-1]} does not match expected state_dim {self.state_dim}")
            
            # 网络前向传播
            x = self.activate_func(self.fc1(s))
            print(f"After fc1: {x.shape}")
            
            x = self.activate_func(self.fc2(x))
            print(f"After fc2: {x.shape}")
            
            mean = self.max_action * torch.tanh(self.mean_layer(x))
            print(f"Output mean shape: {mean.shape}")
            
            return mean
            
        except Exception as e:
            print(f"Debug: Error in forward pass: {e}")
            print(f"Debug: Input tensor info:")
            print(f"  shape: {s.shape}")
            print(f"  device: {s.device}")
            print(f"  dtype: {s.dtype}")
            print(f"Debug: Network info:")
            print(f"  fc1: in={self.fc1.in_features}, out={self.fc1.out_features}")
            print(f"  fc2: in={self.fc2.in_features}, out={self.fc2.out_features}")
            print(f"  mean_layer: in={self.mean_layer.in_features}, out={self.mean_layer.out_features}")
            raise



    def get_dist(self, s):
        try:
            # 获取均值
            mean = self.forward(s)
            print(f"Debug: Get_dist - Mean shape: {mean.shape}")
            
            # 计算标准差
            std = torch.exp(self.log_std).expand_as(mean)
            print(f"Debug: Get_dist - Std shape: {std.shape}")
            
            # 检查数值
            if torch.isnan(mean).any():
                raise ValueError("NaN values detected in mean")
            if torch.isnan(std).any():
                raise ValueError("NaN values detected in std")
                
            # 创建分布
            dist = Normal(mean, std)
            return dist
            
        except Exception as e:
            print(f"Debug: Error in get_dist: {e}")
            print(f"Debug: Input state shape: {s.shape}")
            print(f"Debug: FC1 weight shape: {self.fc1.weight.shape}")
            print(f"Debug: FC2 weight shape: {self.fc2.weight.shape}")
            print(f"Debug: Mean layer weight shape: {self.mean_layer.weight.shape}")
            raise e


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        
        # 获取输入维度
        if hasattr(args, 'observation_space'):
            # 如果是3D输入，展平处理
            obs_shape = args.observation_space.shape
            self.input_dim = int(np.prod(obs_shape))
        else:
            # 使用传入的 state_dim
            self.input_dim = args.state_dim
            
        print(f"Critic input dimension: {self.input_dim}")
        
        self.fc1 = nn.Linear(self.input_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]


    def forward(self, s):
        # 处理输入维度
        if len(s.shape) > 2:
            # 如果是3D输入 (batch, channels, height, width)，展平为2D
            s = s.reshape(s.shape[0], -1)
        elif len(s.shape) == 1:
            # 如果是1D输入，添加batch维度
            s = s.unsqueeze(0)
            
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_discrete():
    def __init__(self, args, state_dim=None, action_dim=None):
        # 使用传入的 state_dim 和 action_dim，如果没有传入则使用 args 中的值
        self.state_dim = state_dim or args.state_dim
        self.action_dim = action_dim or args.action_dim

        self.policy_dist = getattr(args, 'policy_dist', 'Gaussian')
        self.max_action = getattr(args, 'max_action', 1.0)
        # 使用 getattr 添加默认值
        self.batch_size = getattr(args, 'batch_size', 2048)  # 添加默认值
        self.mini_batch_size = getattr(args, 'mini_batch_size', 64)
        self.max_train_steps = getattr(args, 'max_train_steps', int(3e6))
        
        # 学习率
        self.lr_a = getattr(args, 'lr_a', 3e-4)
        self.lr_c = getattr(args, 'lr_c', 3e-4)
        
        # PPO 超参数
        self.gamma = getattr(args, 'gamma', 0.99)
        self.lamda = getattr(args, 'lamda', 0.95)
        self.epsilon = getattr(args, 'epsilon', 0.2)
        self.K_epochs = getattr(args, 'K_epochs', 10)
        
        # 其他超参数
        self.entropy_coef = getattr(args, 'entropy_coef', 0.01)
        self.set_adam_eps = getattr(args, 'set_adam_eps', True)
        self.use_grad_clip = getattr(args, 'use_grad_clip', True)
        self.use_lr_decay = getattr(args, 'use_lr_decay', True)
        self.use_adv_norm = getattr(args, 'use_adv_norm', True)

        # 创建 ReplayBuffer 实例
        self.replay_buffer = ReplayBuffer(args)  # 确保使用 ReplayBuffer

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        # 打印网络结构
        print("Debug: Actor Network:", self.actor)
        print("Debug: Critic Network:", self.critic)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def cuda(self):
        # 将 actor 和 critic 移动到 GPU
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()
        return self

    def to(self, device):
        # 将 actor 和 critic 移动到指定设备
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def prepare_buffer_from_rollouts(self, rollouts):
        """
        将 rollouts 数据转换并存储到 ReplayBuffer 中
        """
        # 清空当前的 ReplayBuffer
        print("Debug: Clearing ReplayBuffer...")
        self.replay_buffer.clear()
        
            # 获取 rollouts 中的数据
        T, N = rollouts.rewards.size(0), rollouts.rewards.size(1)
        
        # 遍历所有时间步和环境
        for t in range(T-1):  # T-1 因为最后一步没有下一个状态
            for n in range(N):
                # 获取当前状态
                state = rollouts.obs[t, n].view(-1).cpu().numpy()
                
                # 获取动作和动作概率
                action = rollouts.actions[t, n].cpu().numpy()
                action_log_prob = rollouts.action_log_probs[t, n].cpu().numpy()
                
                # 获取奖励
                reward = rollouts.rewards[t, n].cpu().numpy()
                
                # 获取下一个状态
                next_state = rollouts.obs[t + 1, n].view(-1).cpu().numpy()
                
                # 获取 done 标志
                done = (1 - rollouts.masks[t + 1, n].cpu().numpy())
                
                # 获取值函数预测
                value = rollouts.value_preds[t, n].cpu().numpy()
                
                # 存储到 ReplayBuffer
                self.replay_buffer.store(
                    state=state,
                    action=action,
                    action_log_prob=action_log_prob,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    value=value
                )
        
        # 打印调试信息
        print(f"Debug: Converted {T}x{N} steps of rollouts to ReplayBuffer")
        print(f"ReplayBuffer size: {self.replay_buffer.size}")

    def update(self, total_steps):
        try:
            # 获取所有数据
            s, a, a_log_p, r, s_, done, v = self.replay_buffer.numpy_to_tensor()
            print("\n=== Starting Update ===")
            print("Initial Data Shapes:")
            print(f"  states: {s.shape}")
            print(f"  actions: {a.shape}")
            print(f"  action_log_probs: {a_log_p.shape}")
            print(f"  rewards: {r.shape}")
            print(f"  next_states: {s_.shape}")
            print(f"  done: {done.shape}")
            print(f"  values: {v.shape}")
            
            # 确保所有数据都在同一设备上
            device = next(self.actor.parameters()).device
            print(f"\nMoving data to device: {device}")
            s = s.to(device)
            a = a.to(device)
            a_log_p = a_log_p.to(device)
            r = r.to(device)
            s_ = s_.to(device)
            done = done.to(device)
            v = v.to(device)

            # 计算 GAE
            print("\nCalculating GAE...")
            adv = []
            gae = 0
            with torch.no_grad():
                # 获取下一个状态的价值
                v_ = self.critic(s_)  # 保持原始维度 [64, 1]
                print(f"  v_ shape: {v_.shape}")
                
                # 计算 delta，确保所有张量维度一致
                delta = r + self.gamma * v_ * (1 - done) - v  # 所有输入都是 [64, 1]
                print(f"  delta shape: {delta.shape}")
                
                # 计算 GAE
                delta = delta.detach().cpu().numpy()  # 转换为 numpy 方便处理
                for t in range(len(delta) - 1, -1, -1):
                    gae = delta[t] + self.gamma * self.lamda * gae
                    adv.insert(0, gae)
                
                # 转换回 tensor
                adv = torch.FloatTensor(adv).to(device)  # [64, 1]
                print(f"  adv shape before reshape: {adv.shape}")
                
                # 确保维度正确
                if len(adv.shape) == 1:
                    adv = adv.unsqueeze(-1)  # 如果需要，添加维度变成 [64, 1]
                print(f"  adv shape after reshape: {adv.shape}")
                
                # 计算目标值
                v_target = adv + v  # 现在两者都是 [64, 1]
                print(f"  v_target shape: {v_target.shape}")
                
                if self.use_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    print(f"  normalized adv shape: {adv.shape}")

            # 开始多次训练
            print(f"\nStarting {self.K_epochs} training epochs...")
            for epoch in range(self.K_epochs):
                print(f"\nEpoch {epoch + 1}/{self.K_epochs}")
                batch_size = len(s)
                index = torch.randperm(batch_size)
                
                # 分批处理
                for start in range(0, batch_size, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, batch_size)
                    idx = index[start:end]
                    
                    # 获取当前批次数据
                    s_batch = s[idx]
                    a_batch = a[idx]
                    a_log_p_batch = a_log_p[idx]
                    adv_batch = adv[idx]
                    v_target_batch = v_target[idx]
                    
                    print(f"\nBatch {start//self.mini_batch_size + 1} shapes:")
                    print(f"  s_batch: {s_batch.shape}")
                    print(f"  a_batch: {a_batch.shape}")
                    print(f"  a_log_p_batch: {a_log_p_batch.shape}")
                    print(f"  adv_batch: {adv_batch.shape}")
                    print(f"  v_target_batch: {v_target_batch.shape}")
                    
                    # 计算新的动作分布
                    dist_now = self.actor.get_dist(s_batch)
                    dist_entropy = dist_now.entropy().mean()
                    a_log_p_now = dist_now.log_prob(a_batch)
                    print(f"  new log_prob shape: {a_log_p_now.shape}")
                    
                    # 确保 log_prob 维度正确
                    if len(a_log_p_now.shape) == 1:
                        a_log_p_now = a_log_p_now.unsqueeze(-1)
                    if len(a_log_p_batch.shape) == 1:
                        a_log_p_batch = a_log_p_batch.unsqueeze(-1)
                    print(f"  adjusted log_prob shapes:")
                    print(f"    a_log_p_now: {a_log_p_now.shape}")
                    print(f"    a_log_p_batch: {a_log_p_batch.shape}")
                    
                    # 计算比率和损失
                    ratio = torch.exp(a_log_p_now - a_log_p_batch)
                    print(f"  ratio shape: {ratio.shape}")
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_batch
                    a_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算 critic loss
                    v_s = self.critic(s_batch).squeeze(-1)
                    print(f"  v_s shape: {v_s.shape}")
                    c_loss = torch.mean(F.mse_loss(v_target_batch, v_s))
                    
                    # 总损失
                    loss = a_loss - dist_entropy * self.entropy_coef + c_loss
                    print(f"\nLoss values:")
                    print(f"  actor_loss: {a_loss.item():.4f}")
                    print(f"  critic_loss: {c_loss.item():.4f}")
                    print(f"  entropy: {dist_entropy.item():.4f}")
                    print(f"  total_loss: {loss.item():.4f}")
                    
                    # 更新网络
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()

            # Learning rate decay
            if self.use_lr_decay:
                print("\nPerforming learning rate decay...")
                self.lr_decay(total_steps)

            print("\n=== Update Complete ===\n")
            return c_loss.item(), a_loss.item(), dist_entropy.item()
            
        except Exception as e:
            print(f"\nERROR in batch update: {e}")
            print("Debug info:")
            print(f"  ReplayBuffer size: {self.replay_buffer.size}")
            print(f"  Actor device: {next(self.actor.parameters()).device}")
            print(f"  Critic device: {next(self.critic.parameters()).device}")
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            raise

    def numpy_to_tensor(self):
        """
        将 ReplayBuffer 的数据转换为 update 方法需要的格式
        """
        states, actions, action_log_probs, rewards, next_states, dones = self.replay_buffer.sample()
        
        # 创建 done 和 dw 张量
        done = torch.tensor(dones, dtype=torch.float)
        dw = torch.zeros_like(done)  # 假设没有特殊的死亡或获胜状态
        
        return states, actions, action_log_probs, rewards, next_states, dw, done
    
    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

        print(f"Debug: Learning rate decay:")
        print(f"  Actor: {lr_a_now:.6f}")
        print(f"  Critic: {lr_c_now:.6f}")
        print(f"  Progress: {total_steps}/{self.max_train_steps}")
   


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_index = 1
    main(args, env_name=env_name[env_index], number=1, seed=10)