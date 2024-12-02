# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import torch.nn.functional as F
import numpy as np
from utils.distributions import Categorical, DiagGaussian


class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        # 初始化监控指标的最佳值和计数器
        self.best_loss = float('inf')
        self.best_spl = float('-inf') 
        self.best_success = float('-inf')
        
        self.loss_counter = 0
        self.spl_counter = 0
        self.success_counter = 0
        
    def step(self, metrics):
        current_lr = self.optimizer.param_groups[0]['lr']
        should_reduce = False
        
        # 监控 loss
        if 'loss' in metrics:
            if metrics['loss'] >= self.best_loss:
                self.loss_counter += 1
                if self.loss_counter >= self.patience:
                    should_reduce = True
                    self.loss_counter = 0
            else:
                self.best_loss = metrics['loss']
                self.loss_counter = 0
                
        # 监控 SPL
        if 'spl' in metrics:
            if metrics['spl'] <= self.best_spl:
                self.spl_counter += 1
                if self.spl_counter >= self.patience:
                    should_reduce = True
                    self.spl_counter = 0
            else:
                self.best_spl = metrics['spl']
                self.spl_counter = 0
                
        # 监控 success rate
        if 'success' in metrics:
            if metrics['success'] <= self.best_success:
                self.success_counter += 1
                if self.success_counter >= self.patience:
                    should_reduce = True
                    self.success_counter = 0
            else:
                self.best_success = metrics['success']
                self.success_counter = 0
                
        # 如果需要降低学习率
        if should_reduce:
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr != current_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f'Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}')
            
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')


class PPO:
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch,
                 value_loss_coef, entropy_coef, lr=None, eps=None,
                 max_grad_norm=None, use_clipped_value_loss=True):
        
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        # 添加学习率调度器
        self.scheduler = AdaptiveLRScheduler(
            self.optimizer,
            patience=5,
            min_lr=1e-6,
            factor=0.5
        )


    def update(self, rollouts, metrics=None):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:

                # 确保 sample 中的所有张量都在正确的设备上
                device = self.actor_critic.device if hasattr(self.actor_critic, 'device') else next(self.actor_critic.parameters()).device
                for k in sample:
                    if isinstance(sample[k], torch.Tensor):
                        sample[k] = sample[k].to(device)

                # 获取动作
                action = sample['actions']
                
                # 如果动作维度不匹配，调整形状
                if action.dim() == 1 and isinstance(self.actor_critic.action_space, gym.spaces.Box):
                    action = action.unsqueeze(-1)
                    sample['actions'] = action

                # 如果 rec_states 不存在，使用全零张量
                if 'rec_states' not in sample:
                    sample['rec_states'] = torch.zeros_like(sample['masks'])


                value_preds = sample['value_preds']
                returns = sample['returns']
                adv_targ = sample['adv_targ']

                

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], action,
                        extras=sample['extras']
                    )

                # 计算策略损失（action loss）
                
                ratio = torch.exp(action_log_probs - sample['old_action_log_probs'])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(
                            -self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

                # 在更新后调整学习率
        if metrics is not None:
            current_metrics = {
                'loss': value_loss_epoch + action_loss_epoch,
                'spl': metrics.get('spl', 0),
                'success': metrics.get('success', 0)
            }
            self.scheduler.step(current_metrics)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class PPOWithAttention(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size, num_attention_layers):
        super(PPOWithAttention, self).__init__()
        self.action_space = action_space  

        # 初始化网络组件
        self.input_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(observation_space.shape), hidden_size)
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=0.1
        )
        
        # 其他网络层
        self.actor_linear = nn.Linear(hidden_size, hidden_size)
        self.critic_linear = nn.Linear(hidden_size, 1)
        
        # 动作分布，根据动作空间类型选择合适的分布
        if isinstance(action_space, gym.spaces.Discrete):
            self.dist = Categorical(hidden_size, action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            self.dist = DiagGaussian(hidden_size, action_space.shape[0])
        else:
            raise NotImplementedError

        # 是否使用循环网络
        self.is_recurrent = False
        self.rec_state_size = 0

    def forward(self, inputs, rnn_hxs=None, masks=None, extras=None):
        # 投影输入
        x = self.input_projection(inputs)
        
        # 添加注意力机制
        x = x.unsqueeze(0)  # 增加序列维度
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x.squeeze(0)
        
        # 处理特征
        actor_features = F.relu(self.actor_linear(x))
        value = self.critic_linear(actor_features)
        
        return value, actor_features, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):
        value, actor_features, _ = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if action.dim() == 1:
            action = action.unsqueeze(-1)

        return value, action_log_probs, dist_entropy, rnn_hxs
