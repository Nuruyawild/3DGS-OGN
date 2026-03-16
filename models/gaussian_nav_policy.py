"""
Enhanced Navigation Policy with 3D Gaussian Features.

Implements:
- CNN + Transformer encoder architecture
- Policy and value networks sharing feature extraction
- 3D semantic Gaussian feature vectors as state space input
- Multi-dimensional reward function
- Discrete action space for navigation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from utils.distributions import Categorical, DiagGaussian
from utils.model import NNBase, Flatten


class TransformerEncoder(nn.Module):
    """Transformer encoder for capturing global semantic relationships."""

    def __init__(self, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pos_encoding(x)
        return self.encoder(x, src_key_padding_mask=mask)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GaussianSemanticPolicy(NNBase):
    """
    Policy network using CNN + Transformer for 3D Gaussian semantic features.

    Takes 3D semantic Gaussian feature vectors as state space input,
    uses CNN for local feature extraction and Transformer for global
    semantic relationship capture. Policy and value networks share
    the feature extraction layers.
    """

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16, gaussian_feature_dim=128,
                 use_transformer=True):
        super(GaussianSemanticPolicy, self).__init__(
            recurrent, hidden_size, hidden_size)

        self.use_transformer = use_transformer
        self.gaussian_feature_dim = gaussian_feature_dim
        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        self.cnn = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        cnn_out_dim = out_size * 32

        self.gaussian_encoder = nn.Sequential(
            nn.Linear(gaussian_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        if use_transformer:
            self.transformer = TransformerEncoder(
                d_model=128, nhead=4, num_layers=2,
                dim_feedforward=256, dropout=0.1)
            self.cnn_to_transformer = nn.Linear(cnn_out_dim, 128)
            self.gaussian_to_transformer = nn.Linear(64, 128)
            self.transformer_out = nn.Linear(128, 256)
            feature_dim = 256 + 8 * 2
        else:
            feature_dim = cnn_out_dim + 64 + 8 * 2

        self.linear1 = nn.Linear(feature_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)

        self.critic_linear = nn.Linear(256, 1)

        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras,
                gaussian_features=None):
        cnn_feat = self.cnn(inputs)

        if gaussian_features is not None:
            g_feat = self.gaussian_encoder(gaussian_features)
        else:
            g_feat = torch.zeros(
                inputs.shape[0], 64, device=inputs.device)

        if self.use_transformer:
            cnn_tokens = self.cnn_to_transformer(cnn_feat).unsqueeze(1)
            g_tokens = self.gaussian_to_transformer(g_feat).unsqueeze(1)
            seq = torch.cat([cnn_tokens, g_tokens], dim=1)
            transformed = self.transformer(seq)
            combined = self.transformer_out(transformed.mean(dim=1))
        else:
            combined = torch.cat([cnn_feat, g_feat], dim=1)

        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])
        x = torch.cat((combined, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


class GaussianNavPolicy(nn.Module):
    """
    Enhanced RL Policy using 3D Gaussian semantic features.

    Wraps GaussianSemanticPolicy with discrete action space for navigation.
    Supports both continuous (Box) and discrete action spaces.
    """

    DISCRETE_ACTIONS = {
        0: {'name': 'stop', 'move': 0.0, 'turn': 0.0},
        1: {'name': 'forward', 'move': 0.25, 'turn': 0.0},
        2: {'name': 'turn_left', 'move': 0.0, 'turn': 30.0},
        3: {'name': 'turn_right', 'move': 0.0, 'turn': -30.0},
        4: {'name': 'forward_left', 'move': 0.25, 'turn': 15.0},
        5: {'name': 'forward_right', 'move': 0.25, 'turn': -15.0},
    }

    def __init__(self, obs_shape, action_space, model_type=1,
                 base_kwargs=None, gaussian_feature_dim=128,
                 use_transformer=True):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}

        base_kwargs['gaussian_feature_dim'] = gaussian_feature_dim
        base_kwargs['use_transformer'] = use_transformer

        self.network = GaussianSemanticPolicy(obs_shape, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras,
                gaussian_features=None):
        return self.network(inputs, rnn_hxs, masks, extras,
                            gaussian_features)

    def act(self, inputs, rnn_hxs, masks, extras=None,
            deterministic=False, gaussian_features=None):
        value, actor_features, rnn_hxs = self(
            inputs, rnn_hxs, masks, extras, gaussian_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None,
                  gaussian_features=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras,
                           gaussian_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,
                         extras=None, gaussian_features=None):
        value, actor_features, rnn_hxs = self(
            inputs, rnn_hxs, masks, extras, gaussian_features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class MultiDimensionalReward:
    """
    Multi-dimensional reward function for navigation.

    Components:
    - Target direction reward: alignment with goal semantic Gaussian
    - Path efficiency reward: step smoothness
    - Collision avoidance penalty: distance threshold to obstacle Gaussians
    - Exploration reward: incentive for visiting new semantic regions
    """

    def __init__(self, direction_weight=1.0, efficiency_weight=0.5,
                 collision_weight=2.0, exploration_weight=0.3,
                 collision_threshold=0.3):
        self.direction_weight = direction_weight
        self.efficiency_weight = efficiency_weight
        self.collision_weight = collision_weight
        self.exploration_weight = exploration_weight
        self.collision_threshold = collision_threshold

        self.visited_regions = set()
        self.path_history = []

    def compute_reward(self, agent_pos: np.ndarray,
                       agent_heading: float,
                       goal_pos: np.ndarray,
                       obstacle_positions: np.ndarray,
                       semantic_region_id: int,
                       prev_dist_to_goal: float) -> Dict[str, float]:
        """
        Compute multi-dimensional reward.

        Args:
            agent_pos: (3,) current agent position
            agent_heading: current heading in degrees
            goal_pos: (3,) goal position
            obstacle_positions: (M, 3) obstacle Gaussian positions
            semantic_region_id: current semantic region identifier
            prev_dist_to_goal: previous distance to goal

        Returns:
            rewards: dict with component rewards and total
        """
        dir_reward = self._direction_reward(
            agent_pos, agent_heading, goal_pos, prev_dist_to_goal)

        eff_reward = self._efficiency_reward(agent_pos)

        col_penalty = self._collision_penalty(agent_pos, obstacle_positions)

        exp_reward = self._exploration_reward(semantic_region_id)

        total = (self.direction_weight * dir_reward +
                 self.efficiency_weight * eff_reward +
                 self.collision_weight * col_penalty +
                 self.exploration_weight * exp_reward)

        self.path_history.append(agent_pos.copy())

        return {
            'direction': dir_reward,
            'efficiency': eff_reward,
            'collision': col_penalty,
            'exploration': exp_reward,
            'total': total
        }

    def _direction_reward(self, agent_pos: np.ndarray, heading: float,
                          goal_pos: np.ndarray,
                          prev_dist: float) -> float:
        """Reward for moving toward the goal semantic Gaussian."""
        curr_dist = np.linalg.norm(agent_pos[:2] - goal_pos[:2])
        dist_reward = prev_dist - curr_dist

        to_goal = goal_pos[:2] - agent_pos[:2]
        goal_angle = np.degrees(np.arctan2(to_goal[1], to_goal[0]))
        angle_diff = abs(((heading - goal_angle + 180) % 360) - 180)
        angle_reward = 1.0 - angle_diff / 180.0

        return dist_reward * 2.0 + angle_reward * 0.5

    def _efficiency_reward(self, agent_pos: np.ndarray) -> float:
        """Reward for path smoothness and efficiency."""
        if len(self.path_history) < 2:
            return 0.0

        prev_pos = self.path_history[-1]
        step_length = np.linalg.norm(agent_pos[:2] - prev_pos[:2])

        smoothness = 0.0
        if len(self.path_history) >= 3:
            p1 = self.path_history[-2][:2]
            p2 = self.path_history[-1][:2]
            p3 = agent_pos[:2]
            v1 = p2 - p1
            v2 = p3 - p2
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0.01 and n2 > 0.01:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                smoothness = cos_angle

        return step_length * 0.5 + smoothness * 0.5

    def _collision_penalty(self, agent_pos: np.ndarray,
                           obstacle_positions: np.ndarray) -> float:
        """Penalty for proximity to obstacle Gaussians."""
        if obstacle_positions is None or len(obstacle_positions) == 0:
            return 0.0

        dists = np.linalg.norm(
            obstacle_positions[:, :2] - agent_pos[:2], axis=1)
        min_dist = dists.min()

        if min_dist < self.collision_threshold:
            return -(1.0 - min_dist / self.collision_threshold)
        return 0.0

    def _exploration_reward(self, region_id: int) -> float:
        """Reward for visiting new semantic regions."""
        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            return 1.0
        return 0.0

    def reset(self):
        """Reset for new episode."""
        self.visited_regions.clear()
        self.path_history.clear()


class EnhancedPPO:
    """
    Enhanced PPO with multi-dimensional reward support.

    Extends standard PPO with:
    - Support for Gaussian feature inputs
    - Adaptive learning rate based on gradient magnitudes
    - Multi-reward component tracking
    """

    def __init__(self, actor_critic, clip_param, ppo_epoch,
                 num_mini_batch, value_loss_coef, entropy_coef,
                 lr=2.5e-5, eps=1e-5, max_grad_norm=0.5,
                 adaptive_lr=True):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.adaptive_lr = adaptive_lr

        self.base_lr = lr
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   actor_critic.parameters()),
            lr=lr, eps=eps)

        self.reward_stats = {
            'direction': [], 'efficiency': [],
            'collision': [], 'exploration': []
        }

    def update(self, rollouts):
        """Update policy using PPO with optional adaptive learning rate."""
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
                values, action_log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], sample['actions'],
                        extras=sample['extras'])

                ratio = torch.exp(
                    action_log_probs - sample['old_action_log_probs'])
                surr1 = ratio * sample['adv_targ']
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param,
                    1.0 + self.clip_param) * sample['adv_targ']
                action_loss = -torch.min(surr1, surr2).mean()

                value_preds = sample['value_preds']
                returns = sample['returns']
                value_pred_clipped = value_preds + \
                    (values - value_preds).clamp(
                        -self.clip_param, self.clip_param)
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(
                    value_losses, value_losses_clipped).mean()

                self.optimizer.zero_grad()
                total_loss = (value_loss * self.value_loss_coef +
                              action_loss -
                              dist_entropy * self.entropy_coef)
                total_loss.backward()

                if self.adaptive_lr:
                    self._adjust_learning_rate()

                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def _adjust_learning_rate(self):
        """Adjust learning rate based on gradient magnitude."""
        total_norm = 0.0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        scale = 1.0 / (1.0 + total_norm * 0.01)
        new_lr = self.base_lr * max(scale, 0.1)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
