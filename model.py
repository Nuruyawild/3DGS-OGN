import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
from algo.gaussian_splatting import GaussianSplatting


class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

        # 添加自注意力层
        self.attention = nn.MultiheadAttention(hidden_size, 
                                             num_heads=8, 
                                             dropout=0.1)
        
        # 添加LayerNorm和Dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, rnn_hxs, masks, extras):
        inputs = inputs.unsqueeze(2)
        x = self.main(inputs)
        
        
        # 确保 x 和线性层的权重在同一个设备上
        device = x.device
        linear_layer = nn.Linear(x.size(-1), 256).to(device)
        x = linear_layer(x)

        # 添加注意力机制
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        
        device = self.orientation_emb.weight.device
        orientation_emb = self.orientation_emb(extras[:, 0].to(device).long())
        device = self.goal_emb.weight.device
        goal_emb = self.goal_emb(extras[:, 1].to(device).long())

        # 确保 x 的形状与 self.linear1 的输入维度匹配
        x = torch.cat((x.squeeze(0), orientation_emb, goal_emb), 1)


        # 调整 self.linear1 的输入维度
        if x.size(1) != self.linear1.in_features:
            self.linear1 = nn.Linear(x.size(1), self.linear1.out_features).to(device)

        x = self.dropout(nn.ReLU()(self.linear1(x)))
        x = self.layer_norm2(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, observation_space, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}


        # 添加一个线性层来调整维度
        self.input_size = observation_space[0] * observation_space[1] * observation_space[2]
        self.embedding_size = base_kwargs.get('hidden_size', 256)
        
        self.input_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.embedding_size)
        )
        
        

        if model_type == 1:
            # 其他网络组件保持不变
            self.network = Goal_Oriented_Semantic_Policy(
                observation_space,
                **base_kwargs
            )
        else:
            raise NotImplementedError

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
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        # 在传入attention之前调整维度
        projected_inputs = self.input_projection(inputs)
        projected_inputs = projected_inputs.view(-1, 1, self.embedding_size)  # 调整形状以适应attention
        
        return self.network(projected_inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        # 确保返回的 action 形状为 [batch_size, action_dim]
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if action.dim() == 1:
            action = action.unsqueeze(-1)

        return value, action_log_probs, dist_entropy, rnn_hxs


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        self.gaussian_splatting = GaussianSplatting(
            num_points=self.vision_range * self.vision_range,  # 根据视野范围动态设置点数
            device=self.device
        )


    def forward(self, obs, pose_obs, maps_last, poses_last):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range

        # 在GaussianSplatting之前调整XYZ_cm_std的维度
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        # 调整XYZ_cm_std的形状以匹配特征维度
        XYZ_cm_std = XYZ_cm_std.reshape(bs, -1, 3)  # [12, 19200, 3]
        XYZ_cm_std = XYZ_cm_std.permute(0, 2, 1)    # [12, 3, 19200]

        # 初始化GaussianSplatting的位置
        self.gaussian_splatting.positions.data = XYZ_cm_std.reshape(bs, -1, 3)

        # 使用深度信息和语义特征计算颜色
        colors = torch.cat([
            self.feat[:, 1:4, :],  # 使用RGB特征
            depth.view(bs, 1, -1)  # 添加深度信息
        ], dim=1)
        self.gaussian_splatting.colors.data = colors.transpose(1, 2)  # (batch_size, num_points, 4)

        # 在GaussianSplatting之前调整特征
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # 运行GaussianSplatting
        positions, covariances, alphas, spherical_harmonics = self.gaussian_splatting()
        
        # 使用球谐系数增强特征
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # 调整球谐系数的维度以匹配特征
        sh_mean = spherical_harmonics.mean(dim=2)  # [12, 19200] <- [12, 19200, 4]
        sh_mean = sh_mean.unsqueeze(1)  # [12, 1, 19200]
        sh_mean = sh_mean.expand(-1, self.feat.size(1), -1)  # [12, 17, 19200]

        # 使用球谐系数增强特征
        enhanced_features = self.feat * (1 + sh_mean)  # [12, 17, 19200]

        # 调整维度
        h_scaled = h // self.du_scale
        w_scaled = w // self.du_scale

        # 调整XYZ_cm_std的形状 [12, 3, 19200] -> [12, 3, h_scaled, w_scaled]
        XYZ_cm_std = XYZ_cm_std.view(bs, 3, h_scaled, w_scaled)

        # 确保坐标在[-1, 1]范围内
        XYZ_cm_std = torch.clamp(XYZ_cm_std, min=-1.0, max=1.0)

        # 调整enhanced_features的形状 [12, 17, 19200] -> [12, 17, h_scaled, w_scaled]
        enhanced_features = enhanced_features.view(bs, -1, h_scaled, w_scaled)

        # 调用splat_feat_nd前，将XYZ_cm_std和enhanced_features展平
        XYZ_cm_std = XYZ_cm_std.view(bs, 3, -1)
        enhanced_features = enhanced_features.view(bs, -1, h_scaled * w_scaled)
    
        # XYZ_cm_std = agent_view_centered_t.float()
        # XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        # XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
        #                        vision_range // 2.) / vision_range * 2.
        # XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        # XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
        #                       (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        # self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
        #     obs[:, 4:, :, :]
        # ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        # XYZ_cm_std = XYZ_cm_std.unsqueeze(1)

        # XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        # XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
        #                              XYZ_cm_std.shape[1],
        #                              XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        # 调用splat_feat_nd
        voxels = du.splat_feat_nd(
            self.init_grid * 0., 
            enhanced_features,  # 使用增强后的特征
            XYZ_cm_std
        ).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses