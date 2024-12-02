import torch
import torch.nn as nn
import numpy as np

class GaussianSplatting(nn.Module):
    def __init__(self, num_points, device):
        super(GaussianSplatting, self).__init__()
        self.positions = nn.Parameter(torch.randn(num_points, 3, device=device))
        self.colors = nn.Parameter(torch.rand(num_points, 3, device=device))
        self.alphas = nn.Parameter(torch.ones(num_points, device=device))
        
        # 初始化旋转矩阵(R)和缩放矩阵(S)
        self.R = nn.Parameter(torch.eye(3, device=device).unsqueeze(0).repeat(num_points, 1, 1))
        self.S = nn.Parameter(torch.eye(3, device=device).unsqueeze(0).repeat(num_points, 1, 1))

    def compute_spherical_harmonics(self, colors):
        # 计算球谐系数
        # 使用0阶和1阶球谐函数
        batch_size = colors.shape[0]
        l0 = torch.ones_like(colors[:, :, 0:1]) * 0.282095
        l1_1 = colors[:, :, 1:2] * 0.488603
        l1_0 = colors[:, :, 0:1] * 0.488603
        l1_1_neg = colors[:, :, 2:3] * 0.488603
        
        sh_coeffs = torch.cat([l0, l1_1, l1_0, l1_1_neg], dim=2)
        return sh_coeffs

    def forward(self):
        # 计算协方差矩阵
        S_squared = torch.bmm(self.S, self.S.transpose(1, 2))
        covariance_matrices = torch.bmm(self.R, torch.bmm(S_squared, self.R.transpose(1, 2)))

        # 计算球谐系数
        spherical_harmonics = self.compute_spherical_harmonics(self.colors)

        return self.positions, covariance_matrices, self.alphas, spherical_harmonics