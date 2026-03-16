"""
3D Gaussian Splatting (3DGS) with Scene Graph Integration.

Implements the 3D spatial perception framework including:
- Five-dimensional Gaussian parameter set (color, center, opacity, covariance, semantic label)
- K-means clustering for initialization
- SSIM-based loss for real-time 3D reconstruction
- Spatial regularization and adaptive learning rate
- Semantic label embedding and fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class GaussianParameter:
    """Five-dimensional 3D Gaussian parameter set."""
    colors: torch.Tensor          # (N, 3) RGB colors
    centers: torch.Tensor         # (N, 3) xyz center positions
    opacities: torch.Tensor       # (N, 1) transparency/opacity
    covariances: torch.Tensor     # (N, 3) diagonal covariance (xyz spread)
    semantic_labels: torch.Tensor  # (N, D) semantic label embeddings


class GaussianSplatting3D(nn.Module):
    """
    3D Gaussian Splatting module integrated with scene graph.

    Constructs an efficient model representing spatial relationships and
    interactions of objects in the scene through 3D Gaussians with
    semantic embeddings.
    """

    def __init__(self, num_sem_categories=16, semantic_embed_dim=32,
                 max_gaussians=4096, device='cuda'):
        super().__init__()
        self.num_sem_categories = num_sem_categories
        self.semantic_embed_dim = semantic_embed_dim
        self.max_gaussians = max_gaussians
        self.device = device

        self.semantic_embedding = nn.Embedding(
            num_sem_categories + 1, semantic_embed_dim)

        self.color_mlp = nn.Sequential(
            nn.Linear(3 + semantic_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        self.opacity_mlp = nn.Sequential(
            nn.Linear(3 + semantic_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.covariance_mlp = nn.Sequential(
            nn.Linear(3 + semantic_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus()
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 3 + semantic_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self._category_opacity_priors = self._init_category_priors()

        self.gaussians: Optional[GaussianParameter] = None
        self._adaptive_lr_scale = nn.Parameter(torch.ones(1))

    def _init_category_priors(self) -> Dict[str, Dict]:
        """Initialize category-specific priors for Gaussian parameters."""
        return {
            'furniture': {
                'opacity_init': 0.7,
                'covariance_scale': torch.tensor([1.0, 1.0, 0.3]),
            },
            'small_object': {
                'opacity_init': 0.9,
                'covariance_scale': torch.tensor([0.2, 0.2, 0.2]),
            },
            'large_object': {
                'opacity_init': 0.6,
                'covariance_scale': torch.tensor([1.5, 1.5, 1.0]),
            },
            'default': {
                'opacity_init': 0.8,
                'covariance_scale': torch.tensor([0.5, 0.5, 0.5]),
            }
        }

    def _get_object_category_type(self, sem_label: int) -> str:
        """Map semantic label to category type for prior initialization."""
        furniture_ids = {0, 1, 3, 6}   # chair, couch, bed, dining-table
        small_ids = {10, 11, 12, 13, 14}  # book, clock, vase, cup, bottle
        large_ids = {4, 5, 7, 8, 9}     # toilet, tv, oven, sink, refrigerator

        if sem_label in furniture_ids:
            return 'furniture'
        elif sem_label in small_ids:
            return 'small_object'
        elif sem_label in large_ids:
            return 'large_object'
        return 'default'

    def initialize_from_pointcloud(self, points: torch.Tensor,
                                   colors: torch.Tensor,
                                   semantic_labels: torch.Tensor,
                                   n_clusters: Optional[int] = None):
        """
        Initialize Gaussians from point cloud using K-means clustering.

        Args:
            points: (N, 3) point cloud xyz coordinates
            colors: (N, 3) RGB colors
            semantic_labels: (N,) integer semantic labels
            n_clusters: number of Gaussian clusters (default: auto)
        """
        if n_clusters is None:
            n_clusters = min(self.max_gaussians, max(points.shape[0] // 8, 64))

        centers, assignments = self._kmeans_cluster(
            points, n_clusters, max_iter=20)

        init_colors = torch.zeros(n_clusters, 3, device=self.device)
        init_opacities = torch.zeros(n_clusters, 1, device=self.device)
        init_covariances = torch.zeros(n_clusters, 3, device=self.device)
        init_sem_labels = torch.zeros(n_clusters, dtype=torch.long,
                                      device=self.device)

        for i in range(n_clusters):
            mask = assignments == i
            if mask.sum() == 0:
                continue

            init_colors[i] = colors[mask].mean(dim=0)

            cluster_labels = semantic_labels[mask]
            mode_label = cluster_labels.mode().values
            init_sem_labels[i] = mode_label

            cat_type = self._get_object_category_type(mode_label.item())
            priors = self._category_opacity_priors[cat_type]
            init_opacities[i] = priors['opacity_init']

            cluster_points = points[mask]
            if cluster_points.shape[0] > 1:
                std = cluster_points.std(dim=0).clamp(min=0.01)
                init_covariances[i] = std * priors['covariance_scale'].to(
                    self.device)
            else:
                init_covariances[i] = priors['covariance_scale'].to(
                    self.device) * 0.1

        sem_embeds = self.semantic_embedding(init_sem_labels)

        self.gaussians = GaussianParameter(
            colors=nn.Parameter(init_colors),
            centers=nn.Parameter(centers),
            opacities=nn.Parameter(init_opacities),
            covariances=nn.Parameter(init_covariances),
            semantic_labels=sem_embeds.detach()
        )

        return self.gaussians

    def _kmeans_cluster(self, points: torch.Tensor, k: int,
                        max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """K-means clustering on point cloud."""
        n = points.shape[0]
        indices = torch.randperm(n, device=self.device)[:k]
        centers = points[indices].clone()

        for _ in range(max_iter):
            dists = torch.cdist(points, centers)
            assignments = dists.argmin(dim=1)

            new_centers = torch.zeros_like(centers)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centers[i] = points[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]

            if torch.allclose(new_centers, centers, atol=1e-4):
                break
            centers = new_centers

        return centers, assignments

    def render_gaussians(self, camera_pose: torch.Tensor,
                         camera_intrinsics: torch.Tensor,
                         image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Render 3D Gaussians to 2D image using differentiable splatting.

        Args:
            camera_pose: (4, 4) camera extrinsic matrix
            camera_intrinsics: (3, 3) camera intrinsic matrix K
            image_size: (H, W) output image size

        Returns:
            rendered: (3+D, H, W) rendered color + semantic feature map
        """
        if self.gaussians is None:
            return torch.zeros(3 + self.semantic_embed_dim,
                               *image_size, device=self.device)

        H, W = image_size
        centers = self.gaussians.centers
        colors = self.gaussians.colors
        opacities = self.gaussians.opacities
        covs = self.gaussians.covariances
        sem_feats = self.gaussians.semantic_labels

        ones = torch.ones(centers.shape[0], 1, device=self.device)
        centers_h = torch.cat([centers, ones], dim=1)
        cam_points = (camera_pose[:3] @ centers_h.T).T  # (N, 3)

        valid = cam_points[:, 2] > 0.1
        cam_points = cam_points[valid]
        valid_colors = colors[valid]
        valid_opacities = opacities[valid]
        valid_covs = covs[valid]
        valid_sem = sem_feats[valid]

        if cam_points.shape[0] == 0:
            return torch.zeros(3 + self.semantic_embed_dim,
                               *image_size, device=self.device)

        proj = (camera_intrinsics @ cam_points.T).T
        px = (proj[:, 0] / proj[:, 2]).long()
        py = (proj[:, 1] / proj[:, 2]).long()

        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[in_bounds]
        py = py[in_bounds]
        depths = cam_points[in_bounds, 2]
        fb_colors = valid_colors[in_bounds]
        fb_opacities = valid_opacities[in_bounds]
        fb_sem = valid_sem[in_bounds]

        depth_order = depths.argsort()
        px = px[depth_order]
        py = py[depth_order]
        fb_colors = fb_colors[depth_order]
        fb_opacities = fb_opacities[depth_order]
        fb_sem = fb_sem[depth_order]

        feat_dim = 3 + self.semantic_embed_dim
        rendered = torch.zeros(feat_dim, H, W, device=self.device)
        accum_alpha = torch.zeros(1, H, W, device=self.device)

        for i in range(min(px.shape[0], 10000)):
            x, y = int(px[i].item()), int(py[i].item())
            alpha = (fb_opacities[i] * (1 - accum_alpha[0, y, x])).squeeze().item()
            rendered[:3, y, x] += alpha * fb_colors[i]
            rendered[3:, y, x] += alpha * fb_sem[i]
            accum_alpha[0, y, x] += alpha

        return rendered

    def compute_ssim_loss(self, rendered: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """SSIM-based reconstruction loss."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_r = F.avg_pool2d(rendered, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)

        mu_r_sq = mu_r ** 2
        mu_t_sq = mu_t ** 2
        mu_rt = mu_r * mu_t

        sigma_r_sq = F.avg_pool2d(rendered ** 2, 3, 1, 1) - mu_r_sq
        sigma_t_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_t_sq
        sigma_rt = F.avg_pool2d(rendered * target, 3, 1, 1) - mu_rt

        ssim = ((2 * mu_rt + C1) * (2 * sigma_rt + C2)) / \
               ((mu_r_sq + mu_t_sq + C1) * (sigma_r_sq + sigma_t_sq + C2))

        return 1 - ssim.mean()

    def compute_spatial_regularization(self) -> torch.Tensor:
        """Spatial regularization constraining adjacent Gaussian parameter differences."""
        if self.gaussians is None or self.gaussians.centers.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)

        centers = self.gaussians.centers
        dists = torch.cdist(centers, centers)

        k = min(6, centers.shape[0] - 1)
        _, nn_idx = dists.topk(k + 1, largest=False, dim=1)
        nn_idx = nn_idx[:, 1:]  # exclude self

        center_diffs = torch.zeros(centers.shape[0], device=self.device)
        opacity_diffs = torch.zeros(centers.shape[0], device=self.device)
        cov_diffs = torch.zeros(centers.shape[0], device=self.device)

        for i in range(k):
            neighbor_idx = nn_idx[:, i]
            center_diffs += (centers - centers[neighbor_idx]).norm(dim=1)
            opacity_diffs += (self.gaussians.opacities -
                              self.gaussians.opacities[neighbor_idx]).abs().squeeze()
            cov_diffs += (self.gaussians.covariances -
                          self.gaussians.covariances[neighbor_idx]).norm(dim=1)

        reg = (center_diffs.mean() * 0.1 +
               opacity_diffs.mean() * 0.5 +
               cov_diffs.mean() * 0.3)
        return reg

    def get_adaptive_learning_rate(self, base_lr: float,
                                   gradients: Optional[torch.Tensor] = None) -> float:
        """Compute adaptive learning rate based on gradient magnitude."""
        if gradients is None:
            return base_lr

        grad_norm = gradients.norm()
        scale = self._adaptive_lr_scale.item()
        adaptive_lr = base_lr * scale / (1.0 + grad_norm)
        return max(adaptive_lr, base_lr * 0.01)

    def get_gaussian_features(self) -> torch.Tensor:
        """
        Extract feature vectors from all Gaussians for policy input.

        Returns:
            features: (N, feature_dim) Gaussian feature vectors
        """
        if self.gaussians is None:
            return torch.zeros(1, 64, device=self.device)

        combined = torch.cat([
            self.gaussians.colors,
            self.gaussians.centers,
            self.gaussians.opacities,
            self.gaussians.covariances,
            self.gaussians.semantic_labels
        ], dim=1)

        features = self.feature_encoder(combined)
        return features

    def update_from_observation(self, depth: torch.Tensor,
                                semantic_pred: torch.Tensor,
                                rgb: torch.Tensor,
                                camera_pose: torch.Tensor,
                                camera_intrinsics: torch.Tensor):
        """
        Update Gaussian representation from new observation.

        Args:
            depth: (H, W) depth image
            semantic_pred: (H, W) semantic label map
            rgb: (H, W, 3) RGB image
            camera_pose: (4, 4) camera pose
            camera_intrinsics: (3, 3) camera intrinsic matrix
        """
        H, W = depth.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32))

        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        valid_mask = (z > 0.1) & (z < 10.0)
        points_cam = torch.stack([x[valid_mask], y[valid_mask],
                                  z[valid_mask]], dim=1)

        if points_cam.shape[0] == 0:
            return

        ones = torch.ones(points_cam.shape[0], 1, device=self.device)
        points_h = torch.cat([points_cam, ones], dim=1)
        pose_inv = torch.inverse(camera_pose)
        points_world = (pose_inv[:3] @ points_h.T).T

        colors_flat = rgb.reshape(-1, 3)[valid_mask.reshape(-1)]
        labels_flat = semantic_pred.reshape(-1)[valid_mask.reshape(-1)].long()

        if self.gaussians is None:
            self.initialize_from_pointcloud(
                points_world, colors_flat, labels_flat)
        else:
            self._merge_new_observations(
                points_world, colors_flat, labels_flat)

    def _merge_new_observations(self, new_points: torch.Tensor,
                                new_colors: torch.Tensor,
                                new_labels: torch.Tensor):
        """Merge new observations with existing Gaussians."""
        if self.gaussians is None:
            return

        existing_centers = self.gaussians.centers.data
        dists = torch.cdist(new_points, existing_centers)
        min_dists, nearest_idx = dists.min(dim=1)

        merge_threshold = 0.1
        merge_mask = min_dists < merge_threshold

        if merge_mask.sum() > 0:
            for idx in nearest_idx[merge_mask].unique():
                point_mask = (nearest_idx == idx) & merge_mask
                if point_mask.sum() > 0:
                    alpha = 0.3
                    self.gaussians.centers.data[idx] = (
                        (1 - alpha) * self.gaussians.centers.data[idx] +
                        alpha * new_points[point_mask].mean(dim=0))
                    self.gaussians.colors.data[idx] = (
                        (1 - alpha) * self.gaussians.colors.data[idx] +
                        alpha * new_colors[point_mask].mean(dim=0))

        new_mask = ~merge_mask
        if new_mask.sum() > 0 and (existing_centers.shape[0] +
                                    new_mask.sum() <= self.max_gaussians):
            new_sem_embeds = self.semantic_embedding(new_labels[new_mask])

            n_new = new_mask.sum()
            new_opacities = torch.full(
                (n_new, 1), 0.8, device=self.device)
            new_covs = torch.full(
                (n_new, 3), 0.1, device=self.device)

            self.gaussians = GaussianParameter(
                colors=nn.Parameter(torch.cat([
                    self.gaussians.colors.data,
                    new_colors[new_mask]], dim=0)),
                centers=nn.Parameter(torch.cat([
                    self.gaussians.centers.data,
                    new_points[new_mask]], dim=0)),
                opacities=nn.Parameter(torch.cat([
                    self.gaussians.opacities.data,
                    new_opacities], dim=0)),
                covariances=nn.Parameter(torch.cat([
                    self.gaussians.covariances.data,
                    new_covs], dim=0)),
                semantic_labels=torch.cat([
                    self.gaussians.semantic_labels,
                    new_sem_embeds.detach()], dim=0)
            )

    def forward(self, depth: torch.Tensor, semantic_pred: torch.Tensor,
                rgb: torch.Tensor, camera_pose: torch.Tensor,
                camera_intrinsics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: update Gaussians and extract features.

        Returns dict with:
            'gaussian_features': (N, 64) features for policy
            'num_gaussians': int
            'scene_feature': (128,) aggregated scene feature
        """
        self.update_from_observation(
            depth, semantic_pred, rgb, camera_pose, camera_intrinsics)

        features = self.get_gaussian_features()

        scene_feature = torch.cat([
            features.mean(dim=0),
            features.max(dim=0).values
        ])

        return {
            'gaussian_features': features,
            'num_gaussians': features.shape[0],
            'scene_feature': scene_feature,
        }
