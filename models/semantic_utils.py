"""
Semantic Processing Utilities for Object Goal Navigation.

Implements:
- CRF post-processing for boundary optimization
- Small object detection with feature pyramid fusion
- Semantic-geometric alignment for 3D coordinate computation
- Semantic label embedding into 3D Gaussians
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class CRFPostProcessor:
    """
    Conditional Random Field post-processing for semantic segmentation.

    Optimizes boundary pixel classification using unary potentials from
    the semantic model and pairwise potentials based on color/spatial proximity.
    """

    def __init__(self, num_classes=16, num_iterations=5,
                 spatial_sigma=3.0, bilateral_sigma_xy=60.0,
                 bilateral_sigma_rgb=10.0):
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.spatial_sigma = spatial_sigma
        self.bilateral_sigma_xy = bilateral_sigma_xy
        self.bilateral_sigma_rgb = bilateral_sigma_rgb

    def apply(self, unary: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Apply CRF post-processing to refine segmentation boundaries.

        Args:
            unary: (C, H, W) unary potentials (log probabilities per class)
            image: (H, W, 3) RGB image

        Returns:
            refined: (H, W) refined semantic label map
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            return self._apply_pydensecrf(unary, image)
        except ImportError:
            return self._apply_simple_crf(unary, image)

    def _apply_pydensecrf(self, unary: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Full CRF using pydensecrf library."""
        import pydensecrf.densecrf as dcrf

        C, H, W = unary.shape
        d = dcrf.DenseCRF2D(W, H, C)

        softmax = self._softmax(unary)
        U = -np.log(softmax + 1e-6).reshape(C, -1).astype(np.float32)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(
            sxy=self.spatial_sigma, compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)

        d.addPairwiseBilateral(
            sxy=self.bilateral_sigma_xy,
            srgb=self.bilateral_sigma_rgb,
            rgbim=image.astype(np.uint8),
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(self.num_iterations)
        result = np.argmax(Q, axis=0).reshape(H, W)
        return result

    def _apply_simple_crf(self, unary: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Simplified mean-field CRF approximation without pydensecrf.
        Uses spatial Gaussian filtering for message passing.
        """
        C, H, W = unary.shape
        Q = self._softmax(unary)

        for _ in range(self.num_iterations):
            spatial_msg = np.zeros_like(Q)
            for c in range(C):
                spatial_msg[c] = self._gaussian_filter(
                    Q[c], sigma=self.spatial_sigma)

            bilateral_msg = np.zeros_like(Q)
            for c in range(C):
                bilateral_msg[c] = self._bilateral_filter(
                    Q[c], image, self.bilateral_sigma_xy,
                    self.bilateral_sigma_rgb)

            msg = 3.0 * spatial_msg + 10.0 * bilateral_msg

            Q_new = unary - msg
            Q = self._softmax(Q_new)

        return Q.argmax(axis=0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax along first axis."""
        x_safe = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
        x_shifted = x_safe - x_safe.max(axis=0, keepdims=True)
        e_x = np.exp(np.clip(x_shifted, -50, 50))
        return e_x / (e_x.sum(axis=0, keepdims=True) + 1e-8)

    @staticmethod
    def _gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
        """Simple Gaussian spatial filter."""
        k = int(2 * sigma + 1)
        if k % 2 == 0:
            k += 1
        x = np.arange(k) - k // 2
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        result = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'), 0, img)
        result = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'), 1, result)
        return result

    @staticmethod
    def _bilateral_filter(img: np.ndarray, guide: np.ndarray,
                          sigma_xy: float, sigma_rgb: float) -> np.ndarray:
        """Simplified bilateral filter using downsampled computation."""
        H, W = img.shape
        ds = max(1, min(H, W) // 32)
        small_img = img[::ds, ::ds]
        small_guide = guide[::ds, ::ds]

        result = np.zeros_like(small_img)
        h, w = small_img.shape
        r = min(3, h // 2, w // 2)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                shifted_img = np.roll(np.roll(small_img, dy, 0), dx, 1)
                shifted_guide = np.roll(np.roll(small_guide, dy, 0), dx, 1)

                spatial_w = np.exp(-(dx ** 2 + dy ** 2) / (2 * (sigma_xy / ds) ** 2))
                color_diff = np.sum((small_guide.astype(np.float64) - shifted_guide.astype(np.float64)) ** 2, axis=-1)
                range_w = np.exp(np.clip(-color_diff / (2 * sigma_rgb ** 2), -50, 0))

                weight = spatial_w * range_w
                result += weight * shifted_img

        from PIL import Image as PILImage
        result_pil = PILImage.fromarray(result.astype(np.float32))
        result_full = np.array(result_pil.resize((W, H), PILImage.BILINEAR))
        return result_full


class SmallObjectDetector(nn.Module):
    """
    Small Object Detection Optimizer.

    Fuses top-level and bottom-level FPN features to improve
    recall for small indoor objects (cups, books, etc.).
    """

    SMALL_OBJECT_IDS = {10, 11, 12, 13, 14}  # book, clock, vase, cup, bottle

    def __init__(self, in_channels=256, num_classes=16):
        super().__init__()
        self.num_classes = num_classes

        self.top_down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.bottom_up_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.small_object_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, len(self.SMALL_OBJECT_IDS), 1),
        )

    def forward(self, fpn_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse FPN features for improved small object detection.

        Args:
            fpn_features: dict with keys like 'p2', 'p3', 'p4', 'p5'
                         containing feature maps at different scales

        Returns:
            enhanced_pred: (B, num_small_classes, H, W) predictions for small objects
        """
        keys = sorted(fpn_features.keys())
        if len(keys) < 2:
            feat = list(fpn_features.values())[0]
            return self.small_object_head(feat)

        bottom_feat = fpn_features[keys[0]]
        top_feat = fpn_features[keys[-1]]

        target_size = bottom_feat.shape[2:]
        top_upsampled = F.interpolate(
            top_feat, size=target_size, mode='bilinear', align_corners=False)

        bottom_processed = self.bottom_up_conv(bottom_feat)
        top_processed = self.top_down_conv(top_upsampled)

        fused = torch.cat([bottom_processed, top_processed], dim=1)
        fused = self.fusion_conv(fused)

        small_pred = self.small_object_head(fused)
        return small_pred

    def enhance_semantic_pred(self, base_pred: torch.Tensor,
                              small_pred: torch.Tensor,
                              threshold: float = 0.5) -> torch.Tensor:
        """
        Enhance base semantic prediction with small object detections.

        Args:
            base_pred: (B, C, H, W) base semantic predictions
            small_pred: (B, num_small, H, W) small object predictions
            threshold: confidence threshold

        Returns:
            enhanced: (B, C, H, W) enhanced semantic predictions
        """
        enhanced = base_pred.clone()
        small_probs = torch.sigmoid(small_pred)

        small_ids = sorted(self.SMALL_OBJECT_IDS)
        for i, cat_id in enumerate(small_ids):
            if cat_id < enhanced.shape[1]:
                high_conf = small_probs[:, i] > threshold
                enhanced[:, cat_id][high_conf] = torch.max(
                    enhanced[:, cat_id][high_conf],
                    small_probs[:, i][high_conf])

        return enhanced


class SemanticGeometricAligner(nn.Module):
    """
    Semantic-Geometric Alignment Module.

    Precisely aligns semantic labels with 3D spatial coordinates using
    camera intrinsic matrix K, and embeds semantic labels into 3D Gaussians.
    """

    def __init__(self, num_sem_categories=16, embed_dim=32, device='cuda'):
        super().__init__()
        self.num_sem_categories = num_sem_categories
        self.embed_dim = embed_dim
        self.device = device

        self.semantic_embedding = nn.Embedding(
            num_sem_categories + 1, embed_dim)

        self.alignment_mlp = nn.Sequential(
            nn.Linear(3 + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 + embed_dim),
        )

    def compute_3d_coordinates(self, depth: torch.Tensor,
                               intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D coordinates from depth using camera intrinsic matrix K.

        Args:
            depth: (H, W) depth image in meters
            intrinsics: (3, 3) camera intrinsic matrix K

        Returns:
            points: (H, W, 3) 3D coordinates in camera frame
        """
        H, W = depth.shape
        device = depth.device

        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32))

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return torch.stack([x, y, z], dim=-1)

    def align_semantic_to_3d(self, semantic_map: torch.Tensor,
                             depth: torch.Tensor,
                             intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align semantic labels with 3D coordinates.

        Args:
            semantic_map: (H, W) integer semantic labels or (C, H, W) probabilities
            depth: (H, W) depth image
            intrinsics: (3, 3) camera intrinsics

        Returns:
            aligned_points: (N, 3) valid 3D points
            aligned_labels: (N,) corresponding semantic labels
        """
        points_3d = self.compute_3d_coordinates(depth, intrinsics)

        if semantic_map.dim() == 3:
            labels = semantic_map.argmax(dim=0)
        else:
            labels = semantic_map.long()

        valid_mask = (depth > 0.1) & (depth < 10.0)
        aligned_points = points_3d[valid_mask]
        aligned_labels = labels[valid_mask]

        return aligned_points, aligned_labels

    def embed_semantics_to_gaussians(self, semantic_labels: torch.Tensor,
                                     positions: torch.Tensor) -> torch.Tensor:
        """
        Embed semantic labels into continuous feature space for Gaussian fusion.

        Args:
            semantic_labels: (N,) integer semantic labels
            positions: (N, 3) 3D positions

        Returns:
            fused_features: (N, 3 + embed_dim) position + semantic features
        """
        sem_embeds = self.semantic_embedding(
            semantic_labels.clamp(0, self.num_sem_categories))

        combined = torch.cat([positions, sem_embeds], dim=1)
        refined = self.alignment_mlp(combined)

        return refined

    def forward(self, semantic_map: torch.Tensor, depth: torch.Tensor,
                intrinsics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full semantic-geometric alignment pipeline.

        Returns dict with aligned points, labels, and fused features.
        """
        points, labels = self.align_semantic_to_3d(
            semantic_map, depth, intrinsics)

        if points.shape[0] == 0:
            return {
                'points': torch.zeros(0, 3, device=self.device),
                'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                'features': torch.zeros(0, 3 + self.embed_dim, device=self.device)
            }

        features = self.embed_semantics_to_gaussians(labels, points)

        return {
            'points': points,
            'labels': labels,
            'features': features
        }
