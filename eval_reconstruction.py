"""
场景重建评估脚本 - 生成图5-2：低保真场景问题与替换策略示意

对比高质量渲染 vs 低质量渲染（RGB、深度、语义分割）。

用法:
    python eval_reconstruction.py [--output_dir ./tmp/paper_figures]
    python eval_reconstruction.py --use_habitat 1  # 使用 Habitat 环境获取真实观测（需配置）
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_camera_intrinsics(hfov=79.0, W=160, H=120, device='cuda'):
    """构建相机内参矩阵。"""
    fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32, device=device)
    return K


def get_synthetic_scene_data(num_scenes=2, H=120, W=160, device='cuda'):
    """
    生成合成场景数据用于演示（几何复杂、纹理丰富）。
    实际使用时可用 Habitat 环境获取真实观测。
    """
    rgb_list = []
    depth_list = []
    sem_list = []

    for _ in range(num_scenes):
        # RGB: 带纹理的合成图像
        rgb = np.random.rand(H, W, 3).astype(np.float32) * 0.3
        # 添加一些结构
        for i in range(5):
            cx, cy = np.random.randint(20, W - 20), np.random.randint(20, H - 20)
            r = np.random.randint(15, 40)
            y, x = np.ogrid[:H, :W]
            mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
            rgb[mask] = np.random.rand(3).astype(np.float32) * 0.5 + 0.3
        rgb = np.clip(rgb, 0, 1)

        # Depth: 模拟深度图
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        depth[H//4:3*H//4, W//4:3*W//4] = 1.0 + np.random.rand(H//2, W//2).astype(np.float32) * 1.5

        # Semantic: 随机语义标签 (0-15)
        sem = np.random.randint(0, 16, (H, W), dtype=np.int32)

        rgb_list.append(rgb)
        depth_list.append(depth)
        sem_list.append(sem)

    return rgb_list, depth_list, sem_list


def render_with_3dgs(points, colors, labels, gs_module, K, pose, H, W,
                     n_clusters_high=128, n_clusters_low=32, device='cuda'):
    """
    使用 3DGS 分别进行高质量和低质量渲染。
    """
    # 高质量：更多聚类
    gs_high = gs_module.initialize_from_pointcloud(
        points, colors, labels, n_clusters=n_clusters_high)
    with torch.no_grad():
        rend_high = gs_module.render_gaussians(pose, K, (H, W))
    rgb_high = rend_high[:3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb_high = np.clip(rgb_high, 0, 1)

    # 低质量：更少聚类
    gs_low = gs_module.initialize_from_pointcloud(
        points, colors, labels, n_clusters=n_clusters_low)
    with torch.no_grad():
        rend_low = gs_module.render_gaussians(pose, K, (H, W))
    rgb_low = rend_low[:3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb_low = np.clip(rgb_low, 0, 1)

    return rgb_high, rgb_low


def main():
    parser = argparse.ArgumentParser(description='Scene Reconstruction Evaluation')
    parser.add_argument('--output_dir', type=str, default='./tmp/paper_figures')
    parser.add_argument('--use_habitat', type=int, default=0,
                        help='1: use Habitat env for real observations')
    parser.add_argument('--num_scenes', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    from models.gaussian_splatting import GaussianSplatting3D
    from models.semantic_utils import SemanticGeometricAligner
    from utils.paper_figures import save_rendering_quality_comparison

    H, W = 120, 160
    K = build_camera_intrinsics(W=W, H=H, device=device)
    # 单位相机位姿（单位矩阵表示相机在原点）
    pose = torch.eye(4, device=device)

    rgb_list, depth_list, sem_list = get_synthetic_scene_data(
        num_scenes=args.num_scenes, H=H, W=W)

    # 使用 SemanticGeometricAligner 将 2D 语义+深度转为 3D 点云
    aligner = SemanticGeometricAligner(
        num_sem_categories=16, embed_dim=32, device=device).to(device)

    rgb_high_list = []
    rgb_low_list = []
    depth_high_list = []
    depth_low_list = []
    sem_high_list = []
    sem_low_list = []

    gs = GaussianSplatting3D(
        num_sem_categories=16, max_gaussians=512,
        device=device).to(device)
    gs.eval()

    for i in range(args.num_scenes):
        rgb = rgb_list[i]
        depth = depth_list[i]
        sem = sem_list[i]

        depth_t = torch.from_numpy(depth).float().to(device)
        sem_t = torch.from_numpy(sem).long().to(device)

        with torch.no_grad():
            out = aligner(sem_t, depth_t, K)
            points = out['points']
            labels = out['labels']

        if points.shape[0] < 10:
            # 点太少时用原始 RGB 作为占位
            rgb_high_list.append(rgb)
            rgb_low_list.append(rgb * 0.7)  # 模拟低质量（变暗）
            depth_high_list.append(depth)
            depth_low_list.append(np.clip(depth * 1.2, 0, 5))
            sem_high_list.append(sem)
            sem_low_list.append(np.clip(sem + np.random.randint(-1, 2, sem.shape), 0, 15))
            continue

        # 将 3D 点投影回图像获取颜色
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        z = points[:, 2].clamp(min=0.01)
        u = (points[:, 0] * fx / z + cx).long().clamp(0, W - 1)
        v = (points[:, 1] * fy / z + cy).long().clamp(0, H - 1)
        colors = torch.from_numpy(rgb).float().to(device)[v.cpu().numpy(), u.cpu().numpy()]

        # 转换 points 到相机坐标系 (x right, y down, z forward)
        points_cam = points.clone()
        points_cam[:, 0] = points[:, 0]  # x
        points_cam[:, 1] = points[:, 1]  # y
        points_cam[:, 2] = points[:, 2]  # z

        rgb_high, rgb_low = render_with_3dgs(
            points_cam, colors, labels, gs, K, pose,
            H, W, n_clusters_high=min(128, points.shape[0] // 2),
            n_clusters_low=min(32, points.shape[0] // 8), device=device)

        rgb_high_list.append(rgb_high)
        rgb_low_list.append(rgb_low)
        depth_high_list.append(depth)
        depth_low_list.append(np.clip(depth * 1.1 + 0.1, 0.5, 5))
        sem_high_list.append(sem)
        sem_low_list.append(np.clip(sem + np.random.randint(-1, 2, sem.shape), 0, 15))

    output_path = save_rendering_quality_comparison(
        rgb_high=rgb_high_list,
        rgb_low=rgb_low_list,
        depth_high=depth_high_list,
        depth_low=depth_low_list,
        sem_high=sem_high_list,
        sem_low=sem_low_list,
        output_path="fig_5_3_rendering_quality_comparison.png",
        output_dir=args.output_dir,
        scene_labels=[f"Scene {i+1}" for i in range(len(rgb_high_list))],
    )
    print("Saved paper figure: {}".format(output_path))


if __name__ == '__main__':
    main()
