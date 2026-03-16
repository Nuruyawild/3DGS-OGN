"""
论文图表自动生成模块

生成论文所需的图像文件：
- 图5-1: 典型场景下与基线方法的定性对比（轨迹示意）
- 图5-2: 低保真场景问题与替换策略示意（渲染质量对比）
- 图5-3: 典型失败案例与改进前后对比
- 图5-4: Web 可视化系统界面示意（由 web_app 截图功能生成）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, List, Dict, Tuple, Any


# 高分辨率输出设置（至少 300 DPI）
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'


def _ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _get_semantic_map_rgb(sem_map: np.ndarray,
                          obstacle_ch: int = 0,
                          explored_ch: int = 1,
                          num_sem_categories: int = 16) -> np.ndarray:
    """
    将语义地图转换为 RGB 可视化。
    sem_map: (C, H, W) 或 (H, W)
    """
    if sem_map.ndim == 3:
        obs_map = sem_map[obstacle_ch]
        exp_map = sem_map[explored_ch]
        sem_channels = sem_map[4:] if sem_map.shape[0] > 4 else None
    else:
        obs_map = np.zeros_like(sem_map)
        exp_map = np.zeros_like(sem_map)
        sem_channels = None

    # 创建 RGB 底图：灰色为未探索，白色为已探索无障碍
    h, w = obs_map.shape
    rgb = np.ones((h, w, 3)) * 0.9  # 浅灰背景
    rgb[exp_map > 0.5] = [1.0, 1.0, 1.0]  # 已探索区域白色
    rgb[obs_map > 0.5] = [0.3, 0.3, 0.3]  # 障碍物深灰

    return rgb


def save_trajectory_comparison(
    sem_map: np.ndarray,
    trajectory_ours: np.ndarray,
    trajectory_baseline: Optional[np.ndarray] = None,
    start_pos: Tuple[float, float] = (0, 0),
    goal_pos: Optional[Tuple[float, float]] = None,
    object_pos: Optional[Tuple[float, float]] = None,
    map_resolution: float = 5.0,
    map_size_cm: int = 2400,
    output_path: str = "fig_5_1_trajectory_comparison_scene1.png",
    output_dir: str = "./tmp/paper_figures",
    scene_label: str = "Scene 1",
    success: bool = True,
) -> str:
    """
    图5-1: 典型场景下与基线方法的定性对比（轨迹示意）

    在 2D 语义地图上叠加显示：本文方法轨迹、基线方法轨迹、起点、终点、目标物体位置。

    Args:
        sem_map: 语义地图 (C, H, W) 或 (H, W)
        trajectory_ours: 本文方法轨迹 (N, 2) 或 (N, 3)，单位为 map cells
        trajectory_baseline: 基线方法轨迹，可选
        start_pos: 起点 (r, c) 或 (x, y) in map cells
        goal_pos: 终点/目标位置
        object_pos: 目标物体位置（若与 goal_pos 不同）
        map_resolution: 地图分辨率 cm/cell
        map_size_cm: 地图尺寸 cm
        output_path: 输出文件名
        output_dir: 输出目录
        scene_label: 场景标签
        success: 是否成功案例
    """
    out_dir = _ensure_output_dir(output_dir)
    full_path = os.path.join(out_dir, output_path)

    map_size = map_size_cm // int(map_resolution)
    if sem_map.ndim == 3:
        h, w = sem_map.shape[1], sem_map.shape[2]
        obs_map = sem_map[0]
        exp_map = sem_map[1]
    else:
        h, w = sem_map.shape[0], sem_map.shape[1]
        obs_map = np.zeros((h, w))
        exp_map = np.ones((h, w))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=FIGURE_DPI)

    # 绘制语义地图底图
    rgb = np.ones((h, w, 3)) * 0.85
    rgb[exp_map > 0.1] = [1.0, 1.0, 0.95]
    rgb[obs_map > 0.5] = [0.4, 0.4, 0.45]
    ax.imshow(rgb, origin='lower', extent=[0, w, 0, h])

    # 绘制轨迹
    if trajectory_ours is not None and len(trajectory_ours) >= 2:
        traj = np.array(trajectory_ours)
        if traj.shape[1] >= 2:
            ax.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=2.5,
                    label='Ours', alpha=0.9)
            ax.plot(traj[-1, 1], traj[-1, 0], 'b*', markersize=14)

    if trajectory_baseline is not None and len(trajectory_baseline) >= 2:
        traj_b = np.array(trajectory_baseline)
        if traj_b.shape[1] >= 2:
            ax.plot(traj_b[:, 1], traj_b[:, 0], 'r--', linewidth=2,
                    label='Baseline', alpha=0.8)
            ax.plot(traj_b[-1, 1], traj_b[-1, 0], 'rs', markersize=10)

    # 起点
    if start_pos is not None:
        sr, sc = start_pos[0], start_pos[1]
        ax.plot(sc, sr, 'go', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)

    # 终点/目标
    pos_to_plot = goal_pos if goal_pos is not None else object_pos
    if pos_to_plot is not None:
        gr, gc = pos_to_plot[0], pos_to_plot[1]
        ax.plot(gc, gr, 'm^', markersize=12, label='Goal', markeredgecolor='darkviolet', markeredgewidth=2)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'{scene_label} ({ "Success" if success else "Failure" })', fontsize=12)
    ax.set_xlabel('Map X (cells)')
    ax.set_ylabel('Map Y (cells)')
    plt.tight_layout()
    plt.savefig(full_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='white', format=FIGURE_FORMAT)
    plt.close()
    return full_path


def save_rendering_quality_comparison(
    rgb_high: np.ndarray,
    rgb_low: np.ndarray,
    depth_high: Optional[np.ndarray] = None,
    depth_low: Optional[np.ndarray] = None,
    sem_high: Optional[np.ndarray] = None,
    sem_low: Optional[np.ndarray] = None,
    output_path: str = "fig_5_3_rendering_quality_comparison.png",
    output_dir: str = "./tmp/paper_figures",
    scene_labels: Optional[List[str]] = None,
) -> str:
    """
    图5-2: 低保真场景问题与替换策略示意

    高质量渲染（左）vs 低质量渲染（右），每个场景可包含 RGB、深度、语义分割。
    """
    out_dir = _ensure_output_dir(output_dir)
    full_path = os.path.join(out_dir, output_path)

    n_scenes = 1
    if isinstance(rgb_high, list):
        n_scenes = len(rgb_high)
        rgb_h_list = rgb_high
        rgb_l_list = rgb_low
    else:
        rgb_h_list = [rgb_high]
        rgb_l_list = [rgb_low]

    if scene_labels is None:
        scene_labels = [f"Scene {i+1}" for i in range(n_scenes)]

    n_rows = n_scenes * 3 if (depth_high is not None or sem_high is not None) else n_scenes
    n_cols = 2  # 高质量 | 低质量

    fig, axes = plt.subplots(n_scenes * 3 if (depth_high is not None or sem_high is not None) else n_scenes,
                             n_cols, figsize=(10, 4 * n_scenes), dpi=FIGURE_DPI)
    if n_scenes == 1 and (depth_high is not None or sem_high is not None):
        axes = axes.reshape(-1, 2)
    elif n_scenes == 1:
        axes = np.array([axes])

    for s in range(n_scenes):
        rh = rgb_h_list[s] if isinstance(rgb_h_list[s], np.ndarray) else np.array(rgb_h_list[s])
        rl = rgb_l_list[s] if isinstance(rgb_l_list[s], np.ndarray) else np.array(rgb_l_list[s])
        if rh.ndim == 3 and rh.shape[2] == 3:
            pass
        elif rh.ndim == 3 and rh.shape[0] == 3:
            rh = np.transpose(rh, (1, 2, 0))
            rl = np.transpose(rl, (1, 2, 0)) if rl.ndim == 3 else rl
        if np.max(rh) <= 1.0:
            rh = (rh * 255).astype(np.uint8)
        if np.max(rl) <= 1.0:
            rl = (rl * 255).astype(np.uint8)

        base_row = s * 3
        axes[base_row, 0].imshow(rh)
        axes[base_row, 0].set_title(f'{scene_labels[s]} - High Quality')
        axes[base_row, 0].axis('off')
        axes[base_row, 1].imshow(rl)
        axes[base_row, 1].set_title(f'{scene_labels[s]} - Low Quality')
        axes[base_row, 1].axis('off')

        if depth_high is not None and base_row + 1 < axes.shape[0]:
            dh = depth_high[s] if isinstance(depth_high, list) else depth_high
            dl = depth_low[s] if isinstance(depth_low, list) else depth_low
            axes[base_row + 1, 0].imshow(dh, cmap='viridis')
            axes[base_row + 1, 0].set_title('Depth (High)')
            axes[base_row + 1, 0].axis('off')
            axes[base_row + 1, 1].imshow(dl, cmap='viridis')
            axes[base_row + 1, 1].set_title('Depth (Low)')
            axes[base_row + 1, 1].axis('off')

        if sem_high is not None and base_row + 2 < axes.shape[0]:
            sh = sem_high[s] if isinstance(sem_high, list) else sem_high
            sl = sem_low[s] if isinstance(sem_low, list) else sem_low
            # 使用离散 colormap 显示语义标签
            n_classes = max(16, int(np.max(sh)) + 1, int(np.max(sl)) + 1)
            try:
                cmap = plt.cm.get_cmap('tab20', n_classes)
            except Exception:
                cmap = plt.cm.viridis
            axes[base_row + 2, 0].imshow(sh, cmap=cmap, vmin=0, vmax=n_classes - 1)
            axes[base_row + 2, 0].set_title('Semantic (High)')
            axes[base_row + 2, 0].axis('off')
            axes[base_row + 2, 1].imshow(sl, cmap=cmap, vmin=0, vmax=n_classes - 1)
            axes[base_row + 2, 1].set_title('Semantic (Low)')
            axes[base_row + 2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(full_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='white', format=FIGURE_FORMAT)
    plt.close()
    return full_path


def save_failure_case_comparison(
    sem_map: np.ndarray,
    trajectory_before: np.ndarray,
    trajectory_after: np.ndarray,
    start_pos: Tuple[float, float],
    goal_pos: Optional[Tuple[float, float]] = None,
    failure_annotations: Optional[List[Dict[str, Any]]] = None,
    map_resolution: float = 5.0,
    map_size_cm: int = 2400,
    output_path: str = "fig_5_4_failure_case_comparison.png",
    output_dir: str = "./tmp/paper_figures",
) -> str:
    """
    图5-3: 典型失败案例与改进前后对比

    显示改进前的轨迹（失败）和改进后的轨迹（成功），标注失败原因。
    """
    out_dir = _ensure_output_dir(output_dir)
    full_path = os.path.join(out_dir, output_path)

    if sem_map.ndim == 3:
        h, w = sem_map.shape[1], sem_map.shape[2]
        obs_map = sem_map[0]
        exp_map = sem_map[1]
    else:
        h, w = sem_map.shape[0], sem_map.shape[1]
        obs_map = np.zeros((h, w))
        exp_map = np.ones((h, w))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=FIGURE_DPI)

    for idx, (ax, traj, title) in enumerate([
        (axes[0], trajectory_before, 'Before (Failure)'),
        (axes[1], trajectory_after, 'After (Success)'),
    ]):
        rgb = np.ones((h, w, 3)) * 0.85
        rgb[exp_map > 0.1] = [1.0, 1.0, 0.95]
        rgb[obs_map > 0.5] = [0.4, 0.4, 0.45]
        ax.imshow(rgb, origin='lower', extent=[0, w, 0, h])

        if traj is not None and len(traj) >= 2:
            t = np.array(traj)
            if t.shape[1] >= 2:
                ax.plot(t[:, 1], t[:, 0], 'b-' if idx == 1 else 'r-',
                        linewidth=2, alpha=0.9)
                ax.plot(t[-1, 1], t[-1, 0], 'b*' if idx == 1 else 'rx',
                        markersize=12)

        if start_pos is not None:
            ax.plot(start_pos[1], start_pos[0], 'go', markersize=10,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        if goal_pos is not None:
            ax.plot(goal_pos[1], goal_pos[0], 'm^', markersize=10,
                    markeredgecolor='darkviolet', markeredgewidth=2)

        if failure_annotations and idx == 0:
            for ann in failure_annotations:
                r, c = ann.get('position', (0, 0))
                text = ann.get('label', 'Failure')
                ax.annotate(text, (c, r), fontsize=8, color='red',
                            xytext=(5, 5), textcoords='offset points')

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Map X (cells)')
        ax.set_ylabel('Map Y (cells)')

    plt.suptitle('Failure Case: Before vs After Improvement', fontsize=13)
    plt.tight_layout()
    plt.savefig(full_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='white', format=FIGURE_FORMAT)
    plt.close()
    return full_path


def pose_to_map_cells(pose: np.ndarray, map_resolution: float = 5.0,
                      map_size_cm: int = 2400) -> Tuple[int, int]:
    """
    将连续位姿 (x, y, o) 转换为地图单元坐标 (r, c)。
    pose 单位为米，原点在地图中心。
    """
    map_size = map_size_cm // int(map_resolution)
    center = map_size / 2.0
    # x -> col, y -> row
    c = int(pose[0] * 100.0 / map_resolution + center)
    r = int(pose[1] * 100.0 / map_resolution + center)
    r = max(0, min(map_size - 1, r))
    c = max(0, min(map_size - 1, c))
    return r, c
