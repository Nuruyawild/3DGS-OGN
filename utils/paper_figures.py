"""
Paper figure utilities for Object-Goal Navigation.
Saves trajectory comparison and failure case visualization.
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from constants import color_palette


def _sem_map_to_rgb(sem_map, map_resolution=5):
    """Convert semantic map (C, H, W) to RGB visualization."""
    if sem_map.ndim == 2:
        sem_map = sem_map[np.newaxis, :, :]
    h, w = sem_map.shape[-2], sem_map.shape[-1]
    # Occupancy: 0=free, 1=obstacle, 2=explored
    if sem_map.shape[0] >= 4:
        occ = sem_map[0]  # obstacle
        exp = sem_map[1]  # explored
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[exp > 0.1] = [200, 200, 200]
        rgb[occ > 0.1] = [80, 80, 80]
        # Semantic: channels 4+ are categories
        if sem_map.shape[0] > 4:
            sem_idx = sem_map[4:].argmax(0)
            for i in range(min(sem_idx.max() + 1, 16)):
                mask = (sem_idx == i) & (exp > 0.1)
                if mask.any() and i * 3 + 2 < len(color_palette):
                    rgb[mask] = np.array(
                        [color_palette[i*3]*255, color_palette[i*3+1]*255,
                         color_palette[i*3+2]*255], dtype=np.uint8)
    else:
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
    return rgb


def save_trajectory_comparison(sem_map, trajectory_ours, trajectory_baseline,
                               start_pos, goal_pos, map_resolution, map_size_cm,
                               output_path, output_dir, scene_label, success=True):
    """Save trajectory comparison figure."""
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_path)

    if sem_map.ndim == 3:
        img = _sem_map_to_rgb(sem_map, map_resolution)
    else:
        img = np.ones((*sem_map.shape[-2:], 3), dtype=np.uint8) * 200

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, origin='upper')

    # Trajectory
    if trajectory_ours is not None and len(trajectory_ours) >= 2:
        traj = np.array(trajectory_ours)
        ax.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=2, label='Ours')
        ax.plot(traj[0, 1], traj[0, 0], 'go', markersize=10, label='Start')
        ax.plot(traj[-1, 1], traj[-1, 0], 'r*' if success else 'rx',
                markersize=12, label='End')

    if trajectory_baseline is not None and len(trajectory_baseline) >= 2:
        traj_b = np.array(trajectory_baseline)
        ax.plot(traj_b[:, 1], traj_b[:, 0], 'c--', linewidth=1.5, label='Baseline')

    if start_pos is not None:
        ax.plot(start_pos[1], start_pos[0], 'gs', markersize=8)
    if goal_pos is not None:
        ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=12)

    ax.set_title(scene_label)
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_failure_case_comparison(sem_map, trajectory_before, trajectory_after,
                                 start_pos, goal_pos, failure_annotations,
                                 map_resolution, map_size_cm,
                                 output_path, output_dir):
    """Save failure case before/after comparison figure."""
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_path)

    if sem_map.ndim == 3:
        img = _sem_map_to_rgb(sem_map, map_resolution)
    else:
        img = np.ones((*sem_map.shape[-2:], 3), dtype=np.uint8) * 200

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, origin='upper')

    # Before trajectory
    if trajectory_before is not None and len(trajectory_before) >= 2:
        traj = np.array(trajectory_before)
        ax.plot(traj[:, 1], traj[:, 0], 'r-', linewidth=2, label='Before (failure)')

    # After trajectory
    if trajectory_after is not None and len(trajectory_after) >= 2:
        traj = np.array(trajectory_after)
        ax.plot(traj[:, 1], traj[:, 0], 'g-', linewidth=2, label='After (improved)')

    if start_pos is not None:
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
    if goal_pos is not None:
        ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=12, label='Goal')

    for ann in (failure_annotations or []):
        pos = ann.get('position')
        lbl = ann.get('label', '')
        if pos is not None:
            ax.plot(pos[1], pos[0], 'kx', markersize=8)
            ax.annotate(lbl, (pos[1], pos[0]), fontsize=8)

    ax.set_title('Failure Case Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
