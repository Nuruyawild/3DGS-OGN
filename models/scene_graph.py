"""
Scene Graph Construction and Management.

Builds and maintains a scene graph representing spatial relationships
between objects detected in the environment, integrated with 3D Gaussian
representations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SceneNode:
    """A node in the scene graph representing a detected object."""
    node_id: int
    semantic_label: int
    label_name: str
    position: np.ndarray        # (3,) xyz world coordinates
    gaussian_indices: List[int]  # indices into Gaussian parameter set
    confidence: float = 1.0
    bbox_3d: Optional[np.ndarray] = None  # (6,) xmin,ymin,zmin,xmax,ymax,zmax
    last_seen_step: int = 0


@dataclass
class SceneEdge:
    """An edge representing spatial relationship between two nodes."""
    source_id: int
    target_id: int
    relation_type: str  # 'near', 'on_top', 'beside', 'inside', 'facing'
    distance: float
    direction: np.ndarray  # (3,) normalized direction vector


class SceneGraph:
    """
    Scene Graph for representing object relationships.

    Maintains a graph of detected objects and their spatial relationships,
    integrated with 3D Gaussian representations for precise localization.
    """

    RELATION_THRESHOLDS = {
        'near': 2.0,       # meters
        'on_top': 0.5,
        'beside': 1.5,
        'inside': 0.3,
    }

    CATEGORY_NAMES = {
        0: "chair", 1: "couch", 2: "potted plant", 3: "bed",
        4: "toilet", 5: "tv", 6: "dining-table", 7: "oven",
        8: "sink", 9: "refrigerator", 10: "book", 11: "clock",
        12: "vase", 13: "cup", 14: "bottle"
    }

    def __init__(self):
        self.nodes: Dict[int, SceneNode] = {}
        self.edges: List[SceneEdge] = []
        self._next_node_id = 0

    def add_or_update_node(self, semantic_label: int, position: np.ndarray,
                           gaussian_indices: List[int],
                           confidence: float = 1.0,
                           step: int = 0) -> int:
        """Add a new node or update existing one if close enough."""
        for nid, node in self.nodes.items():
            if (node.semantic_label == semantic_label and
                    np.linalg.norm(node.position - position) < 0.5):
                alpha = 0.3
                node.position = (1 - alpha) * node.position + alpha * position
                node.gaussian_indices = list(set(
                    node.gaussian_indices + gaussian_indices))
                node.confidence = max(node.confidence, confidence)
                node.last_seen_step = step
                return nid

        node_id = self._next_node_id
        self._next_node_id += 1

        label_name = self.CATEGORY_NAMES.get(semantic_label, f"unknown_{semantic_label}")

        self.nodes[node_id] = SceneNode(
            node_id=node_id,
            semantic_label=semantic_label,
            label_name=label_name,
            position=position,
            gaussian_indices=gaussian_indices,
            confidence=confidence,
            last_seen_step=step
        )

        self._update_edges_for_node(node_id)
        return node_id

    def _update_edges_for_node(self, node_id: int):
        """Update edges involving the given node."""
        self.edges = [e for e in self.edges
                      if e.source_id != node_id and e.target_id != node_id]

        node = self.nodes[node_id]
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue

            dist = np.linalg.norm(node.position - other_node.position)
            direction = (other_node.position - node.position)
            if dist > 0:
                direction = direction / dist

            relations = self._infer_relations(node, other_node, dist, direction)
            for rel_type in relations:
                self.edges.append(SceneEdge(
                    source_id=node_id,
                    target_id=other_id,
                    relation_type=rel_type,
                    distance=dist,
                    direction=direction
                ))

    def _infer_relations(self, node: SceneNode, other: SceneNode,
                         dist: float, direction: np.ndarray) -> List[str]:
        """Infer spatial relations between two nodes."""
        relations = []

        if dist < self.RELATION_THRESHOLDS['near']:
            relations.append('near')

        if dist < self.RELATION_THRESHOLDS['beside']:
            height_diff = abs(node.position[2] - other.position[2])
            horizontal_dist = np.linalg.norm(
                node.position[:2] - other.position[:2])

            if height_diff < 0.3 and horizontal_dist < 1.5:
                relations.append('beside')

            if (node.position[2] > other.position[2] + 0.2 and
                    horizontal_dist < 0.5):
                relations.append('on_top')

        return relations if relations else ['near'] if dist < 3.0 else []

    def get_node_by_label(self, semantic_label: int) -> List[SceneNode]:
        """Get all nodes with a specific semantic label."""
        return [n for n in self.nodes.values()
                if n.semantic_label == semantic_label]

    def get_neighbors(self, node_id: int,
                      relation_type: Optional[str] = None) -> List[Tuple[int, SceneEdge]]:
        """Get neighboring nodes with optional relation type filter."""
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append((edge.target_id, edge))
            elif edge.target_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append((edge.source_id, edge))
        return neighbors

    def get_graph_feature(self, embed_dim: int = 64) -> np.ndarray:
        """
        Compute a fixed-size feature vector summarizing the scene graph.

        Returns:
            feature: (embed_dim,) numpy array
        """
        if len(self.nodes) == 0:
            return np.zeros(embed_dim)

        positions = np.array([n.position for n in self.nodes.values()])
        labels = np.array([n.semantic_label for n in self.nodes.values()])
        confidences = np.array([n.confidence for n in self.nodes.values()])

        pos_feat = np.concatenate([
            positions.mean(axis=0),
            positions.std(axis=0) if len(positions) > 1 else np.zeros(3),
            positions.min(axis=0),
            positions.max(axis=0)
        ])  # 12 dims

        label_hist = np.zeros(16)
        for l in labels:
            if 0 <= l < 16:
                label_hist[l] += 1
        if label_hist.sum() > 0:
            label_hist /= label_hist.sum()

        edge_feat = np.zeros(8)
        if len(self.edges) > 0:
            dists = np.array([e.distance for e in self.edges])
            edge_feat[0] = len(self.edges)
            edge_feat[1] = dists.mean()
            edge_feat[2] = dists.std() if len(dists) > 1 else 0
            rel_types = [e.relation_type for e in self.edges]
            edge_feat[3] = rel_types.count('near')
            edge_feat[4] = rel_types.count('on_top')
            edge_feat[5] = rel_types.count('beside')
            edge_feat[6] = len(self.nodes)
            edge_feat[7] = confidences.mean()

        raw = np.concatenate([pos_feat, label_hist, edge_feat])  # 36 dims

        if len(raw) < embed_dim:
            feature = np.zeros(embed_dim)
            feature[:len(raw)] = raw
        else:
            feature = raw[:embed_dim]

        return feature

    def to_dict(self) -> Dict:
        """Serialize scene graph to dictionary for visualization."""
        nodes_list = []
        for nid, node in self.nodes.items():
            nodes_list.append({
                'id': nid,
                'label': node.label_name,
                'semantic_label': node.semantic_label,
                'position': node.position.tolist(),
                'confidence': node.confidence,
                'last_seen_step': node.last_seen_step
            })

        edges_list = []
        for edge in self.edges:
            edges_list.append({
                'source': edge.source_id,
                'target': edge.target_id,
                'relation': edge.relation_type,
                'distance': edge.distance,
                'direction': edge.direction.tolist()
            })

        return {'nodes': nodes_list, 'edges': edges_list}


class SceneGraphBuilder(nn.Module):
    """
    Neural module for building scene graphs from semantic observations.
    Integrates with 3D Gaussian Splatting for precise localization.
    """

    def __init__(self, num_sem_categories=16, device='cuda'):
        super().__init__()
        self.num_sem_categories = num_sem_categories
        self.device = device
        self.scene_graph = SceneGraph()

        self.relation_classifier = nn.Sequential(
            nn.Linear(6 + 2 * num_sem_categories, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        self.relation_types = ['near', 'on_top', 'beside', 'inside', 'facing']

    def update_from_semantic_map(self, semantic_map: torch.Tensor,
                                 depth_map: torch.Tensor,
                                 camera_pose: torch.Tensor,
                                 camera_intrinsics: torch.Tensor,
                                 step: int = 0):
        """
        Update scene graph from semantic and depth observations.

        Args:
            semantic_map: (H, W) or (C, H, W) semantic predictions
            depth_map: (H, W) depth image
            camera_pose: (4, 4) camera pose
            camera_intrinsics: (3, 3) camera intrinsics
            step: current timestep
        """
        if semantic_map.dim() == 3:
            sem_labels = semantic_map.argmax(dim=0)
        else:
            sem_labels = semantic_map.long()

        H, W = sem_labels.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32))

        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        for cat_id in range(self.num_sem_categories):
            mask = (sem_labels == cat_id) & (z > 0.1) & (z < 10.0)
            if mask.sum() < 10:
                continue

            points_cam = torch.stack([
                x[mask], y[mask], z[mask]], dim=1)

            ones = torch.ones(points_cam.shape[0], 1, device=self.device)
            points_h = torch.cat([points_cam, ones], dim=1)
            pose_inv = torch.inverse(camera_pose)
            points_world = (pose_inv[:3] @ points_h.T).T

            centroid = points_world.mean(dim=0).cpu().numpy()

            self.scene_graph.add_or_update_node(
                semantic_label=cat_id,
                position=centroid,
                gaussian_indices=[],
                confidence=mask.float().sum().item() / (H * W),
                step=step
            )

    def get_scene_graph(self) -> SceneGraph:
        return self.scene_graph

    def reset(self):
        self.scene_graph = SceneGraph()

    def forward(self, semantic_map, depth_map, camera_pose,
                camera_intrinsics, step=0):
        self.update_from_semantic_map(
            semantic_map, depth_map, camera_pose, camera_intrinsics, step)
        return self.scene_graph.get_graph_feature()
