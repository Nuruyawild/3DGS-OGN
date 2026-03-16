"""
Comprehensive test for all enhanced modules.

Tests:
- 4.1 3D Gaussian Splatting + Scene Graph
- 4.2 Panoptic FPN + CRF + Small Object Detection + Semantic-Geometric Alignment
- 4.3 Enhanced PPO + Multi-reward + Transformer Policy
- 4.5.1-3 Web App + Data Manager
"""

import sys
import os
import time
import traceback
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PASS_COUNT = 0
FAIL_COUNT = 0


def test(name):
    def decorator(func):
        def wrapper():
            global PASS_COUNT, FAIL_COUNT
            try:
                print(f"\n{'='*60}")
                print(f"TEST: {name}")
                print('='*60)
                func()
                PASS_COUNT += 1
                print(f"  => PASS")
            except Exception as e:
                FAIL_COUNT += 1
                print(f"  => FAIL: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ===== 4.1 Tests =====

@test("4.1.1 3DGS - Five-Dimensional Gaussian Parameters")
def test_gaussian_params():
    from models.gaussian_splatting import GaussianSplatting3D, GaussianParameter
    gs = GaussianSplatting3D(num_sem_categories=16, max_gaussians=256,
                             device=DEVICE).to(DEVICE)
    points = torch.randn(200, 3).to(DEVICE)
    colors = torch.rand(200, 3).to(DEVICE)
    labels = torch.randint(0, 15, (200,)).to(DEVICE)
    gaussians = gs.initialize_from_pointcloud(points, colors, labels, n_clusters=32)

    assert gaussians.colors.shape[1] == 3, "Colors should be (N,3)"
    assert gaussians.centers.shape[1] == 3, "Centers should be (N,3)"
    assert gaussians.opacities.shape[1] == 1, "Opacities should be (N,1)"
    assert gaussians.covariances.shape[1] == 3, "Covariances should be (N,3)"
    assert gaussians.semantic_labels.shape[1] == 32, "Semantic embeddings should be (N,32)"
    print(f"  Gaussian count: {gaussians.centers.shape[0]}")


@test("4.1.2 3DGS - K-Means Clustering & Parameter Optimization")
def test_kmeans_and_optimization():
    from models.gaussian_splatting import GaussianSplatting3D
    gs = GaussianSplatting3D(num_sem_categories=16, max_gaussians=128,
                             device=DEVICE).to(DEVICE)
    points = torch.cat([
        torch.randn(50, 3) + torch.tensor([0, 0, 0]),
        torch.randn(50, 3) + torch.tensor([5, 5, 0]),
        torch.randn(50, 3) + torch.tensor([10, 0, 5]),
    ]).to(DEVICE)
    colors = torch.rand(150, 3).to(DEVICE)
    labels = torch.cat([torch.zeros(50), torch.ones(50) * 3,
                         torch.ones(50) * 10]).long().to(DEVICE)

    gaussians = gs.initialize_from_pointcloud(points, colors, labels, n_clusters=16)
    n_gaussians = gaussians.centers.shape[0]
    assert n_gaussians == 16, f"Expected 16 clusters, got {n_gaussians}"

    reg = gs.compute_spatial_regularization()
    assert reg.item() >= 0, "Regularization should be non-negative"

    lr = gs.get_adaptive_learning_rate(0.001, torch.randn(100))
    assert 0 < lr <= 0.001, f"Adaptive LR {lr} out of range"
    print(f"  Clusters: {n_gaussians}, Reg: {reg.item():.4f}, LR: {lr:.6f}")


@test("4.1.3 3DGS - SSIM Loss & Scene Update")
def test_ssim_and_update():
    from models.gaussian_splatting import GaussianSplatting3D
    gs = GaussianSplatting3D(num_sem_categories=16, max_gaussians=256,
                             device=DEVICE).to(DEVICE)

    r1 = torch.rand(3, 32, 32).to(DEVICE)
    r2 = r1.clone()
    loss_same = gs.compute_ssim_loss(r1, r2)
    loss_diff = gs.compute_ssim_loss(r1, torch.rand(3, 32, 32).to(DEVICE))
    assert loss_same < loss_diff, "Same images should have lower SSIM loss"

    points = torch.randn(50, 3).to(DEVICE)
    colors = torch.rand(50, 3).to(DEVICE)
    labels = torch.randint(0, 15, (50,)).to(DEVICE)
    gs.initialize_from_pointcloud(points, colors, labels, n_clusters=10)

    features = gs.get_gaussian_features()
    assert features.shape == (10, 64), f"Expected (10,64), got {features.shape}"
    print(f"  SSIM same: {loss_same.item():.4f}, diff: {loss_diff.item():.4f}")


@test("4.1.4 Scene Graph Construction & Relationships")
def test_scene_graph():
    from models.scene_graph import SceneGraph, SceneGraphBuilder
    sg = SceneGraph()
    sg.add_or_update_node(0, np.array([1.0, 2.0, 0.5]), [], confidence=0.9, step=0)
    sg.add_or_update_node(1, np.array([1.3, 2.1, 0.5]), [], confidence=0.8, step=1)
    sg.add_or_update_node(3, np.array([5.0, 5.0, 0.3]), [], confidence=0.7, step=2)

    assert len(sg.nodes) == 3
    assert len(sg.edges) > 0

    chair_nodes = sg.get_node_by_label(0)
    assert len(chair_nodes) == 1

    neighbors = sg.get_neighbors(0)
    assert len(neighbors) > 0

    feat = sg.get_graph_feature(64)
    assert feat.shape == (64,)

    graph_dict = sg.to_dict()
    assert 'nodes' in graph_dict and 'edges' in graph_dict
    print(f"  Nodes: {len(sg.nodes)}, Edges: {len(sg.edges)}")


@test("4.1.5 Semantic Label Embedding & Fusion")
def test_semantic_embedding():
    from models.semantic_utils import SemanticGeometricAligner
    sga = SemanticGeometricAligner(num_sem_categories=16, embed_dim=32,
                                   device=DEVICE).to(DEVICE)
    labels = torch.randint(0, 16, (50,)).to(DEVICE)
    positions = torch.randn(50, 3).to(DEVICE)
    features = sga.embed_semantics_to_gaussians(labels, positions)
    assert features.shape == (50, 35), f"Expected (50,35), got {features.shape}"
    print(f"  Embedded features shape: {features.shape}")


# ===== 4.2 Tests =====

@test("4.2.1 Panoptic FPN Semantic Prediction")
def test_panoptic_fpn():
    from agents.utils.panoptic_prediction import SemanticPredPanopticFPN
    class Args:
        sem_gpu_id = 0
        sem_pred_prob_thr = 0.9
        visualize = 0
        use_crf = False
    pred = SemanticPredPanopticFPN(Args())
    img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    sem_pred, vis = pred.get_prediction(img)
    assert sem_pred.shape == (120, 160, 16)
    print(f"  Prediction shape: {sem_pred.shape}")


@test("4.2.2 CRF Post-Processing")
def test_crf():
    from models.semantic_utils import CRFPostProcessor
    crf = CRFPostProcessor(num_classes=16, num_iterations=3)
    unary = np.random.randn(16, 32, 32).astype(np.float32)
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = crf.apply(unary, image)
    assert result.shape == (32, 32)
    assert result.max() < 16
    print(f"  CRF output: shape={result.shape}, unique labels={len(np.unique(result))}")


@test("4.2.3 Small Object Detection Optimization")
def test_small_object():
    from models.semantic_utils import SmallObjectDetector
    sod = SmallObjectDetector(in_channels=256, num_classes=16).to(DEVICE)
    fpn = {
        'p2': torch.randn(1, 256, 64, 64).to(DEVICE),
        'p5': torch.randn(1, 256, 8, 8).to(DEVICE),
    }
    pred = sod(fpn)
    assert pred.shape[1] == 5  # 5 small object classes
    assert pred.shape[2] == 64 and pred.shape[3] == 64

    base_pred = torch.randn(1, 16, 64, 64).to(DEVICE)
    enhanced = sod.enhance_semantic_pred(base_pred, pred, threshold=0.5)
    assert enhanced.shape == base_pred.shape
    print(f"  Small object pred: {pred.shape}")


@test("4.2.4 Semantic-Geometric Alignment")
def test_alignment():
    from models.semantic_utils import SemanticGeometricAligner
    sga = SemanticGeometricAligner(num_sem_categories=16, embed_dim=32,
                                   device=DEVICE).to(DEVICE)
    depth = torch.rand(60, 80).to(DEVICE) * 5.0
    sem = torch.randint(0, 16, (60, 80)).to(DEVICE)
    K = torch.tensor([[80., 0, 40], [0, 80., 30], [0, 0, 1]]).to(DEVICE)

    result = sga(sem, depth, K)
    assert result['points'].shape[1] == 3
    assert result['labels'].dim() == 1
    n = result['points'].shape[0]
    print(f"  Aligned {n} points with {result['features'].shape[1]}-dim features")


# ===== 4.3 Tests =====

@test("4.3.1 CNN + Transformer Policy Network")
def test_policy_network():
    from models.gaussian_nav_policy import GaussianSemanticPolicy
    policy = GaussianSemanticPolicy(
        input_shape=(24, 120, 120), recurrent=False, hidden_size=256,
        num_sem_categories=16, gaussian_feature_dim=128,
        use_transformer=True
    ).to(DEVICE)

    obs = torch.randn(2, 24, 120, 120).to(DEVICE)
    rnn = torch.zeros(2, 1).to(DEVICE)
    masks = torch.ones(2).to(DEVICE)
    extras = torch.zeros(2, 2).long().to(DEVICE)
    g_feat = torch.randn(2, 128).to(DEVICE)

    value, features, rnn_out = policy(obs, rnn, masks, extras, g_feat)
    assert value.shape == (2,)
    assert features.shape == (2, 256)
    print(f"  Policy output: value={value.shape}, features={features.shape}")


@test("4.3.2 Multi-Dimensional Reward Function")
def test_reward():
    from models.gaussian_nav_policy import MultiDimensionalReward
    reward_fn = MultiDimensionalReward()

    r1 = reward_fn.compute_reward(
        np.array([1.0, 2.0, 0.0]), 0.0, np.array([5.0, 5.0, 0.0]),
        np.array([[2.0, 2.0, 0.0]]), 0, 5.0)
    assert 'direction' in r1 and 'total' in r1

    r2 = reward_fn.compute_reward(
        np.array([1.5, 2.5, 0.0]), 30.0, np.array([5.0, 5.0, 0.0]),
        np.array([[2.0, 2.0, 0.0]]), 1, 4.5)
    assert r2['exploration'] == 1.0  # new region

    r3 = reward_fn.compute_reward(
        np.array([2.0, 3.0, 0.0]), 30.0, np.array([5.0, 5.0, 0.0]),
        np.array([[2.0, 2.0, 0.0]]), 1, 4.0)
    assert r3['exploration'] == 0.0  # visited region
    reward_fn.reset()
    print(f"  Rewards: dir={r1['direction']:.3f}, col={r1['collision']:.3f}")


@test("4.3.3 Enhanced PPO Training Step")
def test_enhanced_ppo():
    from gym import spaces
    from models.gaussian_nav_policy import GaussianNavPolicy, EnhancedPPO
    from utils.storage import GlobalRolloutStorage

    obs_shape = (24, 120, 120)
    action_space = spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

    policy = GaussianNavPolicy(
        obs_shape, action_space, model_type=1,
        base_kwargs={'recurrent': False, 'hidden_size': 256,
                     'num_sem_categories': 16},
        use_transformer=True).to(DEVICE)

    ppo = EnhancedPPO(
        policy, clip_param=0.2, ppo_epoch=2, num_mini_batch=1,
        value_loss_coef=0.5, entropy_coef=0.001, lr=2.5e-5, eps=1e-5,
        adaptive_lr=True)

    rollouts = GlobalRolloutStorage(
        5, 2, obs_shape, action_space, 1, 2).to(DEVICE)

    for i in range(5):
        obs = torch.randn(2, *obs_shape).to(DEVICE)
        rnn = torch.zeros(2, 1).to(DEVICE)
        masks = torch.ones(2).to(DEVICE)
        extras = torch.zeros(2, 2).long().to(DEVICE)
        with torch.no_grad():
            v, a, lp, rnn = policy.act(obs, rnn, masks, extras=extras)
        rew = torch.randn(2).to(DEVICE)
        rollouts.insert(obs, rnn, a, lp, v, rew, masks, extras)

    rollouts.compute_returns(torch.zeros(2).to(DEVICE), False, 0.99, 0.95)
    vl, al, de = ppo.update(rollouts)
    assert isinstance(vl, float) and isinstance(al, float)
    print(f"  PPO update: vl={vl:.4f}, al={al:.4f}, ent={de:.4f}")


@test("4.3.4 Discrete Action Space")
def test_action_space():
    from models.gaussian_nav_policy import GaussianNavPolicy
    actions = GaussianNavPolicy.DISCRETE_ACTIONS
    assert 0 in actions  # stop
    assert 1 in actions  # forward
    assert 2 in actions  # left
    assert 3 in actions  # right
    for aid, info in actions.items():
        assert 'name' in info and 'move' in info and 'turn' in info
    print(f"  Action space: {len(actions)} actions")


# ===== 4.5 Tests =====

@test("4.5.1 Web App Routes")
def test_web_routes():
    from web_app.app import app
    client = app.test_client()

    resp = client.get('/')
    assert resp.status_code == 200
    assert b'3DGS' in resp.data

    resp = client.get('/api/state')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'agent_position' in data

    resp = client.get('/api/categories')
    assert resp.status_code == 200
    cats = resp.get_json()['categories']
    assert len(cats) == 15

    resp = client.get('/api/datasets')
    assert resp.status_code == 200

    resp = client.get('/api/experiments')
    assert resp.status_code == 200

    resp = client.get('/api/gaussians')
    assert resp.status_code == 200

    resp = client.get('/api/scene_graph')
    assert resp.status_code == 200

    print(f"  All {7} API routes: OK")


@test("4.5.2 State Update & Monitoring")
def test_state_monitoring():
    from web_app.app import app, nav_state
    client = app.test_client()

    resp = client.post('/api/state/update', json={
        'agent_position': [1.0, 2.0, 0.5],
        'agent_heading': 45.0,
        'step': 10,
        'episode': 1,
        'is_running': True,
        'metrics': {'sr': 0.5, 'spl': 0.3, 'dtg': 1.2}
    })
    assert resp.status_code == 200

    resp = client.get('/api/state')
    state = resp.get_json()
    assert state['agent_position'] == [1.0, 2.0, 0.5]
    assert state['step'] == 10
    assert state['is_running'] == True

    resp = client.post('/api/goal/set', json={'category': 'chair'})
    assert resp.status_code == 200
    state = client.get('/api/state').get_json()
    assert state['goal_category'] == 'chair'
    print(f"  State update and monitoring: OK")


@test("4.5.3 Data Management")
def test_data_management():
    from web_app.data_manager import ExperimentManager, DatasetBrowser

    exp_mgr = ExperimentManager()
    experiments = exp_mgr.list_experiments()
    assert isinstance(experiments, list)

    ds = DatasetBrowser()
    datasets = ds.get_available_datasets()
    assert 'gibson' in datasets
    assert len(datasets['gibson']['train']) == 25
    assert len(datasets['gibson']['val']) == 5

    stats = ds.get_category_statistics()
    assert len(stats) == 15
    assert 'chair' in stats

    ep_data = ds.get_episode_data('gibson', 'train')
    assert 'path' in ep_data
    print(f"  Data management: {len(stats)} categories, "
          f"{datasets['gibson']['total_scenes']} scenes")


# ===== Run All Tests =====
if __name__ == '__main__':
    start = time.time()
    print("=" * 60)
    print("COMPREHENSIVE ENHANCED MODULE TEST SUITE")
    print("=" * 60)

    tests = [
        # 4.1
        test_gaussian_params,
        test_kmeans_and_optimization,
        test_ssim_and_update,
        test_scene_graph,
        test_semantic_embedding,
        # 4.2
        test_panoptic_fpn,
        test_crf,
        test_small_object,
        test_alignment,
        # 4.3
        test_policy_network,
        test_reward,
        test_enhanced_ppo,
        test_action_space,
        # 4.5
        test_web_routes,
        test_state_monitoring,
        test_data_management,
    ]

    for t in tests:
        t()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS_COUNT} PASSED, {FAIL_COUNT} FAILED "
          f"({elapsed:.1f}s)")
    print("=" * 60)

    if FAIL_COUNT > 0:
        sys.exit(1)
