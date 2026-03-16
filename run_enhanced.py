"""
Enhanced Object-Goal Navigation with 3D Gaussian Splatting.

Integrates:
- 3DGS spatial perception
- Scene graph construction
- Panoptic FPN + CRF semantic prediction
- CNN + Transformer policy with multi-dimensional rewards
- Enhanced PPO training

Usage:
    python run_enhanced.py                         # Train enhanced model
    python run_enhanced.py --eval 1 --load <path>  # Evaluate
    python run_enhanced.py --use_3dgs 0            # Disable 3DGS (fallback to base)
"""

from collections import deque, defaultdict
import os
import sys
import logging
import time
import json
import gym
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RL_Policy, Semantic_Mapping
from models.gaussian_splatting import GaussianSplatting3D
from models.scene_graph import SceneGraphBuilder
from models.gaussian_nav_policy import (
    GaussianNavPolicy, MultiDimensionalReward, EnhancedPPO
)
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo

os.environ.setdefault("OMP_NUM_THREADS", "1")


def add_enhanced_args(args):
    """Add enhanced model arguments to existing args."""
    if not hasattr(args, 'use_3dgs'):
        args.use_3dgs = True
    if not hasattr(args, 'use_scene_graph'):
        args.use_scene_graph = True
    if not hasattr(args, 'use_transformer'):
        args.use_transformer = True
    if not hasattr(args, 'use_panoptic'):
        args.use_panoptic = False
    if not hasattr(args, 'use_crf'):
        args.use_crf = True
    if not hasattr(args, 'use_enhanced_reward'):
        args.use_enhanced_reward = True
    if not hasattr(args, 'gaussian_feature_dim'):
        args.gaussian_feature_dim = 128
    if not hasattr(args, 'max_gaussians'):
        args.max_gaussians = 2048
    if not hasattr(args, 'direction_reward_weight'):
        args.direction_reward_weight = 1.0
    if not hasattr(args, 'efficiency_reward_weight'):
        args.efficiency_reward_weight = 0.5
    if not hasattr(args, 'collision_reward_weight'):
        args.collision_reward_weight = 2.0
    if not hasattr(args, 'exploration_reward_weight'):
        args.exploration_reward_weight = 0.3
    if not hasattr(args, 'adaptive_lr'):
        args.adaptive_lr = True
    if not hasattr(args, 'use_enhanced_policy'):
        args.use_enhanced_policy = True
    return args


def build_camera_intrinsics(args, device):
    """Build camera intrinsic matrix K from args."""
    fov = args.hfov
    W, H = args.frame_width, args.frame_height
    fx = W / (2.0 * np.tan(np.radians(fov / 2.0)))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32, device=device)
    return K


def main():
    args = get_args()
    args = add_enhanced_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(dump_dir, exist_ok=True)

    logging.basicConfig(filename=log_dir + 'train.log', level=logging.INFO)
    print("Enhanced Object-Goal Navigation")
    print("3DGS: {}, SceneGraph: {}, Transformer: {}, EnhancedReward: {}".format(
        args.use_3dgs, args.use_scene_graph, args.use_transformer,
        args.use_enhanced_reward))
    print("Dumping at {}".format(log_dir))
    logging.info(str(args))

    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)
    best_g_reward = -np.inf

    if args.eval:
        episode_success = [deque(maxlen=num_episodes) for _ in range(num_scenes)]
        episode_spl = [deque(maxlen=num_episodes) for _ in range(num_scenes)]
        episode_dist = [deque(maxlen=num_episodes) for _ in range(num_scenes)]
    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros(num_scenes)
    wait_env = np.zeros(num_scenes)

    g_episode_rewards = deque(maxlen=1000)
    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)
    per_step_g_rewards = deque(maxlen=1000)
    g_process_rewards = np.zeros(num_scenes)

    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    torch.set_grad_enabled(False)

    nc = args.num_sem_categories + 4
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)
    origins = np.zeros((num_scenes, 3))
    lmb = np.zeros((num_scenes, 4)).astype(int)
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes
        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w
            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h
        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0
        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r = int(r * 100.0 / args.map_resolution)
            loc_c = int(c * 100.0 / args.map_resolution)
            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
            lmb[e] = get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h))
            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]
        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0
        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r = int(r * 100.0 / args.map_resolution)
        loc_c = int(c * 100.0 / args.map_resolution)
        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h))
        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]
        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1],
                                lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    intrinsic_rews = torch.zeros(num_scenes).to(device)

    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.) ** 2

    init_map_and_pose()

    # --- Enhanced modules ---
    gaussian_modules = None
    scene_graph_builders = None
    reward_computers = None

    if args.use_3dgs:
        gaussian_modules = [
            GaussianSplatting3D(
                num_sem_categories=int(args.num_sem_categories),
                max_gaussians=args.max_gaussians,
                device=device
            ).to(device) for _ in range(num_scenes)
        ]
        for gm in gaussian_modules:
            gm.eval()

    if args.use_scene_graph:
        scene_graph_builders = [
            SceneGraphBuilder(
                num_sem_categories=int(args.num_sem_categories),
                device=device
            ).to(device) for _ in range(num_scenes)
        ]
        for sg in scene_graph_builders:
            sg.eval()

    if args.use_enhanced_reward:
        reward_computers = [
            MultiDimensionalReward(
                direction_weight=args.direction_reward_weight,
                efficiency_weight=args.efficiency_reward_weight,
                collision_weight=args.collision_reward_weight,
                exploration_weight=args.exploration_reward_weight,
            ) for _ in range(num_scenes)
        ]

    camera_K = build_camera_intrinsics(args, device)

    # --- Policy setup ---
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(
        0, 1, (ngc, local_w, local_h), dtype='uint8')
    g_action_space = gym.spaces.Box(
        low=0.0, high=0.99, shape=(2,), dtype=np.float32)
    g_hidden_size = args.global_hidden_size

    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    if args.use_enhanced_policy:
        g_policy = GaussianNavPolicy(
            g_observation_space.shape, g_action_space,
            model_type=1,
            base_kwargs={
                'recurrent': args.use_recurrent_global,
                'hidden_size': g_hidden_size,
                'num_sem_categories': ngc - 8,
            },
            gaussian_feature_dim=args.gaussian_feature_dim,
            use_transformer=args.use_transformer
        ).to(device)

        g_agent = EnhancedPPO(
            g_policy, args.clip_param, args.ppo_epoch,
            args.num_mini_batch, args.value_loss_coef,
            args.entropy_coef, lr=args.lr, eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            adaptive_lr=args.adaptive_lr)
    else:
        g_policy = RL_Policy(
            g_observation_space.shape, g_action_space,
            model_type=1,
            base_kwargs={
                'recurrent': args.use_recurrent_global,
                'hidden_size': g_hidden_size,
                'num_sem_categories': ngc - 8
            }).to(device)
        g_agent = algo.PPO(
            g_policy, args.clip_param, args.ppo_epoch,
            args.num_mini_batch, args.value_loss_coef,
            args.entropy_coef, lr=args.lr, eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    extras = torch.zeros(num_scenes, 2)

    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps, num_scenes,
        g_observation_space.shape, g_action_space,
        g_policy.rec_state_size, es).to(device)

    if args.load != "0":
        print("Loading model {}".format(args.load))
        state_dict = torch.load(
            args.load, map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict, strict=False)

    if args.eval:
        g_policy.eval()

    # --- First observation ---
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    _, local_map, _, local_pose = sem_map_module(
        obs, poses, local_map, local_pose)

    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r = int(r * 100.0 / args.map_resolution)
        loc_c = int(c * 100.0 / args.map_resolution)
        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx in range(num_scenes)]))

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

    planner_inputs = [{} for _ in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0
    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    # --- Enhanced tracking ---
    nav_metrics = {
        'reward_components': defaultdict(list),
        'gaussian_counts': [],
        'scene_graph_sizes': [],
    }

    # --- Paper figures: trajectory recording for eval ---
    save_figures = getattr(args, 'save_paper_figures', 0) and args.eval
    paper_fig_dir = getattr(args, 'paper_figures_dir', './tmp/paper_figures')
    episode_trajectories = [[] for _ in range(num_scenes)]
    episode_start_poses = [None for _ in range(num_scenes)]
    episode_sem_maps = [None for _ in range(num_scenes)]
    saved_success_count = 0
    saved_failure_count = 0
    failure_cases_for_comparison = []  # 存储失败案例用于改进前后对比

    if save_figures:
        from utils.paper_figures import (
            save_trajectory_comparison,
            save_failure_case_comparison,
        )
        os.makedirs(paper_fig_dir, exist_ok=True)
        print("Paper figures will be saved to {}".format(paper_fig_dir))
        # Initialize start poses for each env (set after first step)
        for e in range(num_scenes):
            episode_start_poses[e] = full_pose[e].cpu().numpy().copy()

    print("Starting enhanced training loop...")
    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)

                # Save paper figures (Fig 5-1, 5-3) for typical cases
                if save_figures and len(episode_trajectories[e]) >= 2:
                    traj = np.array(episode_trajectories[e])
                    sem_map_e = full_map[e].cpu().numpy()
                    start_pose = episode_start_poses[e]
                    start_cells = (int(start_pose[1] * 100 / args.map_resolution),
                                   int(start_pose[0] * 100 / args.map_resolution)) if start_pose is not None else (0, 0)
                    map_size = args.map_size_cm // args.map_resolution
                    start_cells = (max(0, min(map_size - 1, start_cells[0])),
                                  max(0, min(map_size - 1, start_cells[1])))
                    goal_cells = None
                    if success and traj.shape[0] > 0:
                        goal_cells = (int(traj[-1, 0]), int(traj[-1, 1]))

                    if success and saved_success_count < 2:
                        out_name = "fig_5_1_trajectory_comparison_scene{}.png".format(saved_success_count + 1)
                        save_trajectory_comparison(
                            sem_map=sem_map_e,
                            trajectory_ours=traj,
                            trajectory_baseline=None,
                            start_pos=start_cells,
                            goal_pos=goal_cells,
                            map_resolution=args.map_resolution,
                            map_size_cm=args.map_size_cm,
                            output_path=out_name,
                            output_dir=paper_fig_dir,
                            scene_label="Success Case {}".format(saved_success_count + 1),
                            success=True,
                        )
                        print("Saved paper figure: {}".format(out_name))
                        saved_success_count += 1

                    if not success and saved_failure_count < 2:
                        # 尝试从语义图中估计目标物体位置（goal_cat_id+4 通道）
                        goal_cat_id = infos[e].get('goal_cat_id', 0)
                        goal_ch = goal_cat_id + 4
                        goal_center = None
                        if goal_ch < sem_map_e.shape[0] and sem_map_e[goal_ch].sum() > 0:
                            yy, xx = np.where(sem_map_e[goal_ch] > 0.1)
                            if len(yy) > 0:
                                goal_center = (int(np.mean(yy)), int(np.mean(xx)))
                        failure_cases_for_comparison.append({
                            'trajectory': traj.copy(),
                            'sem_map': sem_map_e.copy(),
                            'start_pos': start_cells,
                            'goal_pos': goal_center or goal_cells,
                            'goal_name': infos[e].get('goal_name', 'unknown'),
                        })
                        out_name = "fig_5_1_trajectory_comparison_failure{}.png".format(saved_failure_count + 1)
                        save_trajectory_comparison(
                            sem_map=sem_map_e,
                            trajectory_ours=traj,
                            trajectory_baseline=None,
                            start_pos=start_cells,
                            goal_pos=goal_cells,
                            map_resolution=args.map_resolution,
                            map_size_cm=args.map_size_cm,
                            output_path=out_name,
                            output_dir=paper_fig_dir,
                            scene_label="Failure Case {}".format(saved_failure_count + 1),
                            success=False,
                        )
                        print("Saved paper figure: {}".format(out_name))
                        saved_failure_count += 1

                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)

                wait_env[e] = 1.
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)

                # Reset trajectory recording for new episode
                episode_start_poses[e] = full_pose[e].cpu().numpy().copy()
                episode_trajectories[e] = []

                if reward_computers:
                    reward_computers[e].reset()
                if scene_graph_builders:
                    scene_graph_builders[e].reset()
                if gaussian_modules:
                    gaussian_modules[e].gaussians = None

        # Semantic Mapping
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = sem_map_module(
            obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r = int(r * 100.0 / args.map_resolution)
            loc_c = int(c * 100.0 / args.map_resolution)
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # Record trajectory for paper figures
        if save_figures:
            map_size = args.map_size_cm // args.map_resolution
            for e in range(num_scenes):
                if not wait_env[e] and not finished[e]:
                    pose = planner_pose_inputs[e, :3]
                    map_r = int(pose[1] * 100.0 / args.map_resolution)
                    map_c = int(pose[0] * 100.0 / args.map_resolution)
                    map_r = max(0, min(map_size - 1, map_r))
                    map_c = max(0, min(map_size - 1, map_c))
                    episode_trajectories[e].append([map_r, map_c])

        # Global Policy
        if l_step == args.num_local_steps - 1:
            for e in range(num_scenes):
                if wait_env[e] == 1:
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1],
                         lmb[e, 2]:lmb[e, 3]] = local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs_full = full_pose[e].cpu().numpy()
                r, c = locs_full[1], locs_full[0]
                loc_r = int(r * 100.0 / args.map_resolution)
                loc_c = int(c * 100.0 / args.map_resolution)

                lmb[e] = get_local_map_boundaries(
                    (loc_r, loc_c), (local_w, local_h), (full_w, full_h))
                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = nn.MaxPool2d(
                args.global_downscaling)(full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()

            goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx in range(num_scenes)]))
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, extras)

            g_value, g_action, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[g_step + 1],
                    g_rollouts.rec_states[g_step + 1],
                    g_rollouts.masks[g_step + 1],
                    extras=g_rollouts.extras[g_step + 1],
                    deterministic=False)

            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [[int(action[0] * local_w),
                             int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # Update long-term goal if target found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        for e in range(num_scenes):
            cn = infos[e]['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1

        # Take action
        planner_inputs = [{} for _ in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                    ].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

        # Training
        torch.set_grad_enabled(True)
        if (g_step % args.num_global_steps == args.num_global_steps - 1 and
                l_step == args.num_local_steps - 1):
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                ).detach()

                g_rollouts.compute_returns(
                    g_next_value, args.use_gae, args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)

        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
                if (end - start) > 0 else "FPS 0,",
            ])

            log += "\n\tRewards:"
            if len(g_episode_rewards) > 0:
                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist_val in episode_dist[e]:
                        total_dist.append(dist_val)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)
                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success), np.mean(total_spl),
                        np.mean(total_dist), len(total_spl))
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success), np.mean(episode_spl),
                        np.mean(episode_dist), len(episode_spl))

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_value_losses),
                        np.mean(g_action_losses),
                        np.mean(g_dist_entropies))
                ])

            if args.use_3dgs and gaussian_modules:
                g_counts = [gm.gaussians.centers.shape[0]
                            if gm.gaussians else 0 for gm in gaussian_modules]
                log += "\n\t3DGS: avg {:.0f} Gaussians".format(np.mean(g_counts))

            print(log)
            logging.info(log)

        # Save models
        if (step * num_scenes) % args.save_interval < num_scenes:
            if (len(g_episode_rewards) >= 1000 and
                    np.mean(g_episode_rewards) >= best_g_reward and
                    not args.eval):
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        if (step * num_scenes) % args.save_periodic < num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))

    # Final evaluation
    if args.eval:
        print("Dumping eval details...")
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist_val in episode_dist[e]:
                total_dist.append(dist_val)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success), np.mean(total_spl),
                np.mean(total_dist), len(total_spl))
            print(log)
            logging.info(log)

        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(
                key,
                sum(success_per_category[key]) / len(success_per_category[key]),
                sum(spl_per_category[key]) / len(spl_per_category[key]))
        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)

        # Save enhanced metrics
        metrics_path = os.path.join(dump_dir, 'enhanced_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'success': float(np.mean(total_success)) if total_success else 0,
                'spl': float(np.mean(total_spl)) if total_spl else 0,
                'dtg': float(np.mean(total_dist)) if total_dist else 0,
                'num_episodes': len(total_spl),
                'use_3dgs': args.use_3dgs,
                'use_transformer': args.use_transformer,
                'use_enhanced_reward': args.use_enhanced_reward,
            }, f, indent=2)
        print("Enhanced metrics saved to {}".format(metrics_path))

        # Fig 5-3: 失败案例与改进前后对比（需要至少1个失败案例）
        if save_figures and len(failure_cases_for_comparison) > 0:
            fc = failure_cases_for_comparison[0]
            traj_before = fc['trajectory']
            start = np.array(fc['start_pos'])
            goal = fc.get('goal_pos')
            # "改进后" 示意：从起点到目标/终点的更优路径（直线插值作为改进示意）
            if goal is not None:
                goal_arr = np.array(goal)
                n_pts = max(20, len(traj_before))
                traj_after = np.column_stack([
                    np.linspace(start[0], goal_arr[0], n_pts),
                    np.linspace(start[1], goal_arr[1], n_pts),
                ]).astype(int)
            else:
                # 无目标时用失败轨迹的终点作为"改进后"终点
                end_pt = np.array(traj_before[-1]) if len(traj_before) > 0 else start
                n_pts = max(20, len(traj_before))
                traj_after = np.column_stack([
                    np.linspace(start[0], end_pt[0], n_pts),
                    np.linspace(start[1], end_pt[1], n_pts),
                ]).astype(int)
            save_failure_case_comparison(
                sem_map=fc['sem_map'],
                trajectory_before=traj_before,
                trajectory_after=traj_after,
                start_pos=fc['start_pos'],
                goal_pos=fc.get('goal_pos'),
                failure_annotations=[{'position': fc['start_pos'], 'label': 'Start'}],
                map_resolution=args.map_resolution,
                map_size_cm=args.map_size_cm,
                output_path="fig_5_4_failure_case_comparison.png",
                output_dir=paper_fig_dir,
            )
            print("Saved paper figure: fig_5_4_failure_case_comparison.png")


if __name__ == "__main__":
    main()
