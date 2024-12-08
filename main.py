from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import algo
from algo.ppo_discrete import PPO_discrete

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))



    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # 从环境中获取 observation_space 和 action_space
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space

    # 确保所有环境都有初始距离值
    for env_idx in range(num_scenes):
        if infos[env_idx].get('distance_to_goal') is None:
            print(f"Warning: Setting default distance for env {env_idx}")
            infos[env_idx]['distance_to_goal'] = 5.0  # 设置一个默认的初始距离

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
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

        # 在计算距离惩罚之前添加调试信息
        for env_idx in range(num_scenes):
            
            # 安全地计算距离惩罚
            distance = infos[env_idx].get('distance_to_goal')
            if distance is not None:
                distance_penalty = -0.01 * distance
            else:
                print(f"Warning: distance_to_goal is None for env_idx {env_idx}")
                distance_penalty = 0.0  # 设置默认值
                

        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
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
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    def update_intrinsic_rew(e):
        # 确保距离值存在且有效
        if 'distance_to_goal' not in infos[e]:
            infos[e]['distance_to_goal'] = 5.0  # 设置默认最大距离
        if 'prev_distance' not in infos[e]:
            infos[e]['prev_distance'] = infos[e]['distance_to_goal']
        
        current_distance = float(infos[e]['distance_to_goal'])
        prev_distance = float(infos[e]['prev_distance'])
        
        # 优化进度奖励 - 使用非线性奖励
        progress = prev_distance - current_distance
        progress_reward = np.sign(progress) * (np.abs(progress) ** 0.5) * args.distance_reward_scale
        
        # 增加探索奖励的权重
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        area_reward = (curr_explored_area - prev_explored_area) * args.exploration_reward_scale * 2.0
        
        # 增加成功奖励并添加阶段性奖励
        success_reward = args.success_reward if infos[e].get('success', False) else 0.0
        if current_distance < prev_distance * 0.5:  # 添加阶段性奖励
            success_reward += args.success_reward * 0.3
            
        # 更新距离
        infos[e]['prev_distance'] = current_distance
        
        # 组合奖励
        intrinsic_rews[e] = (
            progress_reward + 
            area_reward + 
            success_reward
        ) * (args.map_resolution / 100.)**2
        
        print(f"\nDebug Rewards (env {e}):")
        print(f"  Progress Reward: {progress_reward:.4f}")
        print(f"  Area Reward: {area_reward:.4f}")
        print(f"  Success Reward: {success_reward:.4f}")
        print(f"  Total Intrinsic: {intrinsic_rews[e]:.4f}")
        print(f"  Distance Change: {prev_distance:.4f} -> {current_distance:.4f}")

    init_map_and_pose()

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(0, 1,
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'num_sem_categories': ngc - 8
                                      }).to(device)
    
    # g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
    #                    args.mini_batch_size, args.value_loss_coef,
    #                    args.entropy_coef, lr=args.lr, eps=args.eps,
    #                    max_grad_norm=args.max_grad_norm)
    
    # 全局策略观测空间和动作空间的维度
    state_dim = np.prod(g_observation_space.shape)  # 将观测空间展平
    action_dim = g_action_space.shape[0]  # 动作空间维度

    # 初始化 PPO_discrete 策略
    g_actor_critic = PPO_discrete(
        args,  # 直接传入 args 对象
        state_dim=np.prod(g_observation_space.shape),  # 观测空间展平后的维度
        action_dim=g_action_space.shape[0]  # 连续动作空间的维度
    )
    # 如果需要，可以设置最大动作值
    args.max_action = float(g_action_space.high[0])

    # 将模型移动到 GPU
    if args.cuda:
        g_actor_critic.cuda()

    # 初始化 PPO 优化器
    g_agent = g_actor_critic

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 2)

    # 调试信息：打印传递给 GlobalRolloutStorage 的参数
    print("Debug: GlobalRolloutStorage parameters:")
    print(f"  num_global_steps: {args.num_global_steps}")
    print(f"  num_scenes: {num_scenes}")
    print(f"  g_observation_space.shape: {g_observation_space.shape}")
    print(f"  g_action_space: {g_action_space}")
    print(f"  g_policy.rec_state_size: {g_policy.rec_state_size}")
    print(f"  es: {es}")

    # 获取 rec_state_size
    rec_state_size = g_policy.rec_state_size if hasattr(g_policy, 'rec_state_size') else 0


    # Storage
    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps,
        num_scenes,
        g_observation_space.shape,
        g_action_space,
        
    ).to(device)

    if args.load != "0":
        print("Loading model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    _, local_map, _, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
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

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
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

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for env_idx, x in enumerate(done):
            if x:
                spl = infos[env_idx]['spl']
                success = infos[env_idx]['success']
                dist = infos[env_idx]['distance_to_goal']
                
                if not args.eval:
                    episode_success.append(float(success))
                    episode_spl.append(float(spl))
                    episode_dist.append(float(dist))
                    
                    window_size = 50
                    if len(episode_dist) >= window_size:
                        recent_spl = list(episode_spl)[-window_size:]
                        recent_success = list(episode_success)[-window_size:]
                        recent_dist = list(episode_dist)[-window_size:]
                        
                        print(f"\nRecent {window_size} Episodes Stats:")
                        print(f"  Avg SPL: {np.mean(recent_spl):.4f}")
                        print(f"  Avg Success: {np.mean(recent_success):.4f}")
                        print(f"  Avg Distance: {np.mean(recent_dist):.4f}")
                        print(f"  Min Distance: {np.min(recent_dist):.4f}")
                        print(f"  Max Distance: {np.max(recent_dist):.4f}")
                        print(f"  Distance Std: {np.std(recent_dist):.4f}")

                wait_env[env_idx] = 1.
                update_intrinsic_rew(env_idx)
                init_map_and_pose_for_env(env_idx)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(np.asarray(
                [infos[env_idx]['goal_cat_id'] for env_idx
                 in range(num_scenes)]))
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # If you want to get exploration reward and metrics to debug, uncomment the following code
            # for env_idx in range(num_scenes):
            #     print(f"Env {env_idx} info:")
            #     print(f"  Full info: {infos[env_idx]}")

            # 安全的距离惩罚计算
            distance_penalties = []
            for env_idx in range(num_scenes):
                # 如果distance_to_goal为None，使用上一次的有效距离或默认值
                if infos[env_idx].get('distance_to_goal') is None:
                    # 尝试从环境中获取实际距离
                    try:
                        agent_position = envs.envs[env_idx].habitat_env.sim.get_agent_state().position
                        goal_position = envs.envs[env_idx].habitat_env.current_episode.goals[0].position
                        distance = float(np.linalg.norm(agent_position - goal_position))
                        infos[env_idx]['distance_to_goal'] = distance
                    except:
                        # 如果无法获取实际距离，使用默认值
                        infos[env_idx]['distance_to_goal'] = 5.0
                        
                penalty = -0.01 * infos[env_idx]['distance_to_goal']
                distance_penalties.append(penalty)

            distance_penalty = torch.tensor(distance_penalties, dtype=torch.float32).to(device)



            # 修改全局奖励计算
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx].get('g_reward', 0.0) * 2.0 if infos[env_idx].get('success', False)
                 else infos[env_idx].get('g_reward', 0.0) * 0.5
                for env_idx in range(num_scenes)]
            )).float().to(device)

            # 增加探索奖励
            exploration_reward = args.intrinsic_rew_coeff * intrinsic_rews.detach()

            # 组合多个奖励项
            g_reward = g_reward + exploration_reward + distance_penalty

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value.unsqueeze(-1),
                    g_reward.unsqueeze(-1), g_masks.unsqueeze(-1).expand(-1, 1), extras
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[g_step + 1],
                    g_rollouts.rec_states[g_step + 1],
                    g_rollouts.masks[g_step + 1],
                    extras=g_rollouts.extras[g_step + 1],
                    deterministic=False
                )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [[int(action[0] * local_w),
                             int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
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
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                ).detach()

                g_rollouts.compute_returns(g_next_value.unsqueeze(-1), args.use_gae, args.gamma, args.tau)
                # 收集训练指标
                metrics = {
                    'spl': np.mean(episode_spl) if len(episode_spl) > 0 else 0,
                    'success': np.mean(episode_success) if len(episode_success) > 0 else 0,
                }

                # 准备 ReplayBuffer
                g_agent.prepare_buffer_from_rollouts(g_rollouts)

                # 计算当前的 value loss 和 action loss
                g_value_loss, g_action_loss, g_dist_entropy = g_agent.update(total_steps=total_steps)

                # 更新 losses 队列
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
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
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    avg_success = np.mean(total_success)
                    avg_spl = np.mean(total_spl)
                    avg_dist = np.mean(total_dist)

                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        avg_success,
                        avg_spl,
                        avg_dist,
                        len(total_spl)
                    )
                    # 记录到 TensorBoard
                    writer.add_scalar('Evaluation/Avg_Success', avg_success, step)
                    writer.add_scalar('Evaluation/Avg_SPL', avg_spl, step)
                    writer.add_scalar('Evaluation/Avg_DistanceToGoal', avg_dist, step)

            else:
                if len(episode_success) > 100:
                    avg_episode_success = np.mean(episode_success)
                    avg_episode_spl = np.mean(episode_spl)
                    avg_episode_dist = np.mean(episode_dist)

                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        avg_episode_success,
                        avg_episode_spl,
                        avg_episode_dist,
                        len(episode_spl))

                    # 记录到 TensorBoard
                    writer.add_scalar('Training/Episode_Success', avg_episode_success, step)
                    writer.add_scalar('Training/Episode_SPL', avg_episode_spl, step)
                    writer.add_scalar('Training/Episode_DistanceToGoal', avg_episode_dist, step)

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                avg_value_losses = np.mean(g_value_losses)
                avg_action_losses = np.mean(g_action_losses)
                avg_dist_entropies = np.mean(g_dist_entropies)

                log += " ".join([
                    " Policy Loss value/action/dist:",
                    "{:.3f}/{:.3f}/{:.3f},".format(
                        avg_value_losses,
                        avg_action_losses,
                        avg_dist_entropies)
                ])
                # 记录到 TensorBoard
                writer.add_scalar('Training_Loss/value', avg_value_losses, step)
                writer.add_scalar('Training_Loss/action', avg_action_losses, step)
                writer.add_scalar('Training_Loss/dist', avg_dist_entropies, step)

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)
        
    writer.close()


if __name__ == "__main__":
    main()
