import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer_contrastive import replay_buffer  # Change to contrastive version
from rl_modules.models_sac import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her_dpp_λ_contrastive import her_sampler  # Change to ACDC version
from rl_modules.trajectory_encoder import TrajectoryEncoder

"""
SAC with HER and ACDC (MPI-version)
"""

class sac_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params

        # goal_type is always set by arguments_sac.py
        self.goal_type = args.goal_type
        
        # 使用models.py中定义的网络结构
        self.actor = actor(env_params)
        self.critic = critic(env_params)
        self.critic_target = critic(env_params)
        
        # 同步网络
        sync_networks(self.actor)
        sync_networks(self.critic)
        
        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Create trajectory encoder for ACDC
        self.trajectory_encoder = TrajectoryEncoder(
            goal_dim=env_params['goal'], 
            hidden_dim=64, 
            output_dim=32,
            goal_type=self.goal_type
        )
        
        # Handle hybrid GPU mode for ACDC
        self.hybrid_mode = getattr(args, 'hybrid_gpu', False)
        
        # GPU设置
        if self.args.cuda and not self.hybrid_mode:
            # Full GPU mode
            self.actor.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
            self.trajectory_encoder.cuda()
        elif self.hybrid_mode and torch.cuda.is_available():
            # Hybrid mode: only trajectory encoder on GPU
            self.trajectory_encoder.cuda()
            print("[Hybrid Mode] LSTM on GPU, Actor/Critic on CPU")
        else:
            # CPU mode
            print("[CPU Mode] All components on CPU")
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        
        # Optimizer for trajectory encoder (ACDC)
        self.encoder_optimizer = torch.optim.Adam(self.trajectory_encoder.parameters(), lr=0.001)
        
        # 温度参数（熵正则化系数）
        if getattr(self.args, 'auto_alpha', True):
            self.target_entropy = torch.tensor(-env_params['action'], dtype=torch.float32)
            if self.args.cuda:
                self.target_entropy = self.target_entropy.cuda()
            
            # 可学习的温度参数
            self.log_alpha = torch.zeros(1, requires_grad=True)
            if self.args.cuda:
                self.log_alpha = self.log_alpha.cuda()
            alpha_lr = getattr(self.args, 'lr_alpha', self.args.lr_actor)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha_value = getattr(self.args, 'alpha', 0.2)

        if hasattr(self.env, 'unwrapped'):
            compute_reward_func = self.env.unwrapped.compute_reward
        elif hasattr(self.env, 'env'):
            # Navigate through wrapper layers
            env = self.env.env
            while hasattr(env, 'env') and not hasattr(env, 'compute_reward'):
                env = env.env
            if hasattr(env, 'compute_reward'):
                compute_reward_func = env.compute_reward
            else:
                raise AttributeError("Could not find compute_reward in environment chain")
        else:
            compute_reward_func = self.env.compute_reward
        
        # HER sampler with ACDC
        self.her_module = her_sampler(
            self.args.replay_strategy, 
            self.args.replay_k, 
            compute_reward_func,
            goal_type=self.goal_type
        )
        
        # Contrastive Learning Parameters (ACDC)
        self.contrastive_start_size = 50
        self.contrastive_train_freq = 5
        self.contrastive_lambda = 0.1
        self.enable_contrastive = True
        
        # Determine success rate thresholds for adaptive learning
        self.her_module.adjust_thresholds_for_environment(self.args.env_name)
        
        # Set trajectory encoder for HER module
        self.her_module.set_trajectory_encoder(self.trajectory_encoder)
        
        self.current_success_rate = None
        
        # Replay buffer with ACDC support
        self.buffer = replay_buffer(
            self.env_params, 
            self.args.buffer_size, 
            self.her_module.sample_her_transitions,  # Will be changed dynamically
            goal_type=self.goal_type
        )
        
        # Sync minimum trajectories for contrastive
        self.buffer.min_trajectories_for_contrastive = self.contrastive_start_size
        
        # normalizers
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        
        # Track training progress
        self.current_step = 0
        self.update_count = 0

        # Cache for contrastive trajectory scores (per-cycle, invalidated on store_episode)
        self.cached_contrastive_scores = None
        
        # 保存路径
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + '_sac_acdc')
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
    
    @property
    def alpha(self):
        if getattr(self.args, 'auto_alpha', True):
            return self.log_alpha.exp()
        else:
            return self.alpha_value

    def learn(self):
        """训练网络 with ACDC"""
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                
                for _ in range(self.args.num_rollouts_per_mpi):
                    # 重置rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # 重置环境
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    
                    # 开始采样
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            
                            # Initial exploration phase
                            if epoch < 5:
                                # Random actions for initial exploration
                                action = np.random.uniform(
                                    -self.env_params['action_max'],
                                    self.env_params['action_max'],
                                    size=self.env_params['action']
                                )
                            else:
                                # SAC action sampling
                                action, _, _ = self.actor.sample(input_tensor)
                                action = action.cpu().numpy().squeeze()
                        
                        # 执行动作
                        observation_new, reward, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        
                        # 添加到episode
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        
                        # 更新观察
                        obs = obs_new
                        ag = ag_new
                    
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                
                # 转换为数组
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                
                # 存储episodes with diversity and quality computation
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self.cached_contrastive_scores = None  # buffer 已更新，下一次 update 时重算
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                
                # 训练网络（每次梯度更新后立即软更新目标网络，与标准SAC一致）
                for _ in range(self.args.n_batches):
                    self._update_network()
                    self.update_count += 1
                    self._soft_update_target_network(self.critic_target, self.critic)

                # Update current step
                self.current_step += self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']
                
                # Enable contrastive sampling after sufficient data
                if self.buffer.has_sufficient_trajectories() and not self.her_module.use_contrastive:
                    self.her_module.enable_contrastive_sampling()
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print(f"[Epoch {epoch}] Enabled contrastive sampling")
            
            # 评估
            success_rate = self._eval_agent()
            self.current_success_rate = success_rate  # Store for adaptive lambda
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                # Get learning rate info from ACDC
                lr_info = self.her_module.get_learning_rate_info()
                
                print('[{}] epoch: {}, success rate: {:.3f}, alpha: {:.3f}, lambda_lr: {:.4f}'.format(
                    datetime.now(), epoch, success_rate, 
                    self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha, lr_info['current_lambda_lr']))
                
                # Save model
                if epoch % self.args.save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'o_norm_mean': self.o_norm.mean, 
                        'o_norm_std': self.o_norm.std, 
                        'g_norm_mean': self.g_norm.mean, 
                        'g_norm_std': self.g_norm.std,
                        'actor_state_dict': self.actor.state_dict(), 
                        'critic_state_dict': self.critic.state_dict(),
                        'critic_target_state_dict': self.critic_target.state_dict(),
                        'encoder_state_dict': self.trajectory_encoder.state_dict(),
                        'success_rate': success_rate
                    }, self.model_path + f'/model_epoch_{epoch}.pt')

    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        num_transitions = mb_actions.shape[1]
        
        buffer_temp = {
            'obs': mb_obs, 
            'ag': mb_ag,
            'g': mb_g, 
            'actions': mb_actions, 
            'obs_next': mb_obs_next,
            'ag_next': mb_ag_next,
        }
        
        # Use basic HER sampling for normalizer update
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_network(self):
        """Update SAC networks with ACDC sampling"""
        # Calculate learning step for curriculum
        learning_step = self.current_step / (self.args.num_rollouts_per_mpi * self.env_params['max_timesteps'])
        
        # Get lambda value for curriculum
        lambda_val = self.her_module.dynamic_lambda(learning_step, self.current_success_rate)
        
        # Select sampling function based on contrastive availability
        if self.her_module.use_contrastive and self.buffer.has_sufficient_trajectories():
            sampling_func = self.her_module.sample_her_contrastive_transitions

            # 每个 cycle 只推理一次 LSTM，后续 39 次复用缓存分数
            if self.cached_contrastive_scores is None:
                device = next(self.trajectory_encoder.parameters()).device
                ag_all = self.buffer.buffers['ag'][:self.buffer.current_size]
                T_buf = ag_all.shape[1] - 1
                key_frames = [0, T_buf // 4, T_buf // 2, 3 * T_buf // 4, T_buf]

                with torch.no_grad():
                    if self.goal_type == 'full':
                        traj_t = torch.tensor(ag_all[:, key_frames, :], dtype=torch.float32).to(device)
                    else:
                        traj_t = torch.tensor(ag_all[:, key_frames, 3:], dtype=torch.float32).to(device)
                    lam_t = torch.tensor(
                        [[lambda_val]] * self.buffer.current_size, dtype=torch.float32
                    ).to(device)
                    enc = self.trajectory_encoder(traj_t, lam_t)
                    scores = torch.norm(enc, dim=1).cpu().numpy()
                    scores = np.where(np.isnan(scores), 1e-8, scores)

                self.cached_contrastive_scores = scores
        else:
            sampling_func = self.her_module.sample_her_diversity_transitions

        # 采样transitions with ACDC（contrastive 模式下传入缓存分数，跳过采样内部的 LSTM）
        transitions = self.buffer.sample(
            self.args.batch_size,
            learning_step,
            sampling_func=sampling_func,
            lambda_val=lambda_val,
            cached_scores=self.cached_contrastive_scores
        )
        
        # 预处理
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        
        # 归一化
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # 转换为tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        
        # Contrastive learning update (ACDC)
        if (self.enable_contrastive and 
            self.update_count % self.contrastive_train_freq == 0 and
            self.buffer.current_size >= self.contrastive_start_size):
            
            # Prepare data for contrastive learning
            temp_buffers = {}
            for key in self.buffer.buffers.keys():
                temp_buffers[key] = self.buffer.buffers[key][:self.buffer.current_size]
            
            # Train encoder
            device = next(self.trajectory_encoder.parameters()).device
            total_loss = self.her_module.train_contrastive(
                temp_buffers, 
                learning_step, 
                device,
                lambda_val
            )
            
            if total_loss is not None:
                self.encoder_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trajectory_encoder.parameters(),
                    max_norm=3.0,
                    error_if_nonfinite=False
                )
                
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.encoder_optimizer.step()
                else:
                    print(f"[WARNING] Invalid gradient norm: {grad_norm}")
                    self.encoder_optimizer.zero_grad()

        # 更新Critic
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.sample(inputs_next_norm_tensor)
            qf1_next_target, qf2_next_target = self.critic_target(inputs_next_norm_tensor, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = r_tensor + self.args.gamma * min_qf_next_target
            
            # Clip Q-value for stability
            clip_return = 1 / (1 - self.args.gamma)
            next_q_value = torch.clamp(next_q_value, -clip_return, 0)

        qf1, qf2 = self.critic(inputs_norm_tensor, actions_tensor)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()

        # 更新Actor
        pi, log_pi, _ = self.actor.sample(inputs_norm_tensor)
        qf1_pi, qf2_pi = self.critic(inputs_norm_tensor, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()

        # 更新温度参数（如果启用自动调参）
        if getattr(self.args, 'auto_alpha', True):
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    _, _, actions = self.actor.sample(input_tensor)  # 使用确定性动作进行评估
                    actions = actions.detach().cpu().numpy().squeeze()

                observation_new, _, done, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']

                # Deal with success info（老版gym robotics用 is_success）
                success_val = info.get('is_success', info.get('success', False))
                is_success = 1.0 if success_val else 0.0 if isinstance(success_val, bool) else float(success_val)
                per_success_rate.append(is_success)

                if done:
                    break
            
            # 记录episode的最终成功状态（最后一步的success）
            total_success_rate.append(per_success_rate[-1] if per_success_rate else 0.0)
        
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()