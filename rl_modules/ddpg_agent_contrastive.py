import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer_contrastive import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her_dpp_λ_contrastive import her_sampler
from rl_modules.trajectory_encoder import TrajectoryEncoder


"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):

        self.args = args
        self.env = env
        self.env_params = env_params

        # Ensure the goal type
        if not hasattr(args, 'goal_type'):
            self.goal_type = self._determine_goal_type(args.env_name)
        else:
            self.goal_type = args.goal_type
            
        # print(f"Using goal_type: {self.goal_type} for environment: {args.env_name}")

        self.hybrid_mode = getattr(args, 'hybrid_gpu', False)

        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # 创建轨迹编码器
        self.trajectory_encoder = TrajectoryEncoder(
            goal_dim=env_params['goal'], 
            hidden_dim=64, 
            output_dim=32,
            goal_type=self.goal_type)

        if self.args.cuda and not self.hybrid_mode:
            # 完全GPU模式
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.trajectory_encoder.cuda()

        elif self.hybrid_mode and torch.cuda.is_available():
            # 混合模式：只有LSTM在GPU
            self.trajectory_encoder.cuda()
            # print("[Hybrid Mode] LSTM on GPU, Actor/Critic on CPU")
        else:
            # 完全CPU模式
            # print("[CPU Mode] All components on CPU")
            pass

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # 轨迹编码器的优化器
        self.encoder_optim = torch.optim.Adam(self.trajectory_encoder.parameters(), lr=0.001)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, 
                                      self.args.replay_k, 
                                      self.env.compute_reward, 
                                      goal_type=self.goal_type)
        
        # Contrastive Leaarning Parameter
        self.contrastive_start_size=50 #启动对比学习的最小轨迹数
        self.contrastive_train_freq=5 #每隔多少次更新网络训练一次对比学习
        self.contrastive_lambda = 0.1 #对比学习损失的权重
        self.enable_contrastive = True #是否启用对比学习

        
        # 确定success rate的阈值
        self.her_module.adjust_thresholds_for_environment(self.args.env_name)

        self.current_success_rate = None
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, 
                                    self.her_module.sample_her_transitions, goal_type=self.goal_type)
        
        # Set trajectory encoder (Start Contrastive Learning Phase)
        self.her_module.set_trajectory_encoder(self.trajectory_encoder)
        # print(f"[INIT] trajectory_encoder set: {self.her_module.trajectory_encoder is not None}")
    
        # 将replay_buffer的阈值
        self.buffer.min_trajectories_for_contrastive = self.contrastive_start_size
        # print(f"[INIT] Synchronized min_trajectories: {self.buffer.min_trajectories_for_contrastive}")

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
    
    # determine the goal type
    def _determine_goal_type(self, env_name):
        if 'Fetch' in env_name:
            return 'full'
        elif 'HandManipulateEggFull-v0' in env_name:
            return 'full'
        elif 'HandManipulateBlockRotateXYZ-v0' in env_name:
            return 'rotate'
        elif 'HandManipulatePenRotate-v0' in env_name:
            return 'rotate'
        else:
            return 'full'
    
    # 每个周期(epoch)内会进行多个采集周期,在每个cycle里又执行多个采集任务
    # 每个rollout中，智能体与环境交互,按时间步收集完整的episode
    def learn(self):
        """
        train the network

        """
        # 当前的时间
        current_step=0
        update_count=0

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []

                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']

                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # 将收集到的episode数据转换成numpy数组
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                # mb_g是原目标
                # 将收集到的数据存入 replay buffer 中
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

                # 利用采集的episode数据更新输入数据的归一化器
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                # 检查是否启用对比学习
                if self.enable_contrastive and self.buffer.current_size >= self.contrastive_start_size:
                    self.her_module.enable_contrastive_sampling()

                for _ in range(self.args.n_batches):
                    # train the network
                    update_count += 1
                    self._update_network(current_step, update_count)
                
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)


                # 更新current_step
                current_step += self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']

            # Adjust learning rate
            success_rate = self._eval_agent()
            # self.current_success_rate = success_rate

            if hasattr(self.her_module, 'adaptive_lr_manager'):
                self.her_module.adaptive_lr_manager.update_lambda_learning_rate(success_rate)
                # if MPI.COMM_WORLD.Get_rank() == 0:
                #     print(f"[ADAPTIVE] Success Rate: {success_rate:.3f} -> Lambda LR: {updated_lr:.6f}")

            if MPI.COMM_WORLD.Get_rank() == 0:
                # 获取学习率信息
                lr_info = self.her_module.get_learning_rate_info()
                    
                current_lambda_lr = lr_info.get('current_lambda_lr', lr_info.get('current_lr', 0.01))
                base_lambda_lr = lr_info.get('base_lambda_lr', lr_info.get('current_lr', 0.01))
                
                print('[{}] epoch is: {}, success rate is: {:.3f}, '
                    'current_lambda_lr: {:.4f}, base_lr: {:.3f}'.format(
                    datetime.now(), epoch, success_rate, 
                    current_lambda_lr, base_lambda_lr))
                
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

            # Print GPU information
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     if torch.cuda.is_available():
            #         print(f"GPU is available: {torch.cuda.get_device_name(0)}")
            #         print(f"Number of GPUs: {torch.cuda.device_count()}")
            #         if self.args.cuda:
            #             print("Training will use GPU")
            #         else:
            #             print("GPU available but not used (--cuda flag not set)")
            #     else:
            #         print("GPU is not available, using CPU")

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer

    # 进行归一化，可以保证输入数据始终在合适的范围中
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        # print("mb_actions.shape:",mb_actions.shape)
        num_transitions = mb_actions.shape[1]
        # print("num_transitions:",num_transitions)
        # create the new buffer to store them

        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        
        # if self.her_module.use_contrastive:
        #     sampling_func = self.her_module.sample_her_contrastive_transitions
        # else: 
        #     sampling_func = self.her_module.sample_her_transitions
        sampling_func=self.her_module.sample_her_transitions
        
        transitions = self.buffer.sample(num_transitions, learning_step=None, 
                                         sampling_func=sampling_func)
        
        # print(self.her_module.sample_her_transitions)
        # print("The num_transitions in normalization",num_transitions)

        # k-dpp修改部分
        # batch_size_fixed=25
        # transitions = self.her_module.sample_her_transitions(buffer_temp, batch_size_fixed)

        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    # 策略更新使用batch_size=256
    def _update_network(self, current_step, update_count):

        # 计算当前完成的episode数作为整体训练阶段指标
        learning_step=current_step / (self.args.num_rollouts_per_mpi * self.env_params['max_timesteps'])

        lambda_val = self.her_module.dynamic_lambda(learning_step)
        
        # 采样函数的选择
        if self.her_module.use_contrastive:
            sampling_func = self.her_module.sample_her_contrastive_transitions
            # print('Start Contrastive Learning Stage')
        else:
            sampling_func=self.her_module.sample_her_diversity_transitions
            # print('Start Curriculum Learning Stage')

        # sample the episodes
        # 调用 self.buffer.sample中的transition,已经是通过sample_func进行完her替换目标的transition
        transitions = self.buffer.sample(self.args.batch_size, learning_step, 
                                         sampling_func=sampling_func,lambda_val=lambda_val)

        # print("batch_size",self.args.batch_size)

        # if self.args.cuda and MPI.COMM_WORLD.Get_rank() == 0 and update_count % 100 == 0:
        #     print(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
        #       f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Contrastive learning
        if self.enable_contrastive and update_count % self.contrastive_train_freq == 0 and \
           self.buffer.current_size >= self.contrastive_start_size:
            # 获取临时缓冲区
            temp_buffers = {}
            for key in self.buffer.buffers.keys():
                temp_buffers[key] = self.buffer.buffers[key][:self.buffer.current_size]
                
            # 训练对比学习
            device = next(self.trajectory_encoder.parameters()).device
            total_loss = self.her_module.train_contrastive(temp_buffers, learning_step, device, lambda_val)
            
            if total_loss is not None:
                # print("Start Stage 2: Contrastive Control")
                # 对比学习优化
                self.encoder_optim.zero_grad()
                total_loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trajectory_encoder.parameters(),
                    max_norm=3.0,
                    error_if_nonfinite=False)
                
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.encoder_optim.step()
                else:
                    print(f"[WARNING] Invalid gradient norm: {grad_norm}")
                    self.encoder_optim.zero_grad()
                
                # # 打印对比学习损失
                # if MPI.COMM_WORLD.Get_rank() == 0 and update_count % 100 == 0:
                #     try:
                #         lr_info = self.her_module.get_learning_rate_info()
                #         current_lr = lr_info.get('current_lambda_lr', lr_info.get('current_lr', 'N/A'))
                #         print(f"[{datetime.now()}] Contrastive Loss: {total_loss.item():.4f}, "
                #             f"Lambda Learning Rate: {current_lr}, Adaptive Lambda Val: {lambda_val:.6f}")
                #     except Exception as e:
                #         print(f"[{datetime.now()}] Contrastive Loss: {total_loss.item():.4f}, "
                #             f"Lambda Val: {lambda_val:.6f}")

        # 常规RL更新部分
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        # # 记录网络更新结束时的时间和耗时
        # update_end_time=time.time()
        # elapsed_time = update_end_time - update_start_time
        # print(f"[{datetime.now()}] 网络更新完成, 当前step: {current_step}, 耗时: {elapsed_time:.4f} 秒")

    # do the evaluation
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
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
