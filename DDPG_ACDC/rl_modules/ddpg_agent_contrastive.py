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

        # Create trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            goal_dim=env_params['goal'], 
            hidden_dim=64, 
            output_dim=32,
            goal_type=self.goal_type)

        if self.args.cuda and not self.hybrid_mode:
            # Full GPU mode
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.trajectory_encoder.cuda()

        elif self.hybrid_mode and torch.cuda.is_available():
            # Hybrid GPU mode: only LSTM on GPU
            self.trajectory_encoder.cuda()
            # print("[Hybrid Mode] LSTM on GPU, Actor/Critic on CPU")
        else:
            # Full CPU mode
            # print("[CPU Mode] All components on CPU")
            pass

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        self.encoder_optim = torch.optim.Adam(self.trajectory_encoder.parameters(), lr=0.001)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, 
                                      self.args.replay_k, 
                                      self.env.compute_reward, 
                                      goal_type=self.goal_type)
        
        # Contrastive Leaarning Parameter
        self.contrastive_start_size = 50  # Minimum number of trajectories to start contrastive learning
        self.contrastive_train_freq = 5   # Train contrastive learning every N network updates
        self.contrastive_lambda = 0.1     # Weight of contrastive learning loss
        self.enable_contrastive = True    # Whether to enable contrastive learning

        # Determine success rate thresholds
        self.her_module.adjust_thresholds_for_environment(self.args.env_name)

        self.current_success_rate = None
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, 
                                    self.her_module.sample_her_transitions, goal_type=self.goal_type)
        
        # Set trajectory encoder (Start Contrastive Learning Phase)
        self.her_module.set_trajectory_encoder(self.trajectory_encoder)
        # print(f"[INIT] trajectory_encoder set: {self.her_module.trajectory_encoder is not None}")
    

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
    
    # learn the agent
    def learn(self):
        """
        train the network

        """
        # Current step and update count
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

                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

                # Update input data normalizer with collected episode data
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                # Check if contrastive learning is enabled
                if self.enable_contrastive and self.buffer.current_size >= self.contrastive_start_size:
                    self.her_module.enable_contrastive_sampling()

                for _ in range(self.args.n_batches):
                    # train the network
                    update_count += 1
                    self._update_network(current_step, update_count)
                
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)


                # Update current_step
                current_step += self.args.num_rollouts_per_mpi * self.env_params['max_timesteps']

            # Adjust learning rate
            success_rate = self._eval_agent()
            # self.current_success_rate = success_rate

            if hasattr(self.her_module, 'adaptive_lr_manager'):
                self.her_module.adaptive_lr_manager.update_lambda_learning_rate(success_rate)
                # if MPI.COMM_WORLD.Get_rank() == 0:
                #     print(f"[ADAPTIVE] Success Rate: {success_rate:.3f} -> Lambda LR: {updated_lr:.6f}")

            if MPI.COMM_WORLD.Get_rank() == 0:
                # Gain the learning rate information
                lr_info = self.her_module.get_learning_rate_info()
                    
                current_lambda_lr = lr_info.get('current_lambda_lr', lr_info.get('current_lr', 0.01))
                base_lambda_lr = lr_info.get('base_lambda_lr', lr_info.get('current_lr', 0.01))
                
                print('[{}] epoch is: {}, success rate is: {:.3f}, '
                    'current_lambda_lr: {:.4f}, base_lr: {:.3f}'.format(
                    datetime.now(), epoch, success_rate, 
                    current_lambda_lr, base_lambda_lr))
                
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

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
    # Normalization
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
        
        sampling_func=self.her_module.sample_her_transitions
        
        transitions = self.buffer.sample(num_transitions, learning_step=None, 
                                         sampling_func=sampling_func)
        

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
    def _update_network(self, current_step, update_count):

        learning_step=current_step / (self.args.num_rollouts_per_mpi * self.env_params['max_timesteps'])

        lambda_val = self.her_module.dynamic_lambda(learning_step)

        # Sampling function selection
        if self.her_module.use_contrastive:
            sampling_func = self.her_module.sample_her_contrastive_transitions
        else:
            sampling_func=self.her_module.sample_her_diversity_transitions

        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size, learning_step,
                                         sampling_func=sampling_func, lambda_val=lambda_val)

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
            temp_buffers = {}
            for key in self.buffer.buffers.keys():
                temp_buffers[key] = self.buffer.buffers[key][:self.buffer.current_size]
                
            # Train Encoder Network
            device = next(self.trajectory_encoder.parameters()).device
            total_loss = self.her_module.train_contrastive(temp_buffers, learning_step, device, lambda_val)
            
            if total_loss is not None:
                # print("Start Stage 2: Contrastive Control")
                # Optimization
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
