import threading
import numpy as np
import torch

class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, goal_type='full'):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        # Set goal type
        self.goal_type = goal_type

        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'div': np.empty([self.size, 1]),
                        'quality':np.empty([self.size, 1]),
                        }
        
        # Used to store trajectory encodings
        self.encodings_cache=None
        self.encodings_lambda=None

        # Contrastive learning parameters
        self.min_trajectories_for_contrastive = 50

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the information
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            # print("mb_ag",mb_ag.shape)
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            # print("mb_g",mb_g.shape)
            # print("mb_obs",mb_obs.shape)

            # Calculate and store diversity and quality
            self.buffers['div'][idxs]=self.compute_diversity(mb_ag).reshape(-1,1)
            self.buffers['quality'][idxs] = self.compute_quality(mb_ag, mb_g).reshape(-1,1)

            # Clear the encodings cache after storing new episodes
            self.clear_encodings_cache()

            self.n_transitions_stored += self.T * batch_size

    # Calculate the diversity
    def compute_diversity(self, trajectory_goals, window_size=2, eps=1e-8, clip_max=None):
        # 按 goal_type 提取特征维度（全批量，无 Python 循环）
        if self.goal_type == 'full':
            traj = trajectory_goals                    # [N, T+1, D]
        elif self.goal_type == 'rotate':
            traj = trajectory_goals[:, :, 3:]          # [N, T+1, 4]
        else:
            raise ValueError("Invalid goal_type. Choose 'full' or 'rotate'.")

        # 向量化归一化：[N, T+1, D]
        norms = np.linalg.norm(traj, axis=2, keepdims=True)   # [N, T+1, 1]
        traj_norm = traj / (norms + eps)                       # [N, T+1, D]

        # 向量化计算 window_size=2 时每对相邻向量的 2×2 Gram 行列式：
        #   det([[a·a, a·b],[b·a, b·b]]) = (a·a)(b·b) - (a·b)²
        # 与原始逐步 np.linalg.det 数学完全一致
        a = traj_norm[:, :-1, :]                                    # [N, T, D]
        b = traj_norm[:, 1:,  :]                                    # [N, T, D]
        aa = np.sum(a * a, axis=2)                                  # [N, T]
        bb = np.sum(b * b, axis=2)                                  # [N, T]
        ab = np.sum(a * b, axis=2)                                  # [N, T]
        det_values = aa * bb - ab ** 2                              # [N, T]

        diversity_scores = np.sum(np.maximum(0.0, det_values), axis=1)  # [N]

        if clip_max is not None:
            diversity_scores = np.clip(diversity_scores, 0, clip_max)

        # 归一化到 [0, 1]
        min_val = np.min(diversity_scores)
        max_val = np.max(diversity_scores)
        if max_val - min_val > eps:
            diversity_scores = (diversity_scores - min_val) / (max_val - min_val)
        else:
            diversity_scores = np.ones_like(diversity_scores) * 0.5

        return diversity_scores
    
    # Calculate the quality
    def compute_quality(self, achieved_goals, desired_goals):
        # Calculate the quality of each trajectory (Euclidean distance to the goal)
        final_ag=achieved_goals[:,-1,:]
        final_g = desired_goals[:, -1, :]

        # Select the appropriate features based on goal_type
        if self.goal_type == 'full':
            ag_feature = final_ag
            g_feature = final_g
        elif self.goal_type == 'rotate':
            ag_feature = final_ag[:, 3:]
            g_feature = final_g[:, 3:]
        else:
            raise ValueError("Invalid goal_type. Choose 'full' or 'rotate'.")

        distance = np.linalg.norm(ag_feature - g_feature, axis=1)
        # print("distance:",distance.shape)
        sigma = 1.0
        quality_scores = np.exp(- (distance ** 2) / (2 * sigma ** 2))

        eps = 0.01
        quality_scores = np.maximum(quality_scores, eps) 
        
        return quality_scores


    # sample the data from the replay buffer
    def sample(self, batch_size, learning_step, sampling_func=None,
               lambda_val=None, cached_scores=None):
        temp_buffers = {}

        # Copy the stored data up to current size into a temporary dictionary
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        # Get the stored trajectory data (states, actions, goals)
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # 注入预计算的对比分数，采样函数可直接使用，跳过内部 LSTM 推理
        if cached_scores is not None:
            temp_buffers['contrastive_scores'] = cached_scores

        if sampling_func is None:
            sampling_func = self.sample_func

        if lambda_val is not None:
            transitions = sampling_func(temp_buffers, batch_size, learning_step, lambda_val)
        else:
            transitions = sampling_func(temp_buffers, batch_size, learning_step)

        return transitions
    

    def has_sufficient_trajectories(self):
        return self.current_size >= self.min_trajectories_for_contrastive
    
    def get_trajectory_batch(self, indices):
        return self.buffers['ag'][indices, :-1, :]
    
    def clear_encodings_cache(self):
        self.encodings_cache = None
        self.encodings_lambda = None
    
    def cache_trajectory_encodings(self, encoder, lambda_val, device=None):

        if not self.has_sufficient_trajectories():
            return
            
        if self.encodings_cache is not None and abs(self.encodings_lambda - lambda_val) < 1e-6:
            return
            
        with torch.no_grad():
            # print("relay_buffer.py goal_type:",self.goal_type)
            if self.goal_type == 'full':
                trajectories = torch.tensor(
                    self.buffers['ag'][:self.current_size, :-1, :], 
                    dtype=torch.float32
                )
            elif self.goal_type == 'rotate':
                trajectories = torch.tensor(
                    self.buffers['ag'][:self.current_size, :-1, 3:],
                    dtype=torch.float32
                )
            else:
                raise ValueError("Invalid goal_type. Choose 'full' or 'rotate'.")

            lambda_tensor = torch.tensor(
                [[lambda_val]] * self.current_size, 
                dtype=torch.float32
            )
            
            if device is not None:
                trajectories = trajectories.to(device)
                lambda_tensor = lambda_tensor.to(device)
                
            # Calculate the encodings
            self.encodings_cache = encoder(trajectories, lambda_tensor)
            self.encodings_lambda = lambda_val
    

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        
        # Current buffer not full and has enough consecutive space for new episode
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        
        # Current buffer not full, but remaining consecutive space is not enough
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx