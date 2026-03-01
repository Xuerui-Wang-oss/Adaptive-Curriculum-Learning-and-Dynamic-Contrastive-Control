import numpy as np
import math
import torch
import torch.nn.functional as F
from rl_modules.adaptive_lr_manager import AdaptiveLearningRateManager

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, 
                 lambda_starter_quality=0.1, learning_rate=0.01, goal_type='full'):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k

        # For curriculum learning
        self.lambda_starter_quality = lambda_starter_quality
        self.learning_rate = learning_rate

        # Set goal_type
        self.goal_type = goal_type

        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

        # The encoder required for contrastive learning will be created during agent initialization
        self.trajectory_encoder = None

        # Whether to use contrastive learning for sampling
        self.use_contrastive = False

        # Minimum number of samples for contrastive learning
        self.min_trajectories_for_contrastive = 50

        # Adaptive lambda learning rate
        self.adaptive_lr_manager = AdaptiveLearningRateManager(
            base_lambda_lr=learning_rate,
            smoothing_factor=0.7
        )
    
    def set_trajectory_encoder(self, encoder):
        self.trajectory_encoder = encoder
    
    def enable_contrastive_sampling(self):
        self.use_contrastive = True
    
    def disable_contrastive_sampling(self):
        self.use_contrastive = False

    def get_learning_rate_info(self):
        """Get learning rate related information for debugging and monitoring"""
        return self.adaptive_lr_manager.get_statistics()
    
    def adjust_thresholds_for_environment(self, env_name):
        """Adjust success rate thresholds according to environment type"""
        self.adaptive_lr_manager.adjust_thresholds_for_environment(env_name, self.goal_type)

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, learning_step=None):
        # print("batch_size:",batch_size_in_transitions)
        # traceback.print_stack()
        T = episode_batch['actions'].shape[1]

        # rollout_batch_size indicates the total number of episodes
        # The number of trajectories selected in one sampling process
        rollout_batch_size = episode_batch['actions'].shape[0]

        # batch_size refers to the number of samples to be sampled
        # print("batch_size_in_transitions:",batch_size_in_transitions)
        batch_size = batch_size_in_transitions
        # print("Normalization batch_size:", batch_size)
        
        # Randomly select batch_size trajectory indices from all episodes
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # print("episode_idxs:",episode_idxs.shape)

        # Randomly select a time step t in each selected trajectory, t_samples is an array where each value indicates the selected time step in the corresponding trajectory, range [0, T-1]
        t_samples = np.random.randint(T, size=batch_size)
        # print("t_samples:",t_samples.shape)

        time_keys = ['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']

        # Extract sampled data
        transitions = {key: (episode_batch[key][episode_idxs, t_samples].copy() if key in time_keys
                    else episode_batch[key][episode_idxs].copy())
                    for key in episode_batch.keys()}

        # This is where substitute goal selection happens
        # her idx
        # If future_p is 0.8, it means 80% of samples will use substitute goals
        # her_indexes indicates the indices (i.e., samples) that need to replace the goal
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # Calculate the offset for future time steps
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)

        # Calculate the position of the future time step
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        # Replace the original goal g with the achieved_goal at the future time step
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # print("After rollout__batch_size:",rollout_batch_size)

        return transitions

    def dynamic_lambda(self, learning_step, success_rate=None):

        if learning_step is None:
            learning_step = 0

        # If success rate is provided, update learning rate
        if success_rate is not None:
            current_lr = self.adaptive_lr_manager.update_lambda_learning_rate(success_rate)
        else:
            current_lr = self.adaptive_lr_manager.current_lambda_lr

        # Use the adaptively adjusted learning rate to calculate lambda
        lambda_quality = self.lambda_starter_quality * math.pow(1 + current_lr, learning_step)

        return lambda_quality
    
    def sample_her_diversity_transitions(self, episode_batch, batch_size_in_transitions, 
                                         learning_step, lambda_val=None):
        # print("batch_size:",batch_size_in_transitions)
        # traceback.print_stack()
        T = episode_batch['actions'].shape[1]
        # print("T",T)

        # rollout_batch_size indicates the total number of trajectories
        rollout_batch_size = episode_batch['actions'].shape[0]
        # print(f"Currently there are {rollout_batch_size} episodes in temp_buffers that can be sampled")

        # batch_size refers to the number of samples to be sampled
        # print("batch_size_in_transitions:",batch_size_in_transitions)
        batch_size = batch_size_in_transitions

        diversity_weights = episode_batch['div'] / np.sum(episode_batch['div'])
        quality_weights = episode_batch['quality'] / np.sum(episode_batch['quality'])

        # Dynamic adjustment
        # lambda_quality = self.dynamic_lambda(learning_step)
        if lambda_val is not None:
            lambda_quality = lambda_val  # Use the provided value directly
        else:
            lambda_quality = self.dynamic_lambda(learning_step)  # Fallback calculation

        # Calculate combined weights
        combined_weights = diversity_weights + lambda_quality * quality_weights
        
        episode_batch['combined_weights'] = combined_weights

        # Normalize combined_weight
        combined_weights = combined_weights / np.sum(combined_weights)

        episode_idxs = np.random.choice(
            rollout_batch_size,
            size=batch_size,
            replace=True,
            p=combined_weights.flatten()
        )

        # Randomly select a time step t in each selected trajectory, t_samples is an array where each value indicates the selected time step in the corresponding trajectory, range [0, T-1]
        t_samples = np.random.randint(T, size=batch_size)
        # print("t_samples:",t_samples.shape)

        # Define keys with time dimension
        time_keys = ['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']

        # Extract sampled data
        transitions = {
            key: (episode_batch[key][episode_idxs, t_samples].copy() if key in time_keys 
                  else episode_batch[key][episode_idxs].copy())
            for key in episode_batch.keys() if key not in ['combined_weights']
        }

        # This is where substitute goal selection happens
        # her idx
        # If future_p is 0.8, it means 80% of samples will use substitute goals
        # her_indexes indicates the indices (i.e., samples) that need to replace the goal
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # Calculate the offset for future time steps
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)

        # Calculate the position of the future time step
        future_t = t_samples + 1 + future_offset

        # Convert episode_idx to numpy array
        episode_idxs = np.array(episode_idxs)

        # Get the trajectory indices that need to replace the goal
        selected_episode_idxs = episode_idxs[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][selected_episode_idxs, future_t[her_indexes]]

        # Replace the original goal g with the achieved_goal at the future time step
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # print("After rollout__batch_size:",rollout_batch_size)

        return transitions
    
    def sample_her_contrastive_transitions(self, episode_batch, batch_size_in_transitions, 
                                           learning_step, lambda_val=None):
        if not self.use_contrastive or self.trajectory_encoder is None or \
            len(episode_batch['obs']) < self.min_trajectories_for_contrastive:
            return self.sample_her_diversity_transitions(episode_batch, batch_size_in_transitions, learning_step)
        
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        if lambda_val is not None:
            current_lambda_val = lambda_val  # Use the provided value directly
        else:
            current_lambda_val = self.dynamic_lambda(learning_step)  # Fallback calculation

        # Encode trajectories and lambda
        with torch.no_grad():
            T = episode_batch['ag'].shape[1] - 1
            key_frames = [0, T//4, T//2, 3*T//4, T]
            
            # Prepare trajectory data -- select features according to goal_type
            if self.goal_type == 'full':
                trajectories = torch.tensor(
                    episode_batch['ag'][:, key_frames, :], 
                    dtype=torch.float32
                )
            elif self.goal_type == 'rotate':
                trajectories = torch.tensor(
                    episode_batch['ag'][:, key_frames, 3:], 
                    dtype=torch.float32
                )
            else:
                raise ValueError("Invalid goal_type. Choose 'full' or 'rotate'.")

            # trajectories = torch.tensor(episode_batch['ag'][:, :-1, :], dtype=torch.float32)
            lambda_tensor = torch.tensor([[current_lambda_val]] * rollout_batch_size, dtype=torch.float32)

            # Use GPU
            encoder_device = next(self.trajectory_encoder.parameters()).device
            if encoder_device.type == 'cuda':
                # Move to GPU
                trajectories_gpu = trajectories.cuda()
                lambda_tensor_gpu = lambda_tensor.cuda()
                
                # GPU computation
                encodings = self.trajectory_encoder(trajectories_gpu, lambda_tensor_gpu)
            
                # Move back to CPU
                scores = torch.norm(encodings, dim=1).cpu().numpy()
                scores = np.where(np.isnan(scores), 1e-8, scores)
            else:
                # CPU computation
                encodings = self.trajectory_encoder(trajectories, lambda_tensor)
                scores = torch.norm(encodings, dim=1).numpy()
                scores = np.where(np.isnan(scores), 1e-8, scores)

        # Normalization for the scores
        probability_scores = scores / np.sum(scores)

        # Sample trajectory indices based on probability
        episode_idxs = np.random.choice(
            rollout_batch_size,
            size=batch_size,
            replace=True,
            p=probability_scores.flatten()
        )

        t_samples = np.random.randint(T, size=batch_size)
        time_keys = ['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']
        
        transitions = {
            key: (episode_batch[key][episode_idxs, t_samples].copy() if key in time_keys 
                  else episode_batch[key][episode_idxs].copy())
            for key in episode_batch.keys() if key not in ['combined_weights']
        }
        
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = t_samples + 1 + future_offset
        episode_idxs = np.array(episode_idxs)
        selected_episode_idxs = episode_idxs[her_indexes]
        future_ag = episode_batch['ag'][selected_episode_idxs, future_t[her_indexes]]
        transitions['g'][her_indexes] = future_ag
        
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        
        return transitions

    def train_contrastive(self, episode_batch, learning_step, device=None, lambda_val=None):
        """
        Train the contrastive learning encoder
        
        Args:
        - episode_batch: experience replay buffer
        - device: training device (CPU/GPU)
        
        Returns:
        - contrastive_loss: contrastive loss value
        """
        if self.trajectory_encoder is None:
            return None
        
        if len(episode_batch['obs']) < self.min_trajectories_for_contrastive:
            return None
            
        # Get the number of trajectories
        rollout_batch_size = episode_batch['obs'].shape[0]

        max_trajectories_for_training = 2000
        
        if rollout_batch_size > max_trajectories_for_training:
            sampled_indices = np.random.choice(rollout_batch_size, max_trajectories_for_training, replace=False)
            
            # Create a sampled episode_batch
            sampled_episode_batch = {}
            for key in ['obs', 'ag', 'g', 'actions', 'div', 'quality']:
                if key in episode_batch:
                    sampled_episode_batch[key] = episode_batch[key][sampled_indices]
            
            # Use the sampled data
            episode_batch = sampled_episode_batch
            rollout_batch_size = max_trajectories_for_training
        
        # Calculate dynamic lambda
        if lambda_val is not None:
            current_lambda_val = lambda_val
        else:
            current_lambda_val = self.dynamic_lambda(learning_step)

        # Calculate the score for each trajectory
        diversity = episode_batch['div'].flatten()
        quality = episode_batch['quality'].flatten()
        scores = diversity + current_lambda_val * quality
        
        # Sort by score
        sorted_indices = np.argsort(scores)
        
        # Select top 30% and bottom 30% trajectories
        num_select = int(rollout_batch_size * 0.3)
        
        negative_indices = sorted_indices[:num_select]  # Low score trajectories
        positive_indices = sorted_indices[-num_select:]  # High score trajectories
        
        # If too few samples, return
        if len(positive_indices) < 2 or len(negative_indices) < 2:
            return None
            
        # Prepare data
        if self.goal_type == 'full':
            T = episode_batch['ag'].shape[1] - 1
            key_frames = [0, T//4, T//2, 3*T//4, T]
            
            positive_trajs = torch.tensor(
                episode_batch['ag'][positive_indices][:, key_frames, :], 
                dtype=torch.float32
            )
            negative_trajs = torch.tensor(
                episode_batch['ag'][negative_indices][:, key_frames, :], 
                dtype=torch.float32
            )

        elif self.goal_type == 'rotate':
            T = episode_batch['ag'].shape[1] - 1
            key_frames = [0, T//4, T//2, 3*T//4, T]
            
            positive_trajs = torch.tensor(
                episode_batch['ag'][positive_indices][:, key_frames, 3:], 
                dtype=torch.float32
            )
            negative_trajs = torch.tensor(
                episode_batch['ag'][negative_indices][:, key_frames, 3:], 
                dtype=torch.float32
            )

        # Get Current Lambda Information
        lambda_tensor = torch.tensor([[current_lambda_val]] * len(positive_indices), dtype=torch.float32)
        neg_lambda_tensor = torch.tensor([[current_lambda_val]] * len(negative_indices), dtype=torch.float32)
        
        # Use GPU
        encoder_device = next(self.trajectory_encoder.parameters()).device
        positive_trajs = positive_trajs.to(encoder_device)
        negative_trajs = negative_trajs.to(encoder_device)
        lambda_tensor = lambda_tensor.to(encoder_device)
        neg_lambda_tensor = neg_lambda_tensor.to(encoder_device)
            
        # Get encoding
        from rl_modules.trajectory_encoder import info_nce_loss
        
        positive_encodings = self.trajectory_encoder(positive_trajs, lambda_tensor)
        negative_encodings = self.trajectory_encoder(negative_trajs, neg_lambda_tensor)
        
        # Calculate contrastive loss
        contrastive_loss = info_nce_loss(positive_encodings, negative_encodings)

        positive_norms = torch.norm(positive_encodings, dim=1)
        negative_norms = torch.norm(negative_encodings, dim=1)

        margin = 0.5
        # L2-norm value for positive pairs are larger than negative pairs
        norm_loss = F.relu(negative_norms.mean() - positive_norms.mean() + margin)

        alpha = 0.5  # Weight for L2 norm loss
        total_loss = contrastive_loss + alpha * norm_loss
        
        return total_loss