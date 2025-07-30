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

        # 用于课程学习
        self.lambda_starter_quality=lambda_starter_quality
        self.learning_rate=learning_rate

        # Set goal_type
        self.goal_type = goal_type

        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

        # 对比学习所需的编码器会在agent初始化时创建
        self.trajectory_encoder = None

        # 是否使用对比学习进行采样
        self.use_contrastive = False

        # 对比学习最小样本数
        self.min_trajectories_for_contrastive = 50

        # 自适应lambda learning rate
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
        """获取学习率相关信息，用于调试和监控"""
        return self.adaptive_lr_manager.get_statistics()
    
    def adjust_thresholds_for_environment(self, env_name):
        """根据环境类型调整success rate阈值"""
        self.adaptive_lr_manager.adjust_thresholds_for_environment(env_name, self.goal_type)

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, learning_step=None):
        # print("batch_size:",batch_size_in_transitions)
        # traceback.print_stack()
        T = episode_batch['actions'].shape[1]

        # rollout_batch_size表示episode的总数
        # 指在一次采样过程中选中的轨迹数量
        rollout_batch_size = episode_batch['actions'].shape[0]

        # rollout_batch_size_obs = episode_batch['obs'].shape[0]
        # print("rollout_batch_size (actions):", rollout_batch_size)
        # print("rollout_batch_size (obs):", rollout_batch_size_obs)

        # batch_size指的是采样的样本数量
        # print("batch_size_in_transitions:",batch_size_in_transitions)
        batch_size = batch_size_in_transitions
        # print("Normalization batch_size:", batch_size)
        
        # 从所有episode中随机选择batch_size个轨迹的索引
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # print("episode_idxs:",episode_idxs.shape)

        # 在每条选中的trajectory中随机选择一个时间步t, 随机选择的时间步t, 范围为[0,T-1]
        # t_samples是一个数组，其中每个值表示对应轨迹中选定的时间步t
        t_samples = np.random.randint(T, size=batch_size)
        # print("t_samples:",t_samples.shape)

        time_keys=['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']

        # 提取采样数据
        transitions = {key: (episode_batch[key][episode_idxs, t_samples].copy() if key in time_keys
                    else episode_batch[key][episode_idxs].copy())
                    for key in episode_batch.keys()}

        #这里是进行替代目标的选择的地方
        # her idx
        # future_p假设为0.8，说明有80%的样本会使用替代目标
        # her_indexes表示需要替换目标的索引（即样本）
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # 计算未来时间步的偏移
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)

        # 计算未来时间步的位置
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        # 替换原始目标g为未来时间步的achieved_goal
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # print("After rollout__batch_size:",rollout_batch_size)

        return transitions

    def dynamic_lambda(self, learning_step, success_rate=None):

        if learning_step is None:
            learning_step = 0

        # 如果提供了success rate，更新学习率
        if success_rate is not None:
            current_lr = self.adaptive_lr_manager.update_lambda_learning_rate(success_rate)
        else:
            current_lr = self.adaptive_lr_manager.current_lambda_lr

        # 使用自适应调整后的学习率计算lambda
        lambda_quality = self.lambda_starter_quality * math.pow(1 + current_lr, learning_step)

        return lambda_quality
    
    def sample_her_diversity_transitions(self, episode_batch, batch_size_in_transitions, 
                                         learning_step, lambda_val=None):
        # print("batch_size:",batch_size_in_transitions)
        # traceback.print_stack()
        T = episode_batch['actions'].shape[1]
        # print("T",T)

        # rollout_batch_size表示trajectory的总数
        rollout_batch_size = episode_batch['actions'].shape[0]
        # print(f"当前存储(temp_buffers)里总共有{rollout_batch_size}条episode可以被采样")

        # batch_size指的是采样的样本数量
        # print("batch_size_in_transitions:",batch_size_in_transitions)
        batch_size = batch_size_in_transitions

        diversity_weights=episode_batch['div']/np.sum(episode_batch['div'])
        quality_weights=episode_batch['quality']/np.sum(episode_batch['quality'])

        # print("Raw diversity values:", episode_batch['div'])
        # print("Raw quality values:", episode_batch['quality'])
        # print("Normalized diversity_weights:", diversity_weights)
        # print("Normalized quality_weights:", quality_weights)

        # 动态调整
        # lambda_quality=self.dynamic_lambda(learning_step)
        if lambda_val is not None:
            lambda_quality = lambda_val  # 直接使用传入的值
        else:
            lambda_quality = self.dynamic_lambda(learning_step)  # 后备计算

        # 计算综合权重
        combined_weights = diversity_weights + lambda_quality * quality_weights
        episode_batch['combined_weights']=combined_weights

        # 对combined_weight进行归一化
        combined_weights = combined_weights / np.sum(combined_weights)

        # print(f"combined_weights: {combined_weights}")
        # print(f"diversity_weights: {diversity_weights}")
        # print(f"quality_weights: {quality_weights}")

        episode_idxs=np.random.choice(
            rollout_batch_size,
            size=batch_size,
            replace=True,
            p=combined_weights.flatten()
        )

        # 在每条选中的trajectory中随机选择一个时间步t, 随机选择的时间步t, 范围为[0,T-1]
        # t_samples是一个数组，其中每个值表示对应轨迹中选定的时间步t
        t_samples = np.random.randint(T, size=batch_size)
        # print("t_samples:",t_samples.shape)

        # 定义具有时间维度的键
        time_keys=['obs', 'ag', 'g', 'actions', 'obs_next', 'ag_next']

        # 提取采样数据
        transitions = {
            key: (episode_batch[key][episode_idxs, t_samples].copy() if key in time_keys 
                  else episode_batch[key][episode_idxs].copy())
            for key in episode_batch.keys() if key not in ['combined_weights']
        }

        # # 控制每次新增rollout_batch_size时绘制图像
        # if hasattr(self, 'previous_rollout_batch_size') and self.previous_rollout_batch_size != rollout_batch_size:
        #     # 绘制散点图：diversity vs quality
        #     plt.figure(figsize=(8, 6))
        #     plt.scatter(episode_batch['div'], episode_batch['quality'], c='blue', alpha=0.5)
        #     plt.title('Diversity vs Quality of HER Trajectories')
        #     plt.xlabel('Diversity')
        #     plt.ylabel('Quality')
        #     plt.show()
        
        # self.previous_rollout_batch_size=rollout_batch_size

        #这里是进行替代目标的选择的地方
        # her idx
        # future_p假设为0.8，说明有80%的样本会使用替代目标
        # her_indexes表示需要替换目标的索引（即样本）
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # 计算未来时间步的偏移
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)

        # 计算未来时间步的位置
        future_t = t_samples + 1 + future_offset

        # 将episode_idx转换为numpy数组
        episode_idxs=np.array(episode_idxs)

        # 获取需要替换目标的轨迹索引
        selected_episode_idxs=episode_idxs[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][selected_episode_idxs, future_t[her_indexes]]

        # 替换原始目标g为未来时间步的achieved_goal
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # print("After rollout__batch_size:",rollout_batch_size)

        return transitions
    
    def sample_her_contrastive_transitions(self, episode_batch, batch_size_in_transitions, 
                                           learning_step,lambda_val=None):
        if not self.use_contrastive or self.trajectory_encoder is None or \
            len(episode_batch['obs']) < self.min_trajectories_for_contrastive:
            return self.sample_her_diversity_transitions(episode_batch, batch_size_in_transitions, learning_step)
        
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size=batch_size_in_transitions

        if lambda_val is not None:
            current_lambda_val = lambda_val  # 直接使用传入的值
        else:
            current_lambda_val = self.dynamic_lambda(learning_step)  # 后备计算

        # 将轨迹和lambda编码
        with torch.no_grad():
            T = episode_batch['ag'].shape[1] - 1
            key_frames = [0, 12, 25, 37, 49]
            
            # 准备轨迹数据 -- 根据goal_type选择特征
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
                # 传输到GPU
                trajectories_gpu = trajectories.cuda()
                lambda_tensor_gpu = lambda_tensor.cuda()
                
                # GPU计算
                encodings = self.trajectory_encoder(trajectories_gpu, lambda_tensor_gpu)
            
                # 传回CPU
                scores = torch.norm(encodings, dim=1).cpu().numpy()
                scores = np.where(np.isnan(scores), 1e-8, scores)
            else:
                # CPU计算
                encodings = self.trajectory_encoder(trajectories, lambda_tensor)
                scores = torch.norm(encodings, dim=1).numpy()
                scores = np.where(np.isnan(scores), 1e-8, scores)

            # trajectories = trajectories.to(device)
            # lambda_tensor = lambda_tensor.to(device)
            
            # # 获取编码
            # encodings = self.trajectory_encoder(trajectories, lambda_tensor)

            # # 这里使用L2范数对encoding进行范化作为分数
            # scores = torch.norm(encodings, dim=1).cpu().numpy()

        # Normalization for the scores
        probability_scores = scores / np.sum(scores)

        # 基于概率采样trajectory索引
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
        训练对比学习编码器
        
        参数:
        - episode_batch: 经验回放缓冲区
        - device: 训练设备 (CPU/GPU)
        
        返回:
        - contrastive_loss: 对比损失值
        """
        if self.trajectory_encoder is None:
            return None
        
        if len(episode_batch['obs']) < self.min_trajectories_for_contrastive:
            return None
            
        # 获取轨迹数量
        rollout_batch_size = episode_batch['obs'].shape[0]

        max_trajectories_for_training = 2000
        
        if rollout_batch_size > max_trajectories_for_training:
            sampled_indices = np.random.choice(rollout_batch_size, max_trajectories_for_training, replace=False)
            
            # 创建采样后的episode_batch
            sampled_episode_batch = {}
            for key in ['obs', 'ag', 'g', 'actions', 'div', 'quality']:
                if key in episode_batch:
                    sampled_episode_batch[key] = episode_batch[key][sampled_indices]
            
            # 使用采样后的数据
            episode_batch = sampled_episode_batch
            rollout_batch_size = max_trajectories_for_training
        
        # 计算动态lambda
        if lambda_val is not None:
            current_lambda_val = lambda_val
        else:
            current_lambda_val = self.dynamic_lambda(learning_step)

        # 计算每条轨迹的分数
        diversity = episode_batch['div'].flatten()
        quality = episode_batch['quality'].flatten()
        scores = diversity + current_lambda_val * quality
        
        # 根据分数排序
        sorted_indices = np.argsort(scores)
        
        # 选择30%高分和30%低分轨迹
        num_select = int(rollout_batch_size * 0.3)
        negative_indices = sorted_indices[:num_select]  # 低分轨迹
        positive_indices = sorted_indices[-num_select:]  # 高分轨迹
        
        # 如果样本太少，返回
        if len(positive_indices) < 2 or len(negative_indices) < 2:
            return None
            
        # 准备数据
        if self.goal_type == 'full':
            T = episode_batch['ag'].shape[1] - 1
            key_frames = [0, 12, 25, 37, 49]
            
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
            key_frames = [0, 12, 25, 37, 49]
            
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
            
        # 获取编码
        from rl_modules.trajectory_encoder import info_nce_loss
        
        positive_encodings = self.trajectory_encoder(positive_trajs, lambda_tensor)
        negative_encodings = self.trajectory_encoder(negative_trajs, neg_lambda_tensor)
        
        # 计算对比损失
        contrastive_loss = info_nce_loss(positive_encodings, negative_encodings)

        positive_norms = torch.norm(positive_encodings, dim=1)
        negative_norms = torch.norm(negative_encodings, dim=1)

        margin = 0.5
        # L2-norm value for positive pairs are larger than negative pairs
        norm_loss = F.relu(negative_norms.mean() - positive_norms.mean() + margin)

        alpha = 0.5  # L2范数损失的权重
        total_loss = contrastive_loss + alpha * norm_loss
        
        return total_loss
