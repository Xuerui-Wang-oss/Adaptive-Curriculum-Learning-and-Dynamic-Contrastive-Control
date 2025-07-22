import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrajectoryEncoder(nn.Module):
    def __init__(self, goal_dim, hidden_dim=128, output_dim=64, goal_type='full'):
        """
        轨迹编码器，将轨迹和课程阶段λ编码到一个统一的表示空间
        
        参数:
        - goal_dim: 目标状态的完整维度
        - hidden_dim: LSTM隐藏层维度
        - output_dim: 输出编码维度
        - goal_type: 目标类型，'full' 使用完整7维, 'rotate' 只使用四元数部分(4维)
        """
        super(TrajectoryEncoder, self).__init__()
        
        self.goal_type = goal_type
        
        # 根据goal_type确定实际输入维度
        if goal_type == 'full':
            self.input_dim = goal_dim  # 完整的7维
        elif goal_type == 'rotate':
            self.input_dim = 4  # 只使用四元数部分 (w,x,y,z)
        else:
            raise ValueError("goal_type must be either 'full' or 'rotate'")
            
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        # 用于编码lambda值的层
        self.lambda_encoder = nn.Linear(1, hidden_dim)
        
        # 融合层，将轨迹编码和lambda编码合并
        self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)
        
        # Add device tracking
        self.device = torch.device('cpu')
        
    def to(self, device):
        """Override to method to track device"""
        self.device = device
        return super().to(device)
        
    def cuda(self, device=None):
        """Override cuda method to track device"""
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f'cuda:{device}')
        return super().cuda(device)
        
    def cpu(self):
        """Override cpu method to track device"""
        self.device = torch.device('cpu')
        return super().cpu()
        
    def forward(self, trajectory, lambda_val):
        """
        前向传播
        
        参数:
        - trajectory: 轨迹achieved goals [batch_size, seq_len, goal_dim]
                     如果goal_type='rotate'，这里应该已经是[batch_size, seq_len, 4]
        - lambda_val: 课程阶段λ值 [batch_size, 1]
        
        返回:
        - encoding: 轨迹和λ的联合编码 [batch_size, output_dim]
        """
        # 确保输入在正确的设备上
        trajectory = trajectory.to(self.device)
        lambda_val = lambda_val.to(self.device)
        
        # 确保输入维度正确
        batch_size = trajectory.size(0)
        
        # 验证输入维度是否符合预期
        expected_last_dim = self.input_dim
        if trajectory.size(-1) != expected_last_dim:
            raise ValueError(f"Expected trajectory last dimension to be {expected_last_dim} "
                           f"for goal_type='{self.goal_type}', but got {trajectory.size(-1)}")
        
        trajectory = torch.clamp(trajectory, min=-10, max=10)
        # print('trajectory:',trajectory)
        lambda_val=torch.clamp(lambda_val, min=0, max=10)
        
        # 编码轨迹
        _, (h_n, _) = self.lstm(trajectory)
        traj_encoding = h_n.squeeze(0)  # [batch_size, hidden_dim]
        traj_encoding=torch.clamp(traj_encoding, min=-5, max=5)
        
        # 编码lambda值
        lambda_encoding = self.lambda_encoder(lambda_val)  # [batch_size, hidden_dim]
        lambda_encoding = torch.clamp(lambda_encoding, min=-5, max=5)
        
        # 融合两种编码
        combined = torch.cat([traj_encoding, lambda_encoding], dim=1)
        encoding = self.fusion_layer(combined)
        encoding = torch.clamp(encoding, min=-10, max=10)
        
        return encoding

def info_nce_loss(positive_encodings, negative_encodings, temperature=0.1):
    """
    计算InfoNCE对比损失
    
    参数:
    - positive_encodings: 正样本编码 [batch_size, encoding_dim]
    - negative_encodings: 负样本编码 [batch_size, encoding_dim]
    - temperature: 温度参数，控制分布的平滑度
    
    返回:
    - loss: 对比损失
    """
    # 如果没有足够的样本，返回零损失
    if positive_encodings.shape[0] == 0 or negative_encodings.shape[0] == 0:
        return torch.tensor(0.0, device=positive_encodings.device)
    
    batch_size = positive_encodings.shape[0]

    # Normalization
    pos_norm = F.normalize(positive_encodings, dim=1)
    neg_norm = F.normalize(negative_encodings, dim=1)
    
    # 创建所有样本的编码矩阵
    all_encodings = torch.cat([pos_norm, neg_norm], dim=0)
    
    # 计算相似度矩阵 (内积), 并除以温度
    similarity_matrix = torch.matmul(pos_norm, all_encodings.T) / temperature
    
    # 创建标签: 对于每个正样本，第一个batch_size个样本是正样本(包括自身)
    labels = torch.arange(batch_size, device=positive_encodings.device)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss