import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrajectoryEncoder(nn.Module):
    def __init__(self, goal_dim, hidden_dim=128, output_dim=64, goal_type='full'):
        """
        Trajectory encoder that encodes trajectories and curriculum stage λ into a unified representation space
        
        Parameters:
        - goal_dim: Complete dimension of the goal state
        - hidden_dim: LSTM hidden layer dimension
        - output_dim: Output encoding dimension
        - goal_type: Goal type, 'full' uses complete 7 dimensions, 'rotate' uses only quaternion part (4 dimensions)
        """
        super(TrajectoryEncoder, self).__init__()
        
        self.goal_type = goal_type
        
        # Determine actual input dimension based on goal_type
        if goal_type == 'full':
            self.input_dim = goal_dim  # Complete 7 dimensions
        elif goal_type == 'rotate':
            self.input_dim = 4  # Only use quaternion part (w,x,y,z)
        else:
            raise ValueError("goal_type must be either 'full' or 'rotate'")
            
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        # Layer for encoding lambda values
        self.lambda_encoder = nn.Linear(1, hidden_dim)
        
        # Fusion layer to combine trajectory encoding and lambda encoding
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
        Forward propagation
        
        Parameters:
        - trajectory: Trajectory achieved goals [batch_size, seq_len, goal_dim]
                     If goal_type='rotate', this should already be [batch_size, seq_len, 4]
        - lambda_val: Curriculum stage λ value [batch_size, 1]
        
        Returns:
        - encoding: Joint encoding of trajectory and λ [batch_size, output_dim]
        """
        # Ensure inputs are on the correct device
        trajectory = trajectory.to(self.device)
        lambda_val = lambda_val.to(self.device)
        
        # Ensure input dimensions are correct
        batch_size = trajectory.size(0)
        
        # Verify input dimensions match expectations
        expected_last_dim = self.input_dim
        if trajectory.size(-1) != expected_last_dim:
            raise ValueError(f"Expected trajectory last dimension to be {expected_last_dim} "
                           f"for goal_type='{self.goal_type}', but got {trajectory.size(-1)}")
        
        trajectory = torch.clamp(trajectory, min=-10, max=10)
        # print('trajectory:',trajectory)
        lambda_val=torch.clamp(lambda_val, min=0, max=10)
        
        # Encode trajectory
        _, (h_n, _) = self.lstm(trajectory)
        traj_encoding = h_n.squeeze(0)  # [batch_size, hidden_dim]
        traj_encoding=torch.clamp(traj_encoding, min=-5, max=5)
        
        # Encode lambda value
        lambda_encoding = self.lambda_encoder(lambda_val)  # [batch_size, hidden_dim]
        lambda_encoding = torch.clamp(lambda_encoding, min=-5, max=5)
        
        # Fuse the two encodings
        combined = torch.cat([traj_encoding, lambda_encoding], dim=1)
        encoding = self.fusion_layer(combined)
        encoding = torch.clamp(encoding, min=-10, max=10)
        
        return encoding

def info_nce_loss(positive_encodings, negative_encodings, temperature=0.1):
    """
    Calculate InfoNCE contrastive loss
    
    Parameters:
    - positive_encodings: Positive sample encodings [batch_size, encoding_dim]
    - negative_encodings: Negative sample encodings [batch_size, encoding_dim]
    - temperature: Temperature parameter controlling distribution smoothness
    
    Returns:
    - loss: Contrastive loss
    """
    # If there are not enough samples, return zero loss
    if positive_encodings.shape[0] == 0 or negative_encodings.shape[0] == 0:
        return torch.tensor(0.0, device=positive_encodings.device)
    
    batch_size = positive_encodings.shape[0]

    # Normalization
    pos_norm = F.normalize(positive_encodings, dim=1)
    neg_norm = F.normalize(negative_encodings, dim=1)
    
    # Create encoding matrix for all samples
    all_encodings = torch.cat([pos_norm, neg_norm], dim=0)
    
    # Calculate similarity matrix (dot product) and divide by temperature
    similarity_matrix = torch.matmul(pos_norm, all_encodings.T) / temperature
    
    # Create labels: for each positive sample, the first batch_size samples are positive samples (including itself)
    labels = torch.arange(batch_size, device=positive_encodings.device)
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss