import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

"""
SAC网络结构
输入 x 应该是 [o, g]，其中 o 是观察，g 是目标
"""

class actor(nn.Module):
    """SAC Actor网络 - 随机策略"""
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.action_dim = env_params['action']
        
        # 网络层
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        # 输出层：均值和log_std
        self.mean_layer = nn.Linear(256, env_params['action'])
        self.log_std_layer = nn.Linear(256, env_params['action'])
        
        # 动作缩放
        self.action_scale = self.max_action
        self.action_bias = 0.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        """采样动作 - sac_agent.py中调用此方法"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = distributions.Normal(mean, std)
        
        # 重参数化技巧
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # 计算log概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh的雅可比行列式
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 确定性动作（用于评估）
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action

class critic(nn.Module):
    """SAC Critic网络 - 双Q网络"""
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        
        # Q1网络
        self.q1_fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)
        
        # Q2网络
        self.q2_fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        """sac_agent.py中调用此方法，需要返回(q1, q2)"""
        # 归一化动作并拼接
        xu = torch.cat([x, actions / self.max_action], dim=1)
        
        # Q1网络
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        x1 = F.relu(self.q1_fc3(x1))
        q1 = self.q1_out(x1)
        
        # Q2网络
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        x2 = F.relu(self.q2_fc3(x2))
        q2 = self.q2_out(x2)
        
        return q1, q2
    
    def Q1(self, x, actions):
        """只返回Q1值（用于某些特殊情况）"""
        xu = torch.cat([x, actions / self.max_action], dim=1)
        
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        x1 = F.relu(self.q1_fc3(x1))
        q1 = self.q1_out(x1)
        
        return q1