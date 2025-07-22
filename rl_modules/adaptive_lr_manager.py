import numpy as np
from collections import deque

class AdaptiveLearningRateManager:
    def __init__(self, base_lambda_lr=0.01, smoothing_factor=0.7, 
                 low_threshold=0.3, high_threshold=0.6):
        
        self.base_lambda_lr = base_lambda_lr
        self.smoothing_factor = smoothing_factor
        self.current_lambda_lr = base_lambda_lr
        
        # 阈值设置（可根据环境调整）
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # 记录历史信息用于调试
        self.success_rate_history = deque(maxlen=50)
        self.lambda_lr_history = deque(maxlen=50)
        
    def update_lambda_learning_rate(self, success_rate):

        self.success_rate_history.append(success_rate)
        
        # 根据success rate确定目标学习率
        if success_rate < self.low_threshold:
            # 复杂环境：减慢lambda增长，更注重diversity
            target_lr = self.base_lambda_lr * 0.5
        elif success_rate <= self.high_threshold:
            # 中等环境：保持基础学习率
            target_lr = self.base_lambda_lr
        else:
            # 简单环境：加快lambda增长，更注重quality
            target_lr = self.base_lambda_lr * 2.0
        
        # 平滑更新：current = 0.3 * target + 0.7 * last
        self.current_lambda_lr = (1 - self.smoothing_factor) * target_lr + \
                                 self.smoothing_factor * self.current_lambda_lr
        
        # 记录历史
        self.lambda_lr_history.append(self.current_lambda_lr)
        
        return self.current_lambda_lr
    
    def get_statistics(self):
        if len(self.success_rate_history) == 0:
            return {"current_lr": self.current_lambda_lr, "success_rate": None}
            
        return {
           "current_lambda_lr": self.current_lambda_lr,
            "base_lambda_lr": self.base_lambda_lr,
            "latest_success_rate": self.success_rate_history[-1],
            "avg_success_rate": np.mean(list(self.success_rate_history)),
            "success_rate_std": np.std(list(self.success_rate_history)),
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "env_info": f"{self.current_env_name}({self.current_goal_type})" if self.current_env_name else "Not set"
        }
    
    def adjust_thresholds_for_environment(self, env_name, goal_type='full'):

        # Record current env and goal_type
        self.current_env_name = env_name
        self.current_goal_type = goal_type
        # print(f"[AdaptiveLR] Adjusting thresholds for {env_name} with goal_type={goal_type}")

        if 'Fetch' in env_name:
            self.low_threshold = 0.3
            self.high_threshold = 0.65
        
        elif 'Hand' in env_name:
            if goal_type=='full':
                # Default Threshold
                self.low_threshold = 0.3
                self.high_threshold = 0.65

            elif goal_type=='rotate':
                # Default Threshold
                self.low_threshold = 0.25
                self.high_threshold = 0.35

            else:
                self.low_threshold = 0.2
                self.high_threshold = 0.4
            
        else:
            self.low_threshold = 0.3
            self.high_threshold = 0.6
        
        # print(f"low threshold: {self.low_threshold}, high threshold: {self.high_threshold}")