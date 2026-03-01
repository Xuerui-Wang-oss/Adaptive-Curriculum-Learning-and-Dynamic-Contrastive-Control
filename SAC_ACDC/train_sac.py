import numpy as np
import gym
import os, sys
from arguments_sac import get_args
from mpi4py import MPI
from rl_modules.sac_agent import sac_agent
import random
import torch

"""
使用SAC+HER训练智能体
MPI部分代码来自openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""

def get_env_params(env):
    obs = env.reset()
    # 获取环境参数
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # 创建环境
    env = gym.make(args.env_name)
    
    # print(f"使用 SAC+HER 训练")
    # print(f"环境: {args.env_name}")
    # print(f"Goal类型: {args.goal_type}")
    # print(f"自动调整温度参数: {args.auto_alpha}")
    
    # 设置随机种子以保证可重现性
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env.reset()
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    
    # 获取环境参数
    env_params = get_env_params(env)
    
    # 创建SAC智能体
    sac_trainer = sac_agent(args, env, env_params)
    
    # 开始训练
    sac_trainer.learn()

if __name__ == '__main__':
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    # 获取参数
    args = get_args()
    
    # 启动训练
    launch(args)