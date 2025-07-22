import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent_contrastive import ddpg_agent
import random
import torch

"""
Train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""

def get_env_params(env):
    # old gym version
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def determine_goal_type(env_name):
    """根据环境名称确定goal_type"""
    if 'Fetch' in env_name:
        return 'full'
    elif 'HandManipulateEggFull-v0' in env_name:
        return 'full'
    elif 'HandManipulateBlockRotateXYZ-v0' in env_name:
        return 'rotate'
    elif 'HandManipulatePenRotate-v0' in env_name:
        return 'rotate'
    else:
        # 默认使用完整目标
        return 'full'

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    
    # 确定goal_type
    if not hasattr(args, 'goal_type'):
        args.goal_type = determine_goal_type(args.env_name)
    
    # print(f"Using goal_type: {args.goal_type} for environment: {args.env_name}")
    
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    
    # get the environment parameters
    env_params = get_env_params(env)
    
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    # get the params
    args = get_args()
    
    # launch training
    launch(args)