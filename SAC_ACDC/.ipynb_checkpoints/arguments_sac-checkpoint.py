import argparse

"""
SAC训练参数配置
"""

def get_args():
    parser = argparse.ArgumentParser()
    
    # 环境设置
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    
    # goal_type参数
    parser.add_argument('--goal-type', type=str, default='auto', 
                        help='goal type: "full" for FetchEnv and HandManipulateEgg, "rotate" for HandManipulateBlock and HandManipulatePen')
    
    # 训练超参数
    parser.add_argument('--n-epochs', type=int, default=200, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=50, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    
    # HER参数
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    
    # 缓冲区和采样
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')  # original is 256
    
    # SAC特有参数
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--lr-alpha', type=float, default=3e-4, help='the learning rate of the temperature parameter')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--polyak', type=float, default=0.995, help='the soft update coefficient')
    parser.add_argument('--alpha', type=float, default=0.2, help='initial temperature parameter (if not auto-tuned)')
    parser.add_argument('--auto-alpha', action='store_true', default=True, help='automatically tune temperature parameter')
    
    # 其他参数
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    
    # 保留的DDPG兼容参数（SAC中不使用，但保持向后兼容）
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps (not used in SAC)')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps (not used in SAC)')
    parser.add_argument('--action-l2', type=float, default=0.0, help='l2 reg (SAC默认不使用)')

    args = parser.parse_args()
    
    # 处理自动设置goal_type的情况
    if args.goal_type == 'auto':
        if 'Fetch' in args.env_name:
            args.goal_type = 'full'
        elif 'HandManipulateEgg' in args.env_name:
            args.goal_type = 'full'
        elif 'HandManipulateBlock' in args.env_name:
            args.goal_type = 'rotate'
        elif 'HandManipulatePen' in args.env_name:
            args.goal_type = 'rotate'
        elif 'HandReach' in args.env_name:
            args.goal_type = 'full'
        else:
            # 默认使用完整目标
            args.goal_type = 'full'

    return args