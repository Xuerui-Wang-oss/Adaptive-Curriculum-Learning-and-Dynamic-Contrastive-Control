import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

example_map = [[1, 1, 1, 1, 1],
       [1, C, 0, C, 1],
       [1, 1, 1, 1, 1]]

env = gym.make('AntMaze_UMaze-v5', maze_map=example_map)