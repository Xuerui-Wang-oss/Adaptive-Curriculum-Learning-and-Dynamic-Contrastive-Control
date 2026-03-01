# ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation

## Environments
Here are the robotic manipulation environments used in our experiments:
![Environments](figure/environment.png) 
*(Note: Please replace `environment.png` with your actual image filename)*

## Requirements
- python=3.9.21
- openai-gym
- mujoco-py
- pytorch
- mpi4py
- mujoco210

## Instruction to run the code
If you want to use GPU, just add the flag `--cuda` **(Not Recommended, Better Use CPU)**.
If you only want to use GPU for LSTM in contrastive learning, just add the flag `--hybrid-gpu` **(Not Recommended, Better Use CPU)**.

1. train the **FetchPush-v1**:
```bash
mpirun -np 16 python -u train.py --env-name='FetchPush-v1' 2>&1 | tee push.log
