# ACDC: Adaptive Curriculum and Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning

## Requirements
- python=3.9.21
- openai-gym
- mujoco-py
- pytorch
- mpi4py
- mujoco210

## Instruction to run the code
If you want to use GPU, just add the flag `--cuda` **(Not Recommended, Better Use CPU)**.
If you want to only use GPU for LSTM in contrastive learning, just add the flag '--hybrid-gpu'**(Not Recommended, Better Use CPU)**.

1. train the **FetchPush-v1**:
```bash
mpirun -np 16 python -u train.py --env-name='FetchPush-v1' 2>&1 | tee push.log
```
2. train the **FetchPickAndPlace-v1**:
```bash
mpirun -np 16 python -u train.py --env-name='FetchPickAndPlace-v1' 2>&1 | tee pick.log
```
3. train the **HandManipulateBlockFull-v0**:
```bash
mpirun -np 16 python -u train.py --env-name='HandManipulateBlockFull-v0' --goal-type='full' 2>&1 | tee block.log
```
4. train the **HandManipulateEggFull-v0**:
```bash
mpirun -np 16 python -u train.py --env-name='HandManipulateEggFull-v0' --goal-type='full' 2>&1 | tee egg.log
```
5. train the **HandManipulatePenRotate-v0**:
```bash
mpirun -np 16 python -u train.py --env-name='HandManipulatePenRotate-v0' --goal-type='rotate' 2>&1 | tee pen.log
```
6. train the **HandReach-v0**:
```bash
mpirun -np 16 python -u train.py --env-name='HandReach-v0' --goal-type='full' 2>&1 | tee reach.log
```
### Play Demo
```bash
python demo.py --env-name=<environment name>
```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1dNzIpIcL4x1im8dJcUyNO30m_lhzO9K4), then put the `saved_models` under the current folder.

## Results
### Training Performance
It was plotted by using 5 different seeds, the solid line is the median value. 
![Training_Curve](figure/Benchmark_result.png)
### Demo:
**Tips**: when you watch the demo, you can press **TAB** to switch the camera in the mujoco.  
