# PCGRL-PUZZLE
 
 Project to [OpenAI GYM](https://gym.openai.com/) for Procedural Content Generation using Reinforcement Learning

The initial results is covered in the paper: [Procedural Content Generation using Reinforcement  Learning and Entropy Measure as Feedback](https://doi.org/10.1109/SBGAMES56371.2022.9961076)

## Python packages

It is recommended to use a virtual environment using anaconda or similar.

To install the required python packages, run
`python -m pip install -r requirements.txt`

## Installation
1. Clone this repository to your local machine.
2. To install the package, `run pip install -e .` from the repository folder. The OpenAI GYM environment, Stable Baselines3 and SB3 Contrib will install automatically. 

## Running 
Another way is to use [Conda](https://www.anaconda.com/) by creating a virtual environment then activating it and installing all the dependencies
```sh
conda create --name pcgrl-puzzle
conda activate pcgrl-puzzle
pip install gym
pip install stable-baselines3==1.6.2
pip install sb3-contrib==1.6.2
pip install pygame==2.0.0 
pip install ipython
pip install opencv-python
pip install imageio
pip install pandas
cd pcgrl-puzzle
pip install -e .
```
## How to use

PCGRL-PUZZLE has some registered environments
```python
from gym import envs
import pcgrl

[env.id for env in envs.registry.all() if "pcgrl-puzzle" in env.entry_point]
```

## Included Environments

The environments listed below are implemented in the [pcgrl-puzzle](/pcgrl-puzzle) directory and are registered with OpenAI gym.

### Default environments

1. 'mazecoin-narrow-puzzle-2x3-v0' 
2. 'mazecoin-narrow-puzzle-2x3-v1', 
3. 'mazecoin-narrow-puzzle-2x3-v2',
4. 'dungeon-narrow-puzzle-2x3-v0' 
5. 'zelda-narrow-puzzle-2x3-v0'

### Simple example

```python
import gym
import pcgrl

env = gym.make('mazecoin-narrow-puzzle-2x3-v0')
obs = env.reset()
t = 0
while t < 1000:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  print(obs)
  print(reward)
  env.render()  
  if done:
    print("Episode finished: {} timesteps".format(t+1))
    break
  t += 1
```
