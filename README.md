# PCGRL-PUZZLE
 
 Project to [OpenAI GYM](https://gym.openai.com/) for Procedural Content Generation using Reinforcement Learning

The initial results is covered in the paper: [Procedural Content Generation using Reinforcement  Learning and Entropy Measure as Feedback](https://doi.org/10.1109/SBGAMES56371.2022.9961076)

## Python packages

All the experiments has been tested using Python 3.7.2

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
5. 'dungeon-narrow-puzzle-2x3-v1' 
6. 'dungeon-narrow-puzzle-2x3-v2' 
7. 'zelda-narrow-puzzle-2x3-v0'
8. 'zelda-narrow-puzzle-2x3-v1'
9. 'zelda-narrow-puzzle-2x3-v2'

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

## Traning and Inference

You can train and inference using the [experiment.py](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/pcgrl/utils/experiment.py). You can configure the train and inference for different environments and agents. A  complete code you can see in the file [test_experiment.py](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/pcgrl/utils/test_experiment.py).

Here you can see a simple code for traning agents: [simple_trainv1.py](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/simple_trainv1.py) and [simple_trainv2.py](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/simple_trainv2.py). This code use Stable Baselines3 [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) algorithm for training. 

### [Simple training - V1](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/simple_trainv1.py)
```python
from pcgrl import *
from pcgrl.wrappers import *
from stable_baselines3 import PPO

if __name__ == '__main__':
    
    env = gym.make('mazecoin-narrow-puzzle-2x3-v0')
    env = SegmentWrapper(env)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_mazecoin")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_mazecoin")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
```

### [Simple training - V2](https://github.com/dutrapaulovm/pcgrl-puzzle/blob/main/simple_trainv2.py)
```python
from pcgrl import *
from pcgrl.wrappers import *
from stable_baselines3 import PPO

import torch as th

if __name__ == '__main__':
    
    env = gym.make('mazecoin-narrow-puzzle-2x3-v0')
    env = SegmentWrapper(env)

    #PPO parameters
    total_timesteps = 1000
    learning_rate   = 3e-4
    n_steps         = 2048
    gamma           = 0.99
    batch_size      = 64
    n_epochs        = 10
    activation_fn   = th.nn.Sigmoid
    policy_kwargs   = dict(net_arch = [dict(pi=[64, 64], vf=[64, 64])], activation_fn=activation_fn)

    model = PPO("MlpPolicy", env  = env,     
                            gamma = gamma,         
                            batch_size = batch_size,
                            n_epochs = n_epochs,   
                            n_steps = n_steps,                          
                            learning_rate = learning_rate,       
                            policy_kwargs = policy_kwargs, verbose=1)

    model.learn(total_timesteps=25000)
    model.save("ppo_mazecoin")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_mazecoin")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
```