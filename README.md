# PCGRL
 Projeto utilizando Gym Open AI para aprendizado por reforço

## Python packages

It is recommended to use a virtual environment using anaconda or similar.

To install the required python packages, run
`python -m pip install -r requirements.txt`

## Installation

To install the base Gym library, use `pip install gym`.

## Running 
Another way is to use [Conda](https://www.anaconda.com/) by creating a virtual environment then activating it and installing all the dependencies
```sh
conda create --name pcgrl
conda activate pcgrl
pip install tensorflow==1.15
pip install stable-baselines==2.9.0
cd gym_pcgrl
pip install -e .
cd ..
python train.py
```
Lastly, you can just install directly without using any virtual environment:
```sh
pip install tensorflow==1.15
pip install stable-baselines==2.9.0
cd gym_pcgrl
pip install -e .
cd ..
python train.py
```


## Wrappers

There are a variery of wrappers to change the observation

MiniGrid is built to support tasks involving natural language and sparse rewards.
The observations are dictionaries, with an 'image' field, partially observable
view of the environment, a 'mission' field which is a textual string
describing the objective the agent should reach to get a reward, and a 'direction'
field which can be used as an optional compass. Using dictionaries makes it
easy for you to add additional information to observations
if you need to, without having to encode everything into a single tensor.

There are a variery of wrappers to change the observation format available in [gym_minigrid/wrappers.py](/gym_minigrid/wrappers.py). If your RL code expects one single tensor for observations, take a look at
`FlatObsWrapper`. There is also an `ImgObsWrapper` that gets rid of the 'mission' field in observations,
leaving only the image field tensor.

Please note that the default observation format is a partially observable view of the environment using a
compact and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead,
use the `RGBImgPartialObsWrapper`. You can use it as follows:

```
from gym_minigrid.wrappers import *
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs = env.reset() # This now produces an RGB tensor only
```