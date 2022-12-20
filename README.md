# PCGRL-PUZZLE
 
 Project to [OpenAI GYM](https://gym.openai.com/) for Procedural Content Generation using Reinforcement Learning

The initial results is covered in the paper: [Procedural Content Generation using Reinforcement  Learning and Entropy Measure as Feedback](https://doi.org/10.1109/SBGAMES56371.2022.9961076)

## Python packages

It is recommended to use a virtual environment using anaconda or similar.

To install the required python packages, run
`python -m pip install -r requirements.txt`

## Installation

To install the base Gym library, use `pip install gym`.

## Running 
Another way is to use [Conda](https://www.anaconda.com/) by creating a virtual environment then activating it and installing all the dependencies
```sh
conda create --name pcgrl-puzzle
conda activate pcgrl-puzzle
pip install gym
pip install stable-baselines3
pip install sb3-contrib
pip install pygame==2.0.0 
pip install ipython
pip install opencv-python
pip install imageio
pip install pandas
cd pcgrl-puzzle
pip install -e .
```