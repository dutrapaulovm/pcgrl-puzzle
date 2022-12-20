import gym
import os
import pcgrl
import time
import numpy as np
from pcgrl.Utils import *
from pcgrl.wrappers import OneHotEncodingWrapper, RGBToGrayScaleObservationWrapper, SegmentWrapper
from pcgrl.BasePCGRLEnv import Experiment

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("minimap-narrow-puzzle-v0")
env = SegmentWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_mazecoin")

model = PPO.load("ppo_mazecoin")

obs = env.reset()
while True:        
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)    
    env.render()

"""        
env.save_image_map = False
env.teste = 0
env.reset()
st = time.time()
env.render()
while(True):
  clear_console()
  action = env.action_space.sample()
  ob, rew, done, info = env.step(action)
  time.sleep(0.1)  
  env.render()
  if done:    
    env.reset()
"""        