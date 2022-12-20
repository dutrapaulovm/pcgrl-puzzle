import gym
import os
import pcgrl
import time
import numpy as np
from pcgrl.maze.MazeEnv import MazeEnv
from pcgrl.Utils import *

env = MazeEnv()#gym.make("maze-puzzle-v0")
print("Observation Space: ", env.observation_space)
print("Action Space: ", env.action_space)
env.save_logger = True
env.show_logger = True
env.path = os.path.dirname(__file__) + "/info"
env.reset()
while(True):  
  action = env.action_space.sample()
  ob, rew, done, info = env.step(action)      
  env.render()
  if done:    
    env.reset()    
    time.sleep(2)