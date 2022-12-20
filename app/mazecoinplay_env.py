#MSS used for screen cap
import pandas as pd
from datetime import timedelta
import time     
from timeit import default_timer as timer
from datetime import timedelta

#from mss import mss
import math
import os
import numpy as np

import csv
#import pytesseract
#Sending commands
#import pyautogui
#import pydirectinput            

from matplotlib import pyplot as plt
import time
from PIL import Image
import gym
from gym.spaces import Discrete, Box
#from gym.envs.classic_control import rendering
from gym.utils import seeding
from pcgrl.Utils import clear_console

from pcgrl.mazecoin import MazeCoinGameProblem
from pcgrl.utils.experiment import SaveOnBestTrainingRewardCallback
from pcgrl.wrappers import ExperimentMonitor, MaxStep, RGBToGrayScaleObservationWrapper

from stable_baselines3 import PPO,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker
from gym.wrappers import GrayScaleObservation, ResizeObservation


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model') #_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


#SCREEN_WIDTH  = 208 * 8
#SCREEN_HEIGHT = 144 * 8

#SCREEN_WIDTH  = 52 * 8
#SCREEN_HEIGHT = 36 * 8

#SCREEN_SHAPE = 1, SCREEN_HEIGHT, SCREEN_WIDTH
#SCREEN_SHAPE = SCREEN_HEIGHT, SCREEN_WIDTH, 3

action_map = {
    #0 : 'no_op',
    0 : 'right',
    1 : 'left',
    2 : 'down',
    3 : 'up'    
}

class MazeCoinPlayEnv(gym.Env):

    """
      Actions:
        There are 5 discrete deterministic actions:
        - 2: move Down
        - 3: move Up
        - 1: move Left
        - 0: move Right
        - 4: stopped
    """
    def __init__(self, path_level = None, tile_size = 16, board = (3, 2), piece_size = 8, border = True):
        super(MazeCoinPlayEnv).__init__()        
        
        self.path_level = path_level
        if (self.path_level is None):
            self.path_level = os.path.dirname(os.path.realpath(__file__))                
            self.path_level = "{}\\{}\\{}".format(self.path_level, "mazecoin", "Map200.csv")
        
        offset_border = 0

        if (border):
            offset_border = 2

        self.cols = (board[0] * min(8, piece_size * 2)) + offset_border
        self.rows = (board[1] * min(8, piece_size * 2)) + offset_border

        self.game = MazeCoinGameProblem(cols = self.cols, rows = self.rows, tile_size=tile_size)
        self.game.show_game_hud = True
        self.game.env = self

        if (not self.path_level is None and os.path.exists(self.path_level)):
            self.game.load_map(self.path_level)                   

        a = list(action_map.keys())
        w = int(self.cols * tile_size)
        h = int(self.rows * tile_size)
        screen_shape = (w, h, 3) 
        self.action_space = Discrete(len(a))        
        self.observation_space = Box(low=0, high=255, dtype=np.uint8, shape=screen_shape)
        self.done = False
        self.episode = 0
        self.max_steps = ((len(a) * self.cols) * (self.rows * len(a))) * 2
        self.n_steps   = 0
        self.total_success = 0
        
        # setup a placeholder for a 'human' render mode viewer
        self.viewer = None

        # Penalties and Rewards
        self.penalty_for_step = -0.1                
        self.penalty_finished = -20
        self.reward_finished = 20        
        self.reward_last = 0

        # setup a done flag
        self.done = True
        self._cumulative_reward = 0  
        self._reward = 0  

        self.screen = self.game.render()

    def add_reward(self, reward):
        """[summary]
        Increments the rewards        
        Args:
            reward ([float]): [Value reward to increment]
        """
        self._cumulative_reward += reward
        self._reward += reward
        self._mean_reward = 0
        self.last_action = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):        
        
        self.done = False
        self.n_steps += 1        

        clear_console()        
          
        reward, self.done = self.game.step(action+1)

        bonus = (1 / math.sqrt((self.n_steps / self.max_steps))) * math.pi

        if (self.done):
            reward *= (1+bonus)
            self.total_success += 1
        elif (self.n_steps >= self.max_steps):            
            self.done = True
            reward += self.penalty_finished

        info = {}

        info["episode"]  = self.episode
        info["reward"]   = reward
        info["action"]   = action_map[action]
        info["steps"]    = self.n_steps
        info["max_iterations"] = self.max_steps
        info["bonus"]     = 1+bonus
        print(info)
        
        self.add_reward(reward)

        print("Cumulative Reward: ", self._cumulative_reward)
        print("Success: ", self.total_success)
       
        self.screen = self.render()

        return self.screen, reward, self.done, {}

    def reset(self):            
        self.episode += 1
        self.n_steps = 0
        if (not self.path_level is None and os.path.exists(self.path_level)):
            self.game.load_map(self.path_level)                    
        self.screen = self.game.render()             
        return self.screen 

    def render(self, mode='human'):                
        if mode == 'rgb_array':
            self.screen = self.game.render_rgb()
        elif mode == 'human':            
            self.screen = self.game.render()       
        return self.screen

if __name__ == "__main__":    
    
    path_level = os.path.dirname(os.path.realpath(__file__))                
    path_level = "{}\\{}\\{}".format(path_level, "mazecoin", "Map200.csv")

    game = MazeCoinGameProblem(cols = 26, rows = 18, tile_size=16)
    game.load_map(path_level)     
    while True:
        game.update()
        game.render()

    """
    env = MazeCoinPlayEnv()
    done = False
    env.reset()
    while True:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)              
        env.render()
    """    