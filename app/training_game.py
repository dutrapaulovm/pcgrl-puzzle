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
from pcgrl.wrappers import ExperimentMonitor, MaxStep, RGBToGrayScaleObservationWrapper, ScaledFloatFrame, SVDObservationWrapper, ImageToPyTorch, CV2ImgShowWrapper, BufferWrapper

from stable_baselines3 import PPO,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame, MaxAndSkipEnv, NoopResetEnv

# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker
from gym.wrappers import GrayScaleObservation, ResizeObservation

import app
from custom_policy import CustomCNN
from mazecoinplay_env import MazeCoinPlayEnv, TrainAndLoggingCallback
import torch as th

if __name__ == "__main__":
    
    path_log = os.path.dirname(os.path.realpath(__file__))  
    CHECKPOINT_DIR = "{}{}".format(path_log, '/train/')
    LOG_DIR = "{}{}".format(path_log, '/logs/')

    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    
    steps = 100
    env = gym.make("mazecoinplay-v0")        
    #env = MaxAndSkipEnv(env, skip=4)     
    env = WarpFrame(env)           
    env = ScaledFloatFrame(env)    
    #env = CV2ImgShowWrapper(env)   
    env = ClipRewardEnv(env)            
    env = Monitor(env)    
    env = DummyVecEnv([lambda :env])    
    env = VecFrameStack(env, 4, channels_order='last')    
    
    try:

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )        

        seed = 42       
        time_elapsed_agents = []
        set_random_seed (seed)
        #model = DQN('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=1e6)#,buffer_size=1200000, learning_starts=1000)

        #policy_kwargs = dict(net_arch = [64, 64], activation_fn=th.nn.Sigmoid)
        #model = PPO('CnnPolicy', env, verbose=1, seed=seed, policy_kwargs=policy_kwargs, learning_rate=0.000001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
        model = PPO('CnnPolicy', env, verbose=1, seed=seed,  learning_rate=2.5e-4, n_steps=8192) #, clip_range=.1, gamma=.95, gae_lambda=.9)

        #saveOnBest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=CHECKPOINT_DIR)                                                                    

        start_agent = timer()
        print("Start: ", start_agent)
        print()

        model.learn(total_timesteps=1000000, callback=callback)

        end_agent = timer()        
        print("End: ", end_agent)
        time_ela_agent = timedelta(seconds=end_agent-start_agent)
        print("Time elapsed: ", time_ela_agent)     

        d = {"start": start_agent, "end" : end_agent, "time elapsed": time_ela_agent}
        
        time_elapsed_agents.append(d)                       
                        
        df = pd.DataFrame(time_elapsed_agents)
        filename_timeela = "{}/{}.csv".format(path_log, "/Training Time elapsed") 
        df.to_csv(filename_timeela,  index=False)                                   

        path_model = f"{CHECKPOINT_DIR}/trainedmodel"
        print(f"Saving to {path_model}")
        model.save(path_model)

        env.close()

    except KeyboardInterrupt:              
        pass
    finally:              
        try:
            pass
            # model.env.close()
        except EOFError:
            pass