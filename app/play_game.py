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
from gym.wrappers import GrayScaleObservation, ResizeObservation
#from gym.envs.classic_control import rendering
from gym.utils import seeding
from pcgrl.Utils import clear_console

from pcgrl.mazecoin import MazeCoinGameProblem
from pcgrl.utils.experiment import SaveOnBestTrainingRewardCallback
from pcgrl.wrappers import ScaledFloatFrame

from stable_baselines3 import PPO,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame, MaxAndSkipEnv, NoopResetEnv

import app
from mazecoinplay_env import MazeCoinPlayEnv, TrainAndLoggingCallback

if __name__ == "__main__":
    
    path_log = os.path.dirname(os.path.realpath(__file__))  
    CHECKPOINT_DIR = "{}{}".format(path_log, '/train/')
    LOG_DIR = "{}{}".format(path_log, '/logs/')
    MONITOR_DIR = "{}{}{}".format(path_log, '/logs/', "monitor.csv")

    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)    

    steps = 100
    env = gym.make("mazecoinplay-v0") 
    env = WarpFrame(env)           
    env = ScaledFloatFrame(env)         
    env = ClipRewardEnv(env)            
    env = Monitor(env, filename="monitor.csv")    
    env = DummyVecEnv([lambda :env])    
    env = VecFrameStack(env, 4, channels_order='last')    
    
    try:                
        
        path_model = f"{CHECKPOINT_DIR}/best_model"        
        print(f"Loading {path_model}") 
        model = PPO.load(path_model)
        print("Model is loader...")                                                                                                                                      

        time_elapsed_agents = []

        start_agent = timer()
        print("Start: ", start_agent)
        print()   

        for episode in range(steps): 
            obs = env.reset()
            done = False
            total_reward = 0
            while not done: 
                action, _ = model.predict(obs)
                print(action)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            print('Total Reward for episode {} is {}'.format(episode, total_reward))
            #time.sleep(2)    

        end_agent = timer()                

        print("End: ", end_agent)
        time_ela_agent = timedelta(seconds=end_agent-start_agent)
        print("Time elapsed: ", time_ela_agent) 

        d = {"start": start_agent, "end" : end_agent, "time elapsed": time_ela_agent}
        
        time_elapsed_agents.append(d)                       
                        
        df = pd.DataFrame(time_elapsed_agents)
        filename_timeela = "{}/{}.csv".format(path_log, "/Inference Time elapsed") 
        df.to_csv(filename_timeela,  index=False)                        
        
        env.close()  

    except KeyboardInterrupt:              
        pass
    finally:              
        try:
            pass
            # model.env.close()
        except EOFError:
            pass                   

    