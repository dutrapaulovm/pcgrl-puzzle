import csv
import json
import cv2
import pandas as pad
import math
import gym
import numpy as np
import pandas as pd
import os
import time
import hashlib
from enum import Enum
from matplotlib.image import imread
import matplotlib.pyplot as plt
from collections import OrderedDict
from gym import spaces
from gym.core import Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper 
from pcgrl.Utils import * 

class EnvInfo(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.info_maps = []
        self.counter_best  = 0
        self.counter_worst = 0

    def step(self, action):
        
        print("\033[H\033[J", end="")
        
        obs, reward, done, info = self.env.step(action)
        
        stats = self.env.current_stats
        
        reward = self.env.reward
                
        aux_info = {}                                                
        aux_info["iterations"] = self.env.iterations        
        aux_info["counter_done"] = self.env.counter_done
        aux_info["counter_max_changes"] = self.env.counter_done_max_changes
        aux_info["counter_iterations_done"] = self.env.counter_done_interations
        aux_info["changes"] = self.env.changes
        aux_info["max_iterations"] = self.env.max_iterations
        aux_info["max_changes"] = self.env.max_changes                        
        aux_info["actions"] = action
        aux_info["reward"] = self.env.reward
        aux_info["done"] = self.env.is_done        
        aux_info["maps_stats"] = stats["map_stats"]
        aux_info["reward_info"] = self.env.reward_info                       
        
        if (self.env.is_done) :            
            self.env.render()            
            if (reward > 0):    
                self.counter_best += 1                            
                path = self.path + "/best/MapEnvTraining"+str(self.counter_done)+".png"                        
                self.game.save_screen(path)
            else:                     
                self.counter_worst += 1                       
                path = self.path + "/worst/MapEnvTraining"+str(self.counter_done)+".png"                        
                self.game.save_screen(path)
            
            self.env.counter_done += 1
            aux_info["counter_done"] = self.env.counter_done
            
            info_countermap_row = {"Best" :self.counter_best,
                                  "Worst":self.counter_worst}
                        
            info_counter_maps = []
            info_counter_maps.append(info_countermap_row)                            
            
            df = pad.DataFrame(info_counter_maps)
            df.to_csv(self.env.path + "/InfoCounterMaps.csv")                
            
            self.info_maps.append(aux_info)            
            df = pad.DataFrame(self.info_maps)
            df.to_csv(self.env.path + "/Info.csv")                            
            
            df = pad.DataFrame(self.env.game.map)
            df.to_csv(self.env.path + "/map/Map"+str(self.env.counter_done)+".csv", header=False, index=False)              
        
        print(self.env.game.get_info())
        print("")
        print(aux_info)
                
        return obs, reward, done, info
    
    def reset(self):        
        return self.env.reset()

class ExperimentMonitor(gym.Wrapper):    
    """
    Wrapper to save results of experiments to csv file.
    """    
    EXT = "ExperimentMonitor.csv"
    file_handler = None

    def __init__(self, filename, env = None, experiment = 0):                
        self.env = env
        #gym.Wrapper.__init__(self, env=env)
        self.t_start = time.time()
        if (not env is None):
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            super().__init__(env)
                    
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(ExperimentMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, ExperimentMonitor.EXT)
                else:
                    filename = filename + "." + ExperimentMonitor.EXT
            self.file_handler = open(filename, "wt")            
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('experiment', 'best', 'worst','time'))
            self.logger.writeheader()
            self.file_handler.flush()
        self.experiment = experiment
        
    def reset(self, **kwargs):
        return self.env.reset()        
        
    def step(self, action):
        return self.env.step(action)
                                        
    def end(self, info = {}):                        
        #ep_info = {"experiment": self.experiment, "best": info["best"] , "worst": info["worst"], "time": round(time.time() - self.t_start, 6)}
        ep_info = {"experiment": self.experiment, "time": round(time.time() - self.t_start, 6)}
        if self.logger:
            self.logger.writerow(ep_info)
            self.file_handler.flush()

class RenderMonitor(Monitor):
    """
        Rendering enviroment each step and save results in csv files. 
    """
    def __init__(self, env, rank, log_dir, rrender = True,  **kwargs):
        self.log_dir = log_dir
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.is_render = rrender
        
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'               
        
        self.rank = rank
        #self.render_gui = kwargs.get('render', False)
        #self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def reset(self, **kwargs):        
        if self.is_render:
            self.render()
        return super().reset(**kwargs)

    def step(self, action):   
        if self.is_render:
            self.render()
        return Monitor.step(self, action)

class ScaleRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, scale = 2):                    
        self.env = env
        self.action_space =  env.action_space
        self.observation_space = env.observation_space
        self.steps = 0
        self.scale_reward = scale
        
    def reset(self):
        self.steps = 0
        return super().reset()    
        
    def step(self, action):
        self.steps += 1
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info        
    
    def reward(self, reward):
        if (reward != 0):
            r = (self.scale_reward - (self.steps+1) / (self.env.changes+1)) * reward
        else:
            r = reward
        return r

class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clip the rewards to {+1, 0, -1} by it sign
    """
    def __init__(self, env):                    
        self.env = env
        self.action_space =  env.action_space
        self.observation_space = env.observation_space
        
    def reward(self, reward):
        r = np.sign(reward)
        if (r == 0):
            r = -1.0        
        return r

class ActionBoxWrappers(gym.Wrapper):
    """
        Wrapper to repeat the action on the environment
    """    
    def __init__(self, env):            
        super().__init__(env)
        self.env = env
        n = env.agent.generator.action_space.n        
        action_space = spaces.Box(low=0, high=n, shape=self.env.action_space.shape, dtype=np.int32)                        
        self.action_space = action_space
        self.observation_space = self.env.observation_space              
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'                               
        
    def step(self, action):              
        action = int(action[0])
        action = [action]                
        obs, reward, done, info = self.env.step(action)                                         
        return obs, reward, done, info
        
class ActionRepeatWrapper(gym.Wrapper):
    """
        Wrapper to repeat the action on the environment
    """    
    def __init__(self, env, n_act_repeat = 1):            
        super().__init__(env)
        self.env = env
        self._n_act_repeat = n_act_repeat
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space      
        
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'                               
        
    def step(self, action):
                                
        total_reward = 0.0
                        
        for _ in range(self._n_act_repeat):
            obs, reward, done, info = self.env.step(action)     
            total_reward += reward            
            break
                                    
        return obs, total_reward, done, info
    
class MaxStep(gym.Wrapper):        
    """
    Wrapper to reset the environment after a certain number of steps.    
    """
    def __init__(self, env, max_step):
        super().__init__(env)
        self.env = env

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space      
        self.max_step = max_step
        self.n_step = 0
        
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'                                       

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.n_step += 1
                
        if self.n_step >= self.max_step:
            done = True
            
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.n_step = 0        
        return obs    

class ResetMaxChange(gym.Wrapper):        
    """
    Wrapper to reset the environment after a certain number of changes.    
    """
    def __init__(self, env, max_step):
        super().__init__(env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space      
        self.max_step = max_step
        self.n_step = 0
        
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'                                       

    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        self.n_step += 1
                
        if self.n_step >= self.max_step:            
            self.reset()

        return obs, reward, done, info

    def reset(self):
        changes = self.env.counter_changes
        obs = self.env.reset()
        self.n_step = 0        
        self.env.counter_changes = changes
        return obs            
        
class MapWrapper(gym.Wrapper):
    
    def __init__(self, env): 
        super().__init__(env)        
        self.env = env        
        
        assert isinstance(
            self.env.observation_space, spaces.Dict
        ), "MapWrapper is only usable with dict observations."                
                
        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'                       
        
        self.observation_space = self.env.observation_space["map"]
        
        self._set_action_space()                         
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observatio Space can''t be None'                       
                       
    def _set_action_space(self):
        self.action_space = self.env.action_space
        return self.action_space
            
    def sample_actions(self):
        return self.env.sample_actions()
        
    def reset(self):        
        obs = self.env.reset()        
        return self.observation(obs)
            
    def step(self, action):           
        obs, reward, done, info = self.env.step(action)                                                    
        return self.observation(obs), reward, done, info
        
    def observation(self, obs):
        map = self.env.game.map
        return map

class HashWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)  
        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.uint32, shape=(9,))
        
    def reset(self):        
        obs = self.env.reset()        
        return self.observation(obs)
            
    def step(self, action):                                              
        obs, reward, done, info = self.env.step(action)                                
        return self.observation(obs), reward, done, info
        
    def observation(self, obs):           
        h = self.hash(obs)             
        return [h]
    
    def hash(self, obs, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()        
        to_encode = [obs["pos"], obs["tiles"], obs["map"]]
        for item in to_encode:            
            sample_hash.update(str(item).encode('utf8'))
            
        return sample_hash.hexdigest()[:size]        
        
class SegmentWrapper(gym.Wrapper):
    
    def __init__(self, env): 
        super().__init__(env)        
        self.env = env        
        
        assert isinstance(
            self.env.observation_space, spaces.Dict
        ), "MapWrapper is only usable with dict observations."                
                
        assert not self.env.agent is None , 'Agent can''t be None'               
        self._set_action_space()                         
        assert not self.action_space is None , 'Action Space can''t be None'               
        grid = self.env.agent.grid
        total_pieces = self.env.agent.total_pieces
        self.observation_space = spaces.Box( low = 0, high = total_pieces, shape = grid.shape, dtype=np.int32)

    def _set_action_space(self):
        self.action_space = self.env.action_space
        return self.action_space
            
    def sample_actions(self):
        return self.env.sample_actions()
        
    def reset(self):        
        obs = self.env.reset()        
        return self.observation(obs)
            
    def step(self, action):           
        obs, reward, done, info = self.env.step(action)                                                    
        return self.observation(obs), reward, done, info
            
    def observation(self, obs):
        grid = self.env.agent.grid        
        return grid

class SVDObservationWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use Convert RGB to Grayscale observation and resize  the image    
    """
    def __init__(self, env, shape = 84, rank = 5):
        super().__init__(env)               

        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)   
        self.rank = rank     
        self.action_space = self.env.action_space        
        self.observation_space = spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)        

        assert not self.action_space is None , 'Action Space can''t be None'
        assert not self.observation_space is None , 'Observatio Space can''t be None'
        
    def reset(self):        
        obs = self.env.reset()        
        return self.observation(obs)
            
    def step(self, action):                                        
        obs, reward, done, info = self.env.step(action)                                
        return self.observation(obs), reward, done, info        

    def observation(self, obs):                            
        u, s, v = np.linalg.svd(obs, full_matrices=False)                                      
        obs = cv2.resize(u, self.shape, interpolation=cv2.INTER_AREA)               
        obs = np.reshape(obs, self.shape)                    
        cv2.imshow('Game', obs)        
        return obs

class CV2ImgShowWrapper(gym.core.ObservationWrapper):

    def __init__(self, env = gym.Env):
        gym.ObservationWrapper.__init__(self, env)  
        self.observation_space = env.observation_space   

    def observation(self, frame: np.ndarray):        
        cv2.imshow('Game', frame) 
        #plt.imshow(frame)
        #plt.show()
        return frame

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.uint8):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = spaces.Box(old_space.low.repeat(n_steps, 
                 axis=0),old_space.high.repeat(n_steps, axis=0),     
                 dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
        dtype=self.dtype)
        return self.observation(self.env.reset())
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer        

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,            
                                shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]),
                                dtype=np.float32)
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)  

class ScaledFloatFrame(gym.ObservationWrapper):
    
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)

    def observation(self, obs):
        frame = np.array(obs).astype(np.float32) / 255.0
        #plt.imshow(frame)
        #plt.savefig("D:/Results/frame.png")
        return frame

class RGBToGrayScaleObservationWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use Convert RGB to Grayscale observation and resize  the image    
    """
    def __init__(self, env, shape):
        super().__init__(env)               

        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)        
        self.action_space = self.env.action_space        
        self.observation_space = spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

        #self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        assert not self.action_space is None , 'Action Space can''t be None'
        assert not self.observation_space is None , 'Observatio Space can''t be None'
        
    def reset(self):        
        obs = self.env.reset()        
        return self.observation(obs)
            
    def step(self, action):                                        
        obs, reward, done, info = self.env.step(action)                                
        return self.observation(obs), reward, done, info        

    def observation(self, obs):
        gray    = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)         
        obs = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)               
        obs = np.reshape(obs, self.shape)    
        obs = np.asarray(obs, dtype=np.uint8)                           
        #cv2.imshow('Game', obs) 
        return obs
    
class OneHotEncodingWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert the map observation to one hot enconding
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space           
        
        self.key = "map"
        
        shape = self.observation_space[self.key].shape
        self.y = shape[0]
        self.x = shape[1]
        self.z = self.observation_space[self.key].high.max() - self.observation_space[self.key].low.min() + 1                
        self.observation_space =  spaces.Box(low=0, high=1, shape=(self.y, self.x, self.z), dtype=np.uint32)
        
        assert not self.action_space is None , 'Action Space can''t be None'               
        assert not self.observation_space is None , 'Observation Space can''t be None'                       
        
    def observation(self, observation):            
        observation = np.array(observation[self.key])        
        observation = np.eye(self.z)[observation]          
        return observation
            
class WrappersType(Enum):
    RGB = "RGB"
    MAP = "map"
    SEGMENT  = "segment"
    ONEHOT = "onehot"    

    def __str__(self):
        return format(self.value)                                            
        
    def __int__(self):
        return self.value

def make_env(env, observation = WrappersType.MAP.value):
        
    _env = env
    
    if (observation == WrappersType.RGB.value):
        _env = RGBToGrayScaleObservationWrapper(env, (84, 84))        
    
    elif (observation == WrappersType.MAP.value):
        _env = MapWrapper(env)

    elif (observation == WrappersType.SEGMENT.value):
        _env = SegmentWrapper(env)
        
    elif (observation == WrappersType.ONEHOT.value):
        _env = OneHotEncodingWrapper(env)        
                                
    return _env