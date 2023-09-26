from pcgrl.RewardFunction import *
from pcgrl.Utils import *

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math
import gym 
class SimpleReward(RewardFunction):

    def __init__(self, magnitude = 1, env = None):
        """AI is creating summary for __init__

        Args:
            magnitude (int, optional): [description]. Defaults to 1.
            env ([type], optional): [description]. Defaults to None.
        """
        super(SimpleReward, self).__init__(magnitude = magnitude, env = env)        
        self.segments       = []



    def compute_reward(self, **kwargs):
        """AI is creating summary for compute_reward

        Returns:
            [type]: [description]
        """        
        self.check_env()
        reward = self.env.max_entropy * self.magnitude
        return reward
        
    def __str__(self):
        return "S"


class Entropy(RewardFunction):

    def __init__(self, magnitude:float = 1, env:gym.Env = None):
        """AI is creating summary for __init__

        Args:
            magnitude (float, optional): [description]. Defaults to 1.
            env (gym.Env, optional): [description]. Defaults to None.
        """        
        super(EntropyQuality, self).__init__(magnitude = magnitude, env = env)        
        self.segments       = []


    def compute_reward(self, **kwargs):        
        """AI is creating summary for compute_reward

        Returns:
            [type]: [description]
        """        
        self.check_env()
        self.segments       = kwargs['segments']        
        reward = entropy(self.segments) * self.magnitude
        return reward
    
    def __str__(self):
        return "H"

class EntropyQuality(RewardFunction):
    
    def __init__(self, magnitude:float = 1, threshold:float = 10, env:gym.Env = None):
        """AI is creating summary for __init__

        Args:
            magnitude (float, optional): [description]. Defaults to 1.
            threshold (float, optional): [description]. Defaults to 10.
            env (gym.Env, optional): [description]. Defaults to None.
        """        
        super(EntropyQuality, self).__init__(magnitude = magnitude, env = env)        
        self.entropy_min = 1                
        self.threshold   = threshold       

    def compute_reward(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.check_env()
        self.entropy_min    = kwargs['entropy_min']
        self.segments       = kwargs['segments']
        agent_reward        = kwargs['agent_reward']

        sign = lambda x: math.copysign(1, x)

        reward = 0
        x = math.pi
        e = entropy(self.segments)
        r = (e**x - self.entropy_min**x)                
        f = self.threshold
        reward = (((r + sign(r)) * f)) * self.magnitude       
        return reward
        
    def __str__(self):
        return "HQ"    

class EntropyQualityEx(RewardFunction):

    def __init__(self, magnitude:float = 1, threshold:float = 10):
        """AI is creating summary for __init__

        Args:
            magnitude (float, optional): [description]. Defaults to 1.
            threshold (float, optional): [description]. Defaults to 10.
        """
        super(EntropyQualityEx, self).__init__(magnitude = magnitude)        
        self.entropy_min    = 1
        self.segments       = []                

    def compute_reward(self, **kwargs):        
        """AI is creating summary for compute_reward

        Returns:
            [type]: [description]
        """
        self.check_env()
        self.entropy_min    = kwargs['entropy_min']
        self.segments       = kwargs['segments']
        agent_reward        = kwargs['agent_reward']
        
        sign = lambda x: math.copysign(1, x)

        reward = 0
        x = math.pi
        e = entropy(self.segments)
        r = (e**x - self.entropy_min**x)                
        f = 1
        reward = (((r + sign(r)) * f)) * self.magnitude

        reward += agent_reward

        return reward 
    