from pcgrl.RewardFunction import *
from pcgrl.Utils import *

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math

class SimpleReward(RewardFunction):

    def __init__(self, magnitude = 1, env = None):
        super(SimpleReward, self).__init__(magnitude = magnitude, env = env)        
        self.segments       = []

    def compute_reward(self, **kwargs):
        self.check_env()
        reward = self.env.max_entropy * self.magnitude
        return reward
        
    def __str__(self):
        return "S"

class Entropy(RewardFunction):

    def __init__(self, magnitude = 1, env = None):
        super(EntropyQuality, self).__init__(magnitude = magnitude, env = env)        
        self.segments       = []

    def compute_reward(self, **kwargs):        
        self.check_env()
        self.segments       = kwargs['segments']        
        reward = entropy(self.segments) * self.magnitude
        return reward
    
    def __str__(self):
        return "H"

class EntropyQuality(RewardFunction):

    def __init__(self, magnitude = 1, threshold = 10, env = None):
        super(EntropyQuality, self).__init__(magnitude = magnitude, env = env)        
        self.entropy_min = 1                
        self.threshold   = threshold       

    def compute_reward(self, **kwargs):
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

    def __init__(self, magnitude = 1, threshold = 10):
        super(EntropyQualityEx, self).__init__(magnitude = magnitude)        
        self.entropy_min    = 1
        self.segments       = []                

    def compute_reward(self, **kwargs):        
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
    