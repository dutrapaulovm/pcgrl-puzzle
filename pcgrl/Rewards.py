from pcgrl.RewardFunction import *
from pcgrl.Utils import *

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math

class Entropy(RewardFunction):

    def __init__(self, magnitude = 1):
        super(EntropyQuality, self).__init__(magnitude = magnitude)        
        self.segments       = []

    def compute_reward(self, **kwargs):        
        self.segments       = kwargs['segments']        
        reward = entropy(self.segments) * self.magnitude
        return reward

class EntropyQuality(RewardFunction):

    def __init__(self, magnitude = 1):
        super(EntropyQuality, self).__init__(magnitude = magnitude)        
        self.entropy_min    = 1
        self.segments       = []          

    def compute_reward(self, **kwargs):        
        self.entropy_min    = kwargs['entropy_min']
        self.segments       = kwargs['segments']

        sign = lambda x: math.copysign(1, x)

        reward = 0
        x = math.pi
        e = entropy(self.segments)
        r = (e**x - self.entropy_min**x)                
        f = 1
        reward = (((r + sign(r)) * f)) * self.magnitude       
        return reward    

class EntropyQualityEx(RewardFunction):

    def __init__(self, magnitude = 1):
        super(EntropyQualityEx, self).__init__(magnitude = magnitude)        
        self.entropy_min    = 1
        self.segments       = []                

    def compute_reward(self, **kwargs):        
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

        #for r in self.rewards:
        #   reward += r.compute_reward(**kwargs)

        reward += agent_reward

        return reward