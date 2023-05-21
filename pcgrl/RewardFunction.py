import numpy as np
from abc import abstractmethod, ABC

class RewardFunction:
    def __init__(self, magnitude = 1):        
        self.info = {}
        self.magnitude = magnitude

    @abstractmethod
    def compute_reward(self, **kwargs):
        pass

    def on_reset(self):
        pass

    def reset(self):
        info = self.info
        self.info = {}
        self.on_reset()        
        return info