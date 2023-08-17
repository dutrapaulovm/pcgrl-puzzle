import numpy as np
from abc import abstractmethod, ABC

class RewardFunction:
    def __init__(self, magnitude = 1, env = None):        
        self.info = {}
        self.magnitude = magnitude
        self.env = env

    def check_env(self):
        assert not self.env is None , 'Env can''t be None'

    @abstractmethod
    def compute_reward(self, **kwargs):
        return 0

    def on_reset(self):
        pass

    def reset(self):
        info = self.info
        self.info = {}
        self.on_reset()        
        return info