import numpy as np
from abc import abstractmethod, ABC
from gym import Env

class RewardFunction:
    def __init__(self, magnitude:float = 1, env:Env = None):        
        """AI is creating summary for __init__

        Args:
            magnitude (float, optional): [description]. Defaults to 1.
            env (Env, optional): [description]. Defaults to None.
        """        
        self.info = {}
        self.magnitude = magnitude
        self.env = env

    def check_env(self):
        """AI is creating summary for check_env
        """
        assert not self.env is None , 'Env can''t be None'

    @abstractmethod
    def compute_reward(self, **kwargs):
        """AI is creating summary for compute_reward

        Returns:
            [type]: [description]
        """
        return 0

    def on_reset(self):
        """AI is creating summary for on_reset
        """
        pass

    def reset(self):
        """AI is creating summary for reset

        Returns:
            [type]: [description]
        """
        info = self.info
        self.info = {}
        self.on_reset()        
        return info