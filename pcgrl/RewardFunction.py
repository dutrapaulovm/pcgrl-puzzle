import numpy as np
from abc import abstractmethod, ABC

class RewardFunction:
    def __init__(self):
        self.info = {}

    @abstractmethod
    def compute_reward(self, **kwargs):
        pass

    def on_reset(self):
        pass

    def reset(self):
        self.on_reset()
        return self.info