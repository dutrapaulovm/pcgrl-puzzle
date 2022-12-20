import time     
import random
import os
import numpy as np
from enum import Enum

def mk_dir(path_main, dir_name):
    path = os.path.join(path_main, dir_name)                            
    if not os.path.isdir(path):
        os.mkdir(path)
        
    return path

def gen_random_number(n = 32):
    return np.random.randint(2**n - 1, dtype="int64").item()

class RlAlgo(Enum):                     
    PPO   = "PPO"
    TRPO  = "TRPO"
    A2C   = "A2C"    
    
    def __str__(self):
        return self.value      
    
class ActivationFunc(Enum):                     
    SIGMOID = "Sigmoid"
    ReLU    = "ReLU"
    Tanh    = "Tanh"
    
    def __str__(self):
        return self.value  