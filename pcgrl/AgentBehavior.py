# -*- coding: utf-8 -*-
from gym.utils import seeding
import gym
import numpy as np
from gym import spaces

class AgentInfo:
    """
    Include all information about an agent, such as, actions, status and observations    
    """
    def __init__(self) -> None:
        self.episodeId  = 0
        self.state      = -1
        self.prob       = 1.0
        self.reward     = 0
        self.done       = False                
        self.info       = None        
        self.obs        = []    

class AgentBehavior(gym.Env):        
    """
    AgentBehavior represents the behavior agent. An agent is an actor can observe 
    an environment and decide to take best actions using observations.    
    """    
    def __init__(self, max_iterations = None):
        self.seed()        
        self.iterations = 0
        self.max_iterations = max_iterations        
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(2)
        self._reward = 0     
        self._cumulative_reward = 0
        
    def scale_action(self, raw_action, min, max):
        """[summary]
        Args:
            raw_action ([float]): [The input action value]
            min ([float]): [minimum value]
            max ([flaot]): [maximum value]
        Returns:
            [type]: [description]
        """
        middle = (min + max) / 2
        range = (max - min) / 2
        return raw_action * range + middle

    def set_reward(self, reward):             
        """            
            Function used to replace rewards that agent earn during the current step
        Args:
            reward ([type float]): [New value of reward]
        """
        self._cumulative_reward += (reward - self._reward)
        self._reward = reward
                       
    def add_reward(self, reward):
        """[summary]
        Increments the rewards        
        Args:
            reward ([float]): [Value reward to increment]
        """
        self._cumulative_reward += reward
        self._reward += reward
                
    def get_action_space(self): 
        """        
            Returns the action space for this agent     
        """       
        return self.action_space
    
    def sample_action(self):
        """
        Returns:
            Choose a random action for this agent
        """
        return self.np_random.randint(self.action_space.n)
        
    def get_observation_space(self):
        return self.observation_space   

    def get_current_observation(self):
        return {}
    
    def get_stats(self):     
        """The current stats for this agent behavior
        Returns:
            New observations for this agent behavior
        """        
        raise NotImplementedError('reset is not implement')
    
    def seed(self, seed=None):      
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
            
    def reset(self):        
        """Resets the agent behavior
        Returns:
            New observations for this agent behavior
        """
        self._reward = 0
        self._cumulative_reward = 0        
    
    def step(self, action):
        """Steps of agents and returns observations from ready agents.
        Returns:
            New observations for this agent behavior
        """
        raise NotImplementedError('step is not implement')

    def is_done(self):
        return False

    def get_info(self):
        return {}        