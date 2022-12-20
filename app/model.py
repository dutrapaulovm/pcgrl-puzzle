from email import policy
import os
import time     
import numpy as np
from itertools import count

import gym
from gym.spaces import Discrete, Box
#from gym.envs.classic_control import rendering
from gym.utils import seeding

import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from pcgrl.Utils import clear_console
import pcgrl
from pcgrl.mazecoin import MazeCoinGameProblem
import app
from mazecoinplay_env import MazeCoinPlayEnv, TrainAndLoggingCallback

from pcgrl.mazecoin import MazeCoinGameProblem
from pcgrl.utils.experiment import SaveOnBestTrainingRewardCallback
from pcgrl.wrappers import ExperimentMonitor, MaxStep, RGBToGrayScaleObservationWrapper, ScaledFloatFrame, SVDObservationWrapper, ImageToPyTorch, CV2ImgShowWrapper, BufferWrapper, OneHotEncodingWrapper

from stable_baselines3 import PPO,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame, MaxAndSkipEnv, NoopResetEnv

class Policy(nn.Module):
    def __init__(
        self,
        s_size: int = 1,
        h_size: int = 4,
        a_size: int = 1,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,        
        input_size = 0, output_size = 0
    ):
        super(Policy, self).__init__()        
        
        # Policy network
        #self.fc1 = nn.Linear(s_size, h_size)        
        #self.fc2 = nn.Linear(h_size, a_size)

        self.output_size = output_size

        self.fc1 = nn.Linear(in_features=input_size, out_features=output_size)        
        self.fc2 = nn.Linear(in_features=output_size, out_features=4)                
      

    def forward(self, features):                     
        x = torch.squeeze(features)                         
        x = F.sigmoid(self.fc1(x))        
        x = self.fc2(x)                
        output = F.log_softmax(x, dim=1)  
        return output

class PolicyGradientAgent(object):
    def __init__(self, 
            env,
            learning_rate:float = 3e-4,
            n_steps: int = 128,                    
            gamma: float = 0.99,
            log_interval:int = 10):
        self.learning_rate = learning_rate        
        self.gamma   = gamma
        self.n_steps = n_steps
        self.env =  env
        self.log_interval =log_interval
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.policy = Policy(input_size=self.observation_dim, output_size=self.action_dim) #(feature_dim=self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), learning_rate)
        self.eps = np.finfo(np.float32).eps.item()  
        self.rewards = []
        self.saved_log_probs = []
        self._last_obs = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, total_timesteps: int = 1000):
        running_reward = 10
        ep_reward = 0
        for episode in range(total_timesteps):           
            
            self._last_obs = self.env.reset()
            
            for t in range(self.n_steps):

                action = self.select_action(self._last_obs)     
                print(action)        
                #action = action[0][0]
                #action = action[0]
                #action = action[0]
                #time.sleep(5)
                                
                self._last_obs, reward, done, _ = self.env.step(action[0])                

                self.env.render()

                self.rewards.append(reward)
                ep_reward += reward

                if (done):
                    break
            
            running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
            
            self.finish_episode()

            if episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    episode, ep_reward, running_reward))
            if running_reward > self.env.spec.reward_threshold:  
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break

    def select_action(self, state):
        """
        n_values = np.max(state) + 1
        state = np.array(state)        
        state = np.eye(n_values)[state]                  
        """
        #print("state", state.shape)
        state = torch.from_numpy(state).float().unsqueeze(-1).to(self.device)
        probs = self.policy.forward(state).cpu()       

        m = Categorical(probs)
        action = m.sample() 
        
        self.saved_log_probs.append(m.log_prob(action))
        action = action.numpy()
        #action = action.item()
        #print("Action", action)        
        #time.sleep(5)
        return action#, m.log_prob(action)

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps) #Normalização
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) 
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()          
        policy_loss.backward()        
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

if __name__ == '__main__':
    env = gym.make("mazecoinplay-v2-2x2") 
    env.seed(0)
    env = WarpFrame(env)
    #env = ScaledFloatFrame(env)     
    p = PolicyGradientAgent(env)

    p.train()

