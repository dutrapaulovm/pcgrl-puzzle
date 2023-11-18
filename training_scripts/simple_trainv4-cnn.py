from pcgrl import *
from pcgrl.wrappers import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame, MaxAndSkipEnv, NoopResetEnv

import torch as th
from custom_policy import CustomCNN

if __name__ == '__main__':
    
    env =  MazeCoinLowMapsEnv(agent=Experiment.AGENT_HEQHP.value, rep=Behaviors.NARROW_PUZZLE.value,
                              reward_change_penalty=-0.1)
    env.is_render = True
    set_random_seed(42)
    env.seed(42)    
    #Observation space for the environment
    env = RGBObservationWrapper(env)

    #PPO parameters
    total_timesteps = 25000
    learning_rate   = 3e-4
    n_steps         = 2048
    gamma           = 0.99
    gae_lambda      = 0.95
    batch_size      = 64
    n_epochs        = 10
    
    policy_kwargs = dict(
        features_extractor_class = CustomCNN,
        features_extractor_kwargs = dict(features_dim=512),
    )    

    env = WarpFrame(env)
    env = ScaledFloatFrame(env)    
    #env = CV2ImgShowWrapper(env)   
    env = ClipRewardEnv(env)            
    env = Monitor(env)    
    env = DummyVecEnv([lambda :env])    
    env = VecFrameStack(env, 4, channels_order='last')  

    model = PPO("CnnPolicy", env           = env,     
                             gamma         = gamma,         
                             gae_lambda    = gae_lambda,                                  
                             batch_size    = batch_size,
                             n_epochs      = n_epochs,   
                             n_steps       = n_steps,                          
                             learning_rate = learning_rate,       
                             policy_kwargs = policy_kwargs, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_mazecoin")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_mazecoin")
    
    obs = env.reset()
    success = 0
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            success += 1            
        env.render()
        print('Total Success {}'.format(success))