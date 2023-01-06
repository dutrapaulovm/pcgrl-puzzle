from pcgrl import *
from pcgrl.wrappers import *
from stable_baselines3 import PPO

import torch as th

if __name__ == '__main__':
    
    env = gym.make('mazecoin-narrow-puzzle-2x3-v0')
    env.seed(42)    
    #Observation space for the environment
    env = SegmentWrapper(env)

    #PPO parameters
    total_timesteps = 25000
    learning_rate   = 3e-4
    n_steps         = 2048
    gamma           = 0.99
    batch_size      = 64
    n_epochs        = 10
    activation_fn   = th.nn.Sigmoid
    policy_kwargs   = dict(net_arch = [dict(pi=[64, 64], vf=[64, 64])], activation_fn=activation_fn)

    model = PPO("MlpPolicy", env  = env,     
                            gamma = gamma,         
                            batch_size = batch_size,
                            n_epochs = n_epochs,   
                            n_steps = n_steps,                          
                            learning_rate = learning_rate,       
                            policy_kwargs = policy_kwargs, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_mazecoin")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_mazecoin")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()