from pcgrl import *
from pcgrl.wrappers import *
from stable_baselines3 import PPO

if __name__ == '__main__':
    
    env = gym.make('mazecoin-narrow-puzzle-2x3-v0')
    env = SegmentWrapper(env)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_mazecoin")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_mazecoin")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()