import gym
import pcgrl

env = gym.make('mazecoin-narrow-puzzle-2x3-v2')
obs = env.reset()
t = 0
while t < 1000:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  #print(obs)
  #print(reward)  
  env.render()  
  if done:
    print("Episode finished after {} timesteps".format(t+1))
    break
  t += 1
