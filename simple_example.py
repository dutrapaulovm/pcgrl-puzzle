import gym
import pcgrl

from pcgrl.zelda.ZeldaV2Env import *
from pcgrl.BasePCGRLEnv import Experiment, Behaviors

#prob = {Ground.ID : 0.50, Player.ID: 0.02, Key.ID: 0.02, Coin.ID : 0.25, Enemy.ID: 0.25, Weapon.ID : 0.02}
#tiles = [Ground.ID, Player.ID, Key.ID, Coin.ID, Enemy.ID, Weapon.ID]

#print(random.choices( list(prob.keys()), list(prob.values()), k = (5 * 5)) )  
#map = random.choice( list(prob.keys()), size=(height, width), p=list(prob.values())).astype(np.uint8)


env = ZeldaV2Env(agent=Experiment.AGENT_SS.value,rep=Behaviors.NARROW_PUZZLE.value,rendered=True) #gym.make('mazecoin-narrow-puzzle-2x3-v2')
obs = env.reset()
t = 0
while t < 9000000:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  #print(obs)
  #print(reward)  
  env.render()  
  if done:
    print("Episode finished after {} timesteps".format(t+1))
    break
  t += 1
