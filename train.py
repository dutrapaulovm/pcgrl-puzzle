import pandas as pad

from pcgrl import *
from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.minimap import *
from pcgrl.maze.MazeEnv import MazeEnv
from pcgrl.minimap.MiniMapEnv import MiniMapEnv
from pcgrl.dungeon.DungeonEnv import DungeonEnv
from pcgrl.zelda.ZeldaEnv import ZeldaEnv
from pcgrl.wrappers import *

from sb3_contrib import TRPO
from stable_baselines3 import PPO, DDPG, A2C

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import torch as th

import matplotlib.pyplot as plt
from utils import *
from custom_policy import *

callback_log_dir = os.path.dirname(__file__) + "/results/" #'./'    
n_calls = 0
best_mean_reward = -np.inf
check_freq = 10
verbose = 1

RL_ALGO = {
    "a2c": A2C,    
    "ppo": PPO,            
    "trpo": TRPO,
}

#https://stable-baselines3.readthedocs.io/en/v0.11.1/guide/examples.html
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        #Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          #Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              #Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              #New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def train_ppo(policy:Union[str, Type[ActorCriticPolicy]],                
              rl_algo:str = RlAlgo.PPO.value,
              results_dir = "./results/",
              total_timesteps = 50000,              
              learning_rate: float = 2.5e-4, 
              n_steps:int   = 128,              
              gamma: float = 0.99,              
              policy_size  = [64, 64],
              batch_size:int = 64,
              n_epochs:int = 10,                                          
              act_func = ActivationFunc.SIGMOID.value,
              entropy_min:int = 1.80,
              envs = [Game.ZELDA.value, Game.MINIMAP.value, Game.MAZECOIN.value, Game.MAZE.value, Game.DUNGEON.value],
              representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
              observations = [WrappersType.MAP.value],              
              agent   = Experiment.AGENT_HHP.value,
              seed:int = 1000):
    """
    Training Agents

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)    
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param seed: Seed for the pseudo random generators
    """
    global callback_log_dir    
    
    #set_random_seed(seed)
    
    n_experiments = 1

    n_experiments_per_obs = 1 #The number of experiments
    
    #PPO2
    verbose         = 1 #verbose – the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    RL_ALG = rl_algo
    act_func = act_func
    if act_func == ActivationFunc.ReLU.value:
        activation_fn = th.nn.ReLU
    elif act_func == ActivationFunc.SIGMOID.value:
        activation_fn = th.nn.Sigmoid
    elif act_func == ActivationFunc.Tanh.value:
        activation_fn = th.nn.Tanh
                    
    save_image_level = False
    show_logger      = False
    show_hud         = True

    #Parametros ambiente        
    max_changes            = 61
    use_done_max_changes   = True
    versions               = [agent]
    action_change = False    
    
    timesteps = [total_timesteps]    
    mlp_units = [policy_size[0]]
                            
    games = envs
    
    render = False    
        
    model_dir = "models"
    
    parent_dir = mk_dir(os.path.dirname(__file__), results_dir)        
    path_model_dir = parent_dir
    path_model_dir = mk_dir(path_model_dir, model_dir)
    
    for version in versions:
        
        if (version == Experiment.AGENT_HEQBDC.value):
            action_change = True    
    
        for t_time_s in timesteps:

            total_timesteps = t_time_s            
            

            for mlp_u in mlp_units:        

                n_units = mlp_u

                #layers          = []

                #for i in range(n_layers):
                #    layers.append(n_units)
                                    
                policy_kwargs = dict(net_arch=policy_size, activation_fn=activation_fn)
                
                """
                actor          = []
                vf             = []
                
                for i in range(n_layers):
                    actor.append(n_units)                                                                            
                    vf.append(n_units)
                
                # Custom actor (pi) and value function (vf) networks of two layers each 
                policy_kwargs = dict(activation_fn=act_func,
                        net_arch=[dict(pi=actor, vf=vf)])
                """
                            
                for name_game in games:

                    for par in range(n_experiments):
                        
                        main_dir = "Experiment 0" + str(par+1) + "-"+version+"-"+name_game+"-"+RL_ALG
                            
                        logging_tensorboard = False
                                                                
                        for _rep in representations:            
                            
                            representation = _rep
                            rep_path = representation
                            
                            for _obs in observations:
                                    
                                observation = _obs
                                
                                params = {
                                    "total_timesteps" : total_timesteps, 
                                    "learning_rate" : learning_rate,                
                                    "n_steps" : n_steps,
                                    "gamma" : gamma,                
                                    "batch_size" : batch_size,
                                    "n_epochs" : n_epochs,
                                    "seed" : seed,
                                    "verbose" : verbose,
                                    "policy_size"   : policy_size,                                    
                                    "representation": representation,
                                    "observation": observation,                                                                        
                                    "max_changes" : max_changes,
                                    "use_done_max_changes" : use_done_max_changes,
                                    "entropy" : entropy_min,                                
                                    "RL_ALG" : RL_ALG,                                
                                    "action_change" : action_change
                                }
                                                                                            
                                info_params = []
                                info_params.append(params)
                                    
                                df = pad.DataFrame(info_params)             

                                rep_path =  representation + "-" + observation
                                
                                dirname_experiments2 = "experiments-"+str(total_timesteps)+"-Steps"+str(n_steps)+"-L"+str(n_units)+"-E"+str(entropy_min)+"-LR"+str(learning_rate)+"SD"+str(seed)+act_func
                                    
                                path_experiments2 = mk_dir(parent_dir, dirname_experiments2)
                                
                                dirname_experiments2 = "experiments-"+str(total_timesteps)+"-Steps"+str(n_steps)+"-L"+str(n_units)+"-E"+str(entropy_min)+"-LR"+str(learning_rate)+"SD"+str(seed)+act_func
                                path_experiments2 = os.path.join(parent_dir, dirname_experiments2)            
                                
                                path_experiments2 = os.path.join(path_experiments2, main_dir)

                                if not os.path.isdir(path_experiments2):
                                    os.mkdir(path_experiments2)                                

                                pathrep = mk_dir(path_experiments2, rep_path) 
                                
                                df.to_csv(pathrep+"/params.csv", index=False)
                                env_experiment = ExperimentMonitor(pathrep)                                                        
                                
                                for exp in range(n_experiments_per_obs):

                                    env_experiment.experiment = exp
                                    
                                    path_experiments2 = mk_dir(pathrep, rep_path+str(exp)) 

                                    path_experiments2 = mk_dir(pathrep, rep_path+str(exp))   

                                    mk_dir(path_experiments2, "best")                         

                                    mk_dir(path_experiments2, "max_changes")                         
                                    
                                    mk_dir(path_experiments2, "worst")

                                    path_map = mk_dir(path_experiments2, "map")
                                    
                                    mk_dir(path_map, "best")
                                    
                                    mk_dir(path_map, "worst")
                                    
                                    if name_game == Game.MAZE.value:             
                                        singleEnv = MazeEnv(seed=seed, rep = representation, path=path_experiments2, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change)
                                        singleEnv.use_done_max_changes = use_done_max_changes
                                        singleEnv.max_changes = max_changes                                        
                                        singleEnv.game.show_hud       = show_hud
                                    elif name_game == Game.MAZECOIN.value:             
                                        singleEnv = MazeCoinEnv(seed=seed, rep = representation, path=path_experiments2, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change)                        
                                        singleEnv.use_done_max_changes = use_done_max_changes
                                        singleEnv.max_changes = max_changes    
                                    elif name_game == Game.DUNGEON.value:             
                                        singleEnv = DungeonEnv(seed=seed, rep = representation, path=path_experiments2, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change)                        
                                        singleEnv.use_done_max_changes = use_done_max_changes
                                        singleEnv.max_changes = max_changes                                                                                             
                                        singleEnv.game.show_hud       = show_hud                                                               
                                    elif name_game == Game.ZELDA.value or name_game == Game.ZELDAV3.value:             
                                        singleEnv = ZeldaEnv(seed=seed, rep = representation, path=path_experiments2, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change)                        
                                        singleEnv.use_done_max_changes = use_done_max_changes
                                        singleEnv.max_changes = max_changes                                                                                            
                                        singleEnv.game.show_hud       = show_hud               
                                    elif name_game == Game.MINIMAP.value:             
                                        singleEnv = MiniMapEnv(seed=seed, rep = representation, path=path_experiments2, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change)                        
                                        singleEnv.use_done_max_changes = use_done_max_changes
                                        singleEnv.max_changes = max_changes                                                                                            
                                        singleEnv.game.show_hud       = show_hud               

                                    game = singleEnv.game
                                    singleEnv.exp = version  
                                    singleEnv.is_render = render
                                    singleEnv.entropy_min = entropy_min
                                                        
                                    env_experiment.env = singleEnv                
                                    env_experiment.action_space = singleEnv.action_space
                                    env_experiment.observation_space = singleEnv.observation_space                                                          
                                    
                                    env =  make_env(env_experiment, observation = observation)
                                                                                                
                                    path_monitors_experiments2 = os.path.join(path_experiments2, "monitors")                            
                                    if not os.path.isdir(path_monitors_experiments2):
                                        os.mkdir(path_monitors_experiments2)
                                        
                                    path_logs = os.path.join(path_experiments2, "logs")                            
                                    if not os.path.isdir(path_logs):
                                        os.mkdir(path_logs)                                                                                                                
                                    
                                    
                                    env = RenderMonitor(env, exp, path_monitors_experiments2, rrender=render)                                    
                                                                                                                    
                                    env = DummyVecEnv([lambda: env])
                                    
                                    # Automatically normalize the input features and reward
                                    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
                                    
                                    if rl_algo == RlAlgo.PPO.value:
                                        if logging_tensorboard:                
                                            model = PPO(policy, 
                                                        env,  
                                                        verbose = verbose,
                                                        n_steps = n_steps, 
                                                        seed = seed,  
                                                        gamma = gamma,                                                                                                                                                                  
                                                        learning_rate   = learning_rate,
                                                        batch_size = batch_size,
                                                        n_epochs = n_epochs,
                                                        tensorboard_log = path_experiments2,
                                                        policy_kwargs   = policy_kwargs)                         
                                        else:                                              
                                            model = PPO(policy, 
                                                        env,  
                                                        verbose = verbose,
                                                        n_steps = n_steps, 
                                                        seed    = seed,
                                                        gamma = gamma,                                                                         
                                                        learning_rate = learning_rate,                                    
                                                        batch_size = batch_size,
                                                        n_epochs = n_epochs,                                                        
                                                        policy_kwargs = policy_kwargs)                                            
                                    elif rl_algo == RlAlgo.TRPO.value:
                                        model = TRPO(policy, 
                                                    env,  
                                                    verbose = verbose,
                                                    n_steps = n_steps, 
                                                    seed    = seed,    
                                                    gamma = gamma,                                                                     
                                                    learning_rate = learning_rate,                                    
                                                    policy_kwargs = policy_kwargs)                                            
                                    elif rl_algo == RlAlgo.A2C.value:
                                        model = A2C(policy, 
                                                    env,  
                                                    verbose = verbose,
                                                    n_steps = n_steps, 
                                                    gamma = gamma,
                                                    seed    = seed,                                                                         
                                                    learning_rate = learning_rate,                                    
                                                    policy_kwargs = policy_kwargs)                                         
                                                                                                                        
                                    callback_log_dir = path_monitors_experiments2
                                    
                                    saveOnBest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=callback_log_dir)
                                    
                                    model.learn(total_timesteps = int(total_timesteps), callback = saveOnBest_callback)
                                    
                                    filename_model = representation + "-" + observation+"-"+str(total_timesteps)+"-Steps"+str(n_steps)+"-L"+str(n_units)+"-"+main_dir+"-E"+str(entropy_min)+"-LR"+str(learning_rate)+"SD"+str(seed)+act_func
                                    path_model = path_model_dir + "/"+filename_model
                                    model.save(path_model)
                                    
                                    #stats_path = path_model + "/" + filename_model+".pkl"
                                    #env.save(stats_path)                                    
                                                                        
                                    results_plotter.plot_results([path_monitors_experiments2], int(total_timesteps), results_plotter.X_TIMESTEPS, name_game)
                                    plt.savefig(path_monitors_experiments2+"/Monitor.png")
                                    
                                    env_experiment.end()                                    

if __name__ == '__main__':
    train_ppo()