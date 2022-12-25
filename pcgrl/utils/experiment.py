from pickle import TRUE
import imageio
import pandas as pad
from uuid import uuid4
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pcgrl import *
from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.minimap import *
from pcgrl.maze.MazeEnv import MazeEnv
from pcgrl.minimap.MiniMapEnv import MiniMapEnv
from pcgrl.dungeon.DungeonEnv import DungeonEnv
from pcgrl.zelda.ZeldaEnv import ZeldaEnv
from pcgrl.wrappers import *
from pcgrl.utils.utils import *
from pcgrl.utils.utils import gen_random_number
from pcgrl.utils.experiment import *
from pcgrl.utils.plot_results import PlotResults

from stable_baselines3 import PPO, DDPG, A2C

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy

from custom_policy import CustomCNN

import torch as th

import pylab
import matplotlib.pyplot as plt

callback_log_dir = os.path.dirname(__file__) + "/results/" #'./'    
n_calls = 0
best_mean_reward = -np.inf
check_freq = 10
verbose = 1

RL_ALGO = {
    "A2C": A2C,    
    "PPO": PPO
}

def plot_bar(df, path, filename, title, steps):           
    
    total = np.array(df["Total"])
    error = total / steps

    labels = ["Narrow Puzzle", "Wide Puzzle"]
    
    #for index, row in df.iterrows():
    #    labels.append(row['Representation']+'/'+row['Observation'])        
        
    plt.rcdefaults()
    fig, ax = plt.subplots()    
    
    y_pos = np.arange(len(total))
    hbars = ax.barh(y_pos, total, xerr=error,align='center')
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Steps')
    ax.set_title(title)  
    ax.bar_label(hbars) #, fmt='%.2f')
    ax.set_xlim(right=steps)
    fig.tight_layout()
    plt.savefig(path+"/"+filename+".png")        
    plt.close()

def plot_all_rewards(average, scores, episodes, path, filename, title):    
    
    linestyle_str = [
        ('solid', 'solid'),      # Same as (0, ()) or '-'
        ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
        ('dashed', 'dashed'),    # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]    
    
    linestyles = []
    for i, (name, linestyle) in enumerate(linestyle_tuple[::-1]):
        linestyles.append(linestyle)    
    
    fig, axs = plt.subplots()
    plt.figure(figsize=(18, 9))
    c = 0    
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]        
        #plt.plot(episodes[i][2], scores[i][2], linestyle=linestyles[i], label='Rewards: ' + label)      
        plt.plot(episodes[i][2], scores[i][2], label='Rewards: ' + label)      
        c += 2
            
    plt.title("Inference: " + title)
    plt.ylabel('Rewards', fontsize=18)
    plt.xlabel('Steps', fontsize=18) 
    plt.grid(True)
    plt.legend()  
    plt.savefig(path+"/"+filename+".png")    
    plt.close()
    
def plot_all_average(average, scores, episodes, path, filename, title):    
    
    linestyle_str = [
        ('solid', 'solid'),      # Same as (0, ()) or '-'
        ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
        ('dashed', 'dashed'),    # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),

        ('loosely dashed',        (0, (5, 10))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1))),

        ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]    
    
    #linestyles = []
    #for i, (name, linestyle) in enumerate(linestyle_tuple[::-1]):
    #    linestyles.append(linestyle)        
    
    fig, axs = plt.subplots()
    plt.figure(figsize=(18, 9))
    c = 0
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]
        #plt.plot(episodes[i][2], average[i][2], linestyle=linestyles[i], label='Average: ' + label)        
        plt.plot(episodes[i][2], average[i][2], label='Average: ' + label)        
        c += 2
    
    plt.title("Inference: " + title)
    plt.ylabel('Rewards Average', fontsize=18)
    plt.xlabel('Steps', fontsize=18) 
    plt.grid(True)
    plt.legend()  
    plt.savefig(path+"/"+filename+".png")    
    plt.close()    

def plot_all(average, scores, episodes, path, filename):
    
    fig, axs = plt.subplots()
    for i in range(len(average)):
        label = episodes[i][0] + ", " + episodes[i][1]
        plt.plot(episodes[i][2], average[i][2], label='Average: ' + label)
        plt.plot(episodes[i][2], scores[i][2],  label='Rewards: ' + label)    
    
    plt.ylabel('Rewards', fontsize=18)
    plt.xlabel('Steps', fontsize=18)        
    plt.title("Inference")
    plt.grid()
    plt.legend()
    plt.savefig(path+"/"+filename+".png")
    plt.close()

def plot(average, scores, episodes, path, filename, rep, game):
    pylab.figure(figsize=(18, 9))
    pylab.plot(episodes, average,'r', label='Average')
    pylab.plot(episodes, scores, 'b', label='Rewards')
    pylab.ylabel('Rewards', fontsize=18)
    pylab.xlabel('Steps', fontsize=18)        
    pylab.title(game + ", Representation: " + rep)
    pylab.grid()
    pylab.legend()
    pylab.savefig(path+"/"+filename+".png")

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
    
class ExperimentManager(object):    
    def __init__(
        self,
            policy:Union[str, Type[ActorCriticPolicy]] = None,                
            rl_algo:str = RlAlgo.PPO.value,
            results_dir = "./results/",            
            total_timesteps = 50000,              
            learning_rate: float = 3e-4, 
            n_steps:int   = 2048,              
            gamma: float = 0.99,
            batch_size:int = 64,            
            n_epochs:int = 10,
            act_func = ActivationFunc.SIGMOID.value,
            entropy_min:int = 1.80,
            max_changes:int = 61,
            action_change:bool = False,
            action_rotate:bool = False,
            factor_reward:float = 1.0,
            reward_best_done_bonus:float = 50,
            reward_medium_done_bonus:float = 10,
            reward_entropy_penalty:float = 0,
            board = (3,2),
            reward_low_done_bonus:float = 0,                  
            reward_change_penalty = None,
            piece_size = 8,
            env_rewards = False,            
            envs = [Game.ZELDA.value, Game.MAZECOINLOWMAPS.value, Game.DUNGEON.value],
            representations = [Behaviors.NARROW_PUZZLE.value, Behaviors.WIDE_PUZZLE.value],
            observations = [WrappersType.MAP.value],              
            agent   = [Experiment.AGENT_HHP.value], 
            seed:int = -1,
            uuid:str = "",            
            hyperparams: Optional[Dict[str, Any]] = None,                        
            policy_kwargs : Optional[Dict[str, Any]] = dict(net_arch = [64, 64], activation_fn=th.nn.Sigmoid),
            verbose: int = 1):    
        super(ExperimentManager, self).__init__()
                
        if (policy is None):
            policy = "MlpPolicy"
            
        if seed < 0:
            seed = gen_random_number() 
        
        self.policy  = policy
        
        self.rl_algo = rl_algo
        self.results_dir = results_dir
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate 
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.act_func = act_func        
        self.envs = envs
        self.representations = representations
        self.observations = observations
        self.agent = agent
        self.seed = seed
        self.verbose = verbose
        self.tensorboard_log = False
        self.hyperparams = hyperparams
        self.policy_kwargs = policy_kwargs                
        self.max_changes = max_changes
        self.entropy_min = calc_entropy(n_repetead = 2, size=board) #entropy_min
        self.uuid = uuid
        self.reward_best_done_bonus = reward_best_done_bonus
        self.reward_medium_done_bonus = reward_medium_done_bonus
        self.reward_low_done_bonus = reward_low_done_bonus
        self.reward_change_penalty = reward_change_penalty
        self.reward_entropy_penalty = reward_entropy_penalty
        self.factor_reward = factor_reward
        self.board = board
        self.action_change = action_change
        self.action_rotate = action_rotate
        self.piece_size = piece_size
        self.env_rewards = env_rewards        

    def __make_env(self, name_game, max_changes = 61, 
                        save_image_level = False,
                        show_logger      = False,
                        show_hud         = False,
                        action_change    = False,
                        action_rotate    = False,
                        path_experiments = None,
                        agent            = None,
                        render           = False,
                        observation      = None,
                        representation   = Behaviors.NARROW_PUZZLE.value):
        singleEnv = None       
        
        kwargs = {"seed" : self.seed,
                  "rep" : representation,
                  "path" : path_experiments,
                  "save_logger" : True,
                  "save_image_level" : save_image_level,
                  "show_logger" : show_logger,
                  "action_change" : action_change,
                  "action_rotate" : action_rotate,
                  "board" : self.board,
                  "piece_size" : self.piece_size}        
        
        if name_game == Game.MAZE.value:                         
            singleEnv = MazeEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate, board = self.board, piece_size = self.piece_size, env_rewards = self.env_rewards)
        elif name_game == Game.MAZECOIN.value:                         
            singleEnv = MazeCoinEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate,board = self.board, piece_size = self.piece_size,env_rewards = self.env_rewards)                                  
        elif name_game == Game.MAZECOINLOWMAPS.value:                         
            singleEnv = MazeCoinLowMapsEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate,board = self.board, piece_size = self.piece_size, env_rewards = self.env_rewards)                     
        elif name_game == Game.DUNGEON.value:             
            singleEnv = DungeonEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate,board = self.board, piece_size = self.piece_size,env_rewards = self.env_rewards)      
        elif name_game == Game.ZELDA.value:             
            singleEnv = ZeldaEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate, board = self.board, piece_size = self.piece_size,env_rewards = self.env_rewards)
        elif name_game == Game.MINIMAP.value:             
            singleEnv = MiniMapEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate,board = self.board, piece_size = self.piece_size,env_rewards = self.env_rewards)      
        elif name_game == Game.MINIMAPLOWMODELS.value:             
            singleEnv = MiniMapLowMapsEnv(seed=self.seed, rep = representation, path=path_experiments, save_logger=True, save_image_level=save_image_level, show_logger=show_logger, action_change=action_change, action_rotate=action_rotate, board = self.board, piece_size = self.piece_size,env_rewards = self.env_rewards)       
        elif name_game == Game.SMB.value:             
            singleEnv = SMBEnv(seed = self.seed, 
                               rep  = representation, 
                               path = path_experiments, 
                               save_logger = True, 
                               save_image_level = save_image_level, 
                               show_logger = show_logger, 
                               action_change = action_change, 
                               board = self.board, 
                               piece_size = self.piece_size, 
                               env_rewards = self.env_rewards)
        else:
            singleEnv = gym.make(name_game)
            singleEnv = singleEnv.env            

        singleEnv.max_changes = max_changes                                                                                            
        singleEnv.game.show_hud  = show_hud                                                               
        singleEnv.reward_best_done_bonus = self.reward_best_done_bonus
        singleEnv.reward_medium_done_bonus = self.reward_medium_done_bonus
        singleEnv.reward_low_done_bonus = self.reward_low_done_bonus
        singleEnv.reward_change_penalty = self.reward_change_penalty
        singleEnv.reward_entropy_penalty = self.reward_entropy_penalty
        singleEnv.factor_reward = self.factor_reward                
                  
        game = singleEnv.game
        singleEnv.exp = agent
        singleEnv.is_render = render
        singleEnv.entropy_min = self.entropy_min
        

        experiment_monitor = ExperimentMonitor(path_experiments)                           
        experiment_monitor.env = singleEnv                
        experiment_monitor.action_space = singleEnv.action_space
        experiment_monitor.observation_space = singleEnv.observation_space                                                          
        experiment_monitor.experiment = 1
                
        env =  make_env(experiment_monitor, observation = observation)   
        
        path_monitors_experiments = os.path.join(path_experiments, "monitors")                            
        if not os.path.isdir(path_monitors_experiments):
            os.mkdir(path_monitors_experiments)
            
        path_logs = os.path.join(path_experiments, "logs")                            
        if not os.path.isdir(path_logs):
            os.mkdir(path_logs)

        env = RenderMonitor(env, 1, path_monitors_experiments, rrender=render)
        
        #env = DummyVecEnv([lambda: env])
          
        return env, path_monitors_experiments, experiment_monitor        
    
    def __create_results_dir(self):
        self.path_results_experiments = "results-{}-{}-{}-{}-{}-{}-{}-{}".format(self.total_timesteps, self.n_steps, self.batch_size, self.n_epochs, self.entropy_min, self.learning_rate, self.seed, self.act_func)
        self.path_results = mk_dir(self.results_dir, f"{self.path_results_experiments}{self.uuid}")
            
    def __save_trained_model(self, model, save_path, filename):            
        path_model_dir = mk_dir(save_path, "models")  
        path_model = f"{path_model_dir}/{filename}"
        print(f"Saving to {path_model_dir}")
        model.save(path_model)
            
    def inference(self, time_steps = 100, 
                        save_image_level = True, 
                        show_hud = False, 
                        render = False, 
                        n_experiments = 1, 
                        seeds = [1000], 
                        use_function_set_random_seed = True,
                        record_video = False):
        """
        Start inference of experiments.

        Args:
            time_steps (int, optional): Number of time steps for each experiments. Defaults to 100.
            save_image_level (bool, optional): Set if the image of window will be saved. Defaults to False.
            show_hud (bool, optional): Set if the status will be showed. Defaults to False.
            render (bool, optional): Set if the game window will be rendered. Defaults to False.
            n_experiments (int, optional): Number of experimentos. Defaults to 1.
            seed (list, optional): List of seeds for each experiment. This parameter must be used in the value generated for each learning. Defaults to [1000].
        """        
        for experiment in range(n_experiments):
        
            l = len(seeds)
            if (l > 1):
                self.seed = seeds[experiment]
            else:
                self.seed = seeds[0]
            if use_function_set_random_seed:
                if n_experiments > 1:
                    set_random_seed(self.seed)            
                           
            self.__create_results_dir()           
            self.__inference(time_steps = time_steps, 
                             experiment = experiment, 
                             save_image_level = save_image_level, 
                             show_hud = show_hud, render = render,
                             record_video = record_video)
        
    def __inference(self, time_steps = 100, 
                          experiment = 1, 
                          save_image_level = True, show_hud = False, render = False, record_video = False):
        
        self.__create_results_dir()
            
        path_results = self.path_results
        n_units = 64
        n_layers = 2
        kwargs = {}        
        
        act_func = self.act_func
        activation_fn = th.nn.Sigmoid
        if act_func == ActivationFunc.ReLU.value:
            activation_fn = th.nn.ReLU
        elif act_func == ActivationFunc.SIGMOID.value:
            activation_fn = th.nn.Sigmoid
        elif act_func == ActivationFunc.Tanh.value:
            activation_fn = th.nn.Tanh

        if (self.policy_kwargs is None):        
            self.policy_kwargs = dict(net_arch = [dict(pi=[64, 64], vf=[64, 64])], activation_fn=activation_fn)
        
        time_elapsed_agents = []

        for agent in self.agent:
                                    
            start_agent = timer()
            print("Start: ", start_agent)
            print()           
            muuid = uuid4()
        
            for env_name in self.envs:
                    
                main_dir = "Experiment 0{}-{}-{}-{}".format(1, agent, env_name, self.rl_algo)                                        
                path_experiments = mk_dir(path_results, main_dir) 
                
                scores_inf, episodes_inf, average_inf = [], [], []       
                all_time_elapsed                      = []
                map_counter                           = []                    
                
                for _rep in self.representations:            
                    
                    representation = _rep
                    rep_path = representation
                    path_monitors_experiments = ""
                    for _obs in self.observations:          
                        
                        plotResults = PlotResults()
                        
                        observation = _obs

                        path_rep   =  "{}-{}".format(representation, observation)                            
                        
                        inferente_name = "{}-{}".format("inference", experiment)
                        
                        path_inference = mk_dir(path_results, inferente_name)
                        
                        path_inference = mk_dir(path_inference, main_dir)
                                                    
                        path_rep    = mk_dir(path_inference, path_rep)
                        
                        mk_dir(path_rep, "best")                         

                        mk_dir(path_rep, "max_changes")                         
                        
                        mk_dir(path_rep, "worst")

                        path_map = mk_dir(path_rep, "map")
                        
                        mk_dir(path_map, "best")
                        
                        mk_dir(path_map, "worst")                                                       
                        
                        
                        filename_model = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(representation,
                                                                                observation,
                                                                                self.total_timesteps, 
                                                                                self.n_steps, 
                                                                                self.batch_size, 
                                                                                self.n_epochs,
                                                                                self.entropy_min,
                                                                                self.learning_rate, 
                                                                                self.seed, 
                                                                                self.act_func,
                                                                                agent,
                                                                                env_name)
                        model = None                                                
                        try:
                            
                            path_model_dir = mk_dir(path_results, "models")  
                            path_model = f"{path_model_dir}/{filename_model}"
                            #path_model = "D:\Models\EX8ZELDANARROW"
                            print(f"Loading {path_model}") 
                            model = RL_ALGO[self.rl_algo].load(path_model)
                            print("Model is loader...")                                                                                            
                            
                        except Exception:              
                            print("Error ao carregar o modelo")
                            #exit()
                            pass
                        finally:              
                            try:
                                pass                                    
                            except EOFError:
                                #del model
                                pass                                
                        
                        if not model is None:  

                            env, path_monitors_experiments, experiment_monitor  = self.__make_env(env_name, 
                                                                representation=representation, 
                                                                path_experiments=path_rep,
                                                                observation = observation,
                                                                agent = agent,
                                                                action_change=self.action_change,
                                                                action_rotate=self.action_rotate,
                                                                save_image_level=save_image_level,
                                                                show_hud=show_hud, 
                                                                render = render,                                                               
                                                                max_changes=self.max_changes)            
                            video_folder = ""
                            if record_video:
                                images = []
                                video_folder = mk_dir(path_results, "videos")  
                                
                                video_length = 100                                                                

                                obs = env.reset()

                                # Record the video starting at the first step
                                #env = VecVideoRecorder(env, video_folder,
                                 #                   record_video_trigger=lambda x: x == 0, video_length=video_length,
                                  #                  name_prefix=f"{agent}-{env_name}")

                                #env.close()

                            scores, episodes, average = [], [], []
                            
                            obs = env.reset()                        
                            print()
                            done = False
                            start = timer()
                            print("Start: ", start)
                            print()
                            time_elapsed = []
                            total_rewards = []                                
                            sum_rewards = 0
                            steps = 0                                
                            
                            if record_video:
                                img = env.render()

                            for e in range(time_steps):                            
                                obs = env.reset()
                                done = False                
                                
                                img = env.render()
                                
                                score = 0                                                
                                steps = 0
                                
                                while not done:
                                    
                                    if record_video:
                                        images.append(img)
                                    
                                    action, _states = model.predict(obs)                        
                                    obs, reward, done, info = env.step(action)
                                    score += reward
                                    sum_rewards += reward
                                    steps += 1
                                    
                                    if record_video:
                                        img = env.render()

                                    print()
                                    print("Representation: {}, Observation: {}".format(representation, observation))                                         
                                    print("Episode: {}, Score: {}, Total rewards: ".format(e, score, sum_rewards))                                                                                            

                                    if done:
                                        scores.append(score)
                                        episodes.append(e)
                                        average.append(sum(scores) / len(scores))                
                                        total_rewards.append(sum_rewards)

                            if record_video:                                
                                imageio.mimsave(f"{video_folder}/{agent}-{env_name}.gif", [np.array(img) for i, img in enumerate(images)], fps=30)

                            map_counter.append({"Representation" : representation, "Observation" : observation, "Total" : env.counter_done})
                            
                            inference = {"Episodes": episodes, "Scores" : scores, "Average": average, "Total Rewards" : total_rewards}                
                            df = pad.DataFrame(inference)
                            df.to_csv(path_inference+"/Inference"+_rep+"-"+_obs+".csv",  index=False)                                            
                            
                            plot(average, scores, episodes, path_experiments, "inference", _rep + ", Observation: " + observation, env_name)                                
                            end = timer()        
                            print("End: ", end)
                            time_ela = timedelta(seconds=end-start)
                            print("Time elapsed: ", time_ela)
                            
                            d = {"Representation": representation, "Observation" : _obs, "start": start, "end" : end, "time elapsed": time_ela}
                            
                            time_elapsed.append(d)                
                            all_time_elapsed.append(d)
                                            
                            df = pad.DataFrame(time_elapsed)
                            df.to_csv(path_experiments+"/Time elapsed.csv",  index=False)                            
                                        
                            scores_inf.append((_rep, _obs, scores))
                            episodes_inf.append((_rep, _obs, episodes))
                            average_inf.append((_rep, _obs, average))                                                        
                    
                    title = env_name
                    
                    df = pad.DataFrame(all_time_elapsed)
                    df.to_csv(path_inference+"/Time elapsed-"+title+".csv",  index=False)                              
                    
                    df = pad.DataFrame(map_counter)
                    df.to_csv(path_inference+"/MapCounter-"+title+".csv",  index=False)
                    info_mlp = "MLP: Units {}, Layers {}".format(n_units, n_layers)                        
                    t = "Quantidade de Mapas Gerados\n"+title+" - "+info_mlp 
                    
                    #plot_bar(df, path_inference, "Mapcounter-"+title, t, self.total_timesteps)
                    
                    plot_all_rewards(average_inf, scores_inf, episodes_inf, path_inference, "inference-rewards-all-"+title, title+" - "+info_mlp)
                    plot_all_average(average_inf, scores_inf, episodes_inf, path_inference, "inference-average-all-"+title, title+" - "+info_mlp)                   
                    results_plotter.plot_results([path_monitors_experiments], int(self.total_timesteps), results_plotter.X_TIMESTEPS, env_name)
                    
                    path_info = os.path.join(path_inference, _rep+"-"+_obs)    
                    path_info = os.path.join(path_info, "Info.csv")
                    
                    if os.path.exists(path_info):
                        print("Dados com arquivo")             
                        data_info = pd.read_csv(path_info, index_col=False)
                        plotResults.add(env_name, data_info)                                                        
                    else:                   
                        print("Dados sem arquivo")             
                        dt = []                                        
                        data_info = []
                        data_info.append(dt)                                               
                        plotResults.add(env_name, data_info)
                        
                    title = "Representação: {} \n {}".format(_rep, agent)
                    plotResults.plot_entropy(path_monitors_experiments, "Entropy-games"+_rep, title)
                                                
                    plotResults.plot_boxplot(path_monitors_experiments, "Entropy-games-boxplot"+_rep, title)                                                
            
            end_agent = timer()        
            print("End: ", end_agent)
            time_ela_agent = timedelta(seconds=end_agent-start_agent)
            print("Time elapsed: ", time_ela_agent)
            
            d = {"Agent": agent, "start": start_agent, "end" : end_agent, "time elapsed": time_ela_agent}
            
            time_elapsed_agents.append(d)                       
                            
            df = pad.DataFrame(time_elapsed_agents)
            filename_timeela = "{}/{}-{}-{}-{}.csv".format(self.path_results, "/Inference Time elapsed", agent, env_name, muuid) 
            df.to_csv(filename_timeela,  index=False)
            
    def __train(self, save_image_level = False, show_hud = False, render = False):
        
        path_results = self.path_results
        
        act_func = self.act_func
        activation_fn = th.nn.Sigmoid
        if act_func == ActivationFunc.ReLU.value:
            activation_fn = th.nn.ReLU
        elif act_func == ActivationFunc.SIGMOID.value:
            activation_fn = th.nn.Sigmoid
        elif act_func == ActivationFunc.Tanh.value:
            activation_fn = th.nn.Tanh

        if (self.policy_kwargs is None):        
            self.policy_kwargs = dict(net_arch = [dict(pi=[64, 64], vf=[64, 64])], activation_fn=activation_fn)
        
        time_elapsed_agents = []
        
        for agent in self.agent:
            
            start_agent = timer()
            print("Start: ", start_agent)
            print()
            muuid = uuid4()
                                
            for env_name in self.envs:
                    
                main_dir = "Experiment 0{}-{}-{}-{}".format(1, agent, env_name, self.rl_algo)                                        
                path_experiments = mk_dir(path_results, main_dir) 
                
                for _rep in self.representations:            
                    
                    representation = _rep
                    rep_path = representation
                    
                    for _obs in self.observations:          
                        
                        observation = _obs
                        
                        plotResults = PlotResults()
                        
                        params = {
                            "RL_ALG" : self.rl_algo,
                            "total_timesteps" : self.total_timesteps, 
                            "learning_rate" : self.learning_rate,                
                            "n_steps" : self.n_steps,
                            "gamma" : self.gamma,                
                            "batch_size" : self.batch_size,
                            "n_epochs" : self.n_epochs,
                            "seed" : self.seed,
                            "verbose" : verbose,                                
                            "policy_kwargs"   : self.policy_kwargs,                                                                
                            "observation": observation,                                                                        
                            "max_changes" : self.max_changes,                                
                            "entropy" : self.entropy_min,                                                                                
                            "reward_best_done_bonus"   :self.reward_best_done_bonus,
                            "reward_medium_done_bonus" : self.reward_medium_done_bonus,
                            "reward_low_done_bonus"    : self.reward_low_done_bonus,
                            "reward_entropy_penalty"   : self.reward_entropy_penalty,
                            "reward_change_penalty"    : self.reward_change_penalty,
                            "factor_reward" : self.factor_reward,                            
                            "board" : self.board
                        }
                                                                                    
                        info_params = []
                        info_params.append(params)
                            
                        df = pad.DataFrame(info_params)                             
                        
                        df.to_csv(path_experiments+"/params.csv", index=False)                                                          
                    
                        pathrep = mk_dir(path_experiments, rep_path) 
                        mk_dir(pathrep, "best")                         

                        mk_dir(pathrep, "max_changes")                         
                        
                        mk_dir(pathrep, "worst")

                        path_map = mk_dir(pathrep, "map")
                        
                        mk_dir(path_map, "best")
                        
                        mk_dir(path_map, "worst")                            
                        
                        env,path_monitors_experiments, experiment_monitor  = self.__make_env(env_name, 
                                                representation=representation, 
                                                path_experiments=pathrep,
                                                observation = observation,
                                                agent = agent,
                                                action_change=self.action_change,
                                                action_rotate=self.action_rotate,
                                                save_image_level=save_image_level,
                                                show_hud=show_hud, 
                                                render = render,                                                  
                                                max_changes=self.max_changes)
                        
                        model = RL_ALGO[self.rl_algo](
                            policy = self.policy,
                            env = env,              
                            seed = self.seed,
                            gamma = self.gamma,         
                            batch_size = self.batch_size,
                            n_epochs = self.n_epochs,   
                            n_steps = self.n_steps,                          
                            learning_rate = self.learning_rate,   
                            verbose = self.verbose,              
                            policy_kwargs = self.policy_kwargs
                        )
                                                                   
                        callback_log_dir = path_monitors_experiments                                                                            
                                            
                        try:
                            
                            saveOnBest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=callback_log_dir)
                                
                            model.learn(total_timesteps = self.total_timesteps, callback = saveOnBest_callback)
                            
                            filename_model = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(representation,
                                                                                    observation,
                                                                                    self.total_timesteps, 
                                                                                    self.n_steps, 
                                                                                    self.batch_size, 
                                                                                    self.n_epochs,
                                                                                    self.entropy_min,
                                                                                    self.learning_rate, 
                                                                                    self.seed, 
                                                                                    self.act_func,
                                                                                    agent,
                                                                                    env_name)
                            
                            self.__save_trained_model(model = model, save_path = path_results, filename=filename_model)
                            
                            results_plotter.plot_results([path_monitors_experiments], int(self.total_timesteps), results_plotter.X_TIMESTEPS, env_name)
                            plt.savefig(path_monitors_experiments+"/Monitor.png")
                            
                            experiment_monitor.end()
                                                                                            
                            path_info = os.path.join(path_experiments, _rep+"-"+_obs)    
                            path_info = os.path.join(path_info, "Info.csv")
                            
                            if os.path.exists(path_info):
                                print("Dados com arquivo")             
                                data_info = pd.read_csv(path_info, index_col=False)
                                plotResults.add(env_name, data_info)                                                        
                            else:                   
                                print("Dados sem arquivo")             
                                dt = []                                        
                                data_info = []
                                data_info.append(dt)                                               
                                plotResults.add(env_name, data_info)
                                
                            title = "Representação: {} \n {}".format(_rep, agent)
                            plotResults.plot_entropy(path_monitors_experiments, "Entropy-games"+_rep, title)
                                  
                            plotResults.plot_boxplot(path_monitors_experiments, "Entropy-games-boxplot"+_rep, title)                            
                                                                                                                    
                        except KeyboardInterrupt:              
                            pass
                        finally:              
                            try:
                                pass
                                # model.env.close()
                            except EOFError:
                                pass                                                         
            
            end_agent = timer()        
            print("End: ", end_agent)
            time_ela_agent = timedelta(seconds=end_agent-start_agent)
            print("Time elapsed: ", time_ela_agent)
            
            d = {"Agent": agent, "start": start_agent, "end" : end_agent, "time elapsed": time_ela_agent}
            
            time_elapsed_agents.append(d)                       
                            
            df = pad.DataFrame(time_elapsed_agents)
            filename_timeela = "{}/{}-{}-{}-{}.csv".format(self.path_results, "/Training Time elapsed", agent, env_name, muuid) 
            df.to_csv(filename_timeela,  index=False)    
    
    def learn(self, save_image_level:bool = False, 
                    show_hud:bool = False, 
                    render:bool = False,                     
                    n_experiments:int = 1, use_function_set_random_seed = True):        
       """
       Start experiment. If the parameter n_experiment is greater than 1, a random seed
       is used for each experiment.
        
       Args:
           save_image_level (bool, optional): Set if the image of window will be saved. Defaults to False.
           show_hud (bool, optional): Set if the status will be showed. Defaults to False.
           render (bool, optional): Set if the game window will be rendered. Defaults to False.
           n_experiments (int, optional): Number of experimentos. Defaults to 1.
       """
       if n_experiments > 1:
           self.seed = gen_random_number()
    
       time_elapsed = []
       for experiment in range(n_experiments):            
            start = timer()
            muuid = uuid4()
            print("Start: ", start)
            print()           
            if use_function_set_random_seed:
                set_random_seed(self.seed)
            self.__create_results_dir()           
            self.__train(save_image_level = save_image_level, show_hud = show_hud, render = render)
            
            end = timer()        
            print("End: ", end)
            time_ela = timedelta(seconds=end-start)
            print("Time elapsed: ", time_ela)
            
            d = {"Experiment": experiment, "seed" : self.seed, "start": start, "end" : end, "time elapsed": time_ela}
            
            time_elapsed.append(d)                       
                            
            df = pad.DataFrame(time_elapsed)
            filename_timeela = "{}/{}-{}.csv".format(self.results_dir, "/Time elapsed - experiments", muuid)
                        
            df.to_csv(filename_timeela,  index=False)                         
            
            if n_experiments > 1:
                self.seed = gen_random_number()

