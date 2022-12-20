from gym.utils import seeding
#from gym.envs.classic_control import rendering
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pygame.image import save
from pcgrl.AgentBehavior import * 
from pcgrl.Agents import * 
from pcgrl.Utils import *
from pcgrl.log import ResultsWriter
from pcgrl.callbacks import *


class PCGRLEnv(gym.Env):
    """
    Create a new PCGRL environment
    """
    def __init__(self, seed = None, game = None,
                       show_logger = False,  save_logger = False,
                       path = "", name = "PCGRLEnv", 
                       callback = BasePCGRLCallback()):
        super().__init__()
        self.seed(seed=seed)
        self.id                   = self.np_random.randint(1, 10000000) #Id Env                
        self.name = name
        self.game                 = game
        self.game.np_random       = self.np_random
        self.is_done_success      = False #Indicate if episode terminate with success       
        self.use_done_max_changes = True #Indicate if use max changes                
        self.hist = {}
        self.path = path
        self.show_logger = show_logger
        self.save_logger = save_logger
        self.render_gym = False
        self.viewer = None
        
        if (game is None):
            raise ValueError('Game can''t is none')

        self.game.generate_map(self.np_random)
        self.game.reset(self.np_random)                        
        self.game.env = self
        self.np_random = self.np_random
        self.last_map = self.game.map
        self.num_tiles = len(self.game.get_tiles())              
        self.width      = self.game.get_width()
        self.height     = self.game.get_height()
        self.state_w    = self.game.get_state_width()
        self.state_h    = self.game.get_state_height()
        self.rows       = self.game.get_rows()
        self.cols       = self.game.get_cols()        
        self.counter_changes    = 0 #Store the number of changes
        self.iterations = 0 #Store the number of iterations
        self.counter_changes = 0 #Store the number of changes
        self.last_counter_changes = 0        
        w = self.width / self.state_w
        h = self.height / self.state_h
        self.max_changes = max(int(0.70 * w * h), 1) #Maximum number of changes until the enviroment be finished
        self.max_iterations = self.max_changes * w   #Maximum number of iterations until the enviroment be finished
        self.max_reset = 1                           #Maximum number of reset until the environment be finished        
        self.current_stats = {}           
        self.finished_changes = False
        self.resetted = False
        self.counter_done = 0
        self.counter_done_max_changes = 0
        self.counter_done_interations = 0        
        self.counter_reset = 0       
        self.info = {}
        self.observation_space = self.create_observation_space()
        self.action_space = self.create_action_space()                

        self._reward = 0

        self.current_action = []
        self.old_stats = []
        self.current_stats = []
        columnsnames = {"reward_game",  "reward",  "discount_reward",  "bonus_factor",  "experience_bonus", "done_bonus", "done_penalty", "reward_experience_bonus", "changes_penalty",  "piece_penalty", "counter_changes", "counter_done", "representation", "counter_done_interations", "counter_done_max_changes", "is_done_success", "agent", "segments", "entropy", "entropy_map","historical", 'rewards_sum', 'discount_rewards'}
        
        self.results_writer = None        
        
        if self.path is not None:
            self.results_writer = ResultsWriter(
                filename="Info.csv",
                path=self.path,                 
                fieldsnames=columnsnames
            )
        else:
            self.results_writer = None        
        
        self.callback = callback       
        if (callback is not None):
            self.callback.on_create_env()

    def reset(self):      
        self.info = {}
        self.hist = {}
        self.hist['rewards'] = []
        self.hist['action'] = []
        self.game.reset(self.np_random)        
        self.resetted = True                
        self.iterations = 0        
        self.changes   = 0      
        self.callback.on_reset()
        return {}
    
    def create_action_space(self):        
        return spaces.Discrete(2)

    def create_observation_space(self):
        self.observation_space = None
        return self.observation_space        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
    
    def sample_actions(self):
        action = self.action_space.sample_action()            
        return action
    
    def step(self, actions = None): 
        """
        Take a step given the action
        Returns:
            a tuple of:
            - state (numpy.ndarray) the current state after take a action
            - reward (float) the reward earn by taking the action
            - done (bool) a flag that indicate whether the episode has ended
            - info (dict) a dictionary of extra information
        
        """
        if (self.game is None):
            raise ValueError('Game can''t is none') 

        self.callback.on_before_step(actions)
        
        if (self.show_logger):
            clear_console()
            print()             
            
        clear_console()
        print("Env name: ", self.name)                               

        self.iterations += 1

        self.old_stats = self.current_stats

        obs = self._do_step(actions)
        
        done = self._get_done(actions)

        self._compute_extra_rewards()

        self._reward = self._calc_reward()
        
        #self.hist['rewards'].append(self._reward)
        self.hist['action'].append(actions)        

        self.info["historical"] = self.hist

        if (self.is_done_success):
            
            self.info['rewards_sum'] = sum(self.hist['rewards'])
            #self.info['discount_rewards'] = self.discount_rewards(self.hist['rewards'], discount_rate = 0.97).mean()

            if (self.save_logger):                                      
                if (not self.path is None and not os.path.exists(self.path)):
                    print("Logger not saved. Path logger not defined...")
                    raise ValueError('Set a valid path to save logger.')

                if (not self.results_writer is None):
                    self.results_writer.write_row(self.info)

        if (self.show_logger):
            print(self.info)
            print(self.game.get_information())

        self._after_step(self._reward, done)
        
        self.callback.on_after_step(actions, self._reward, done, self.info, self.hist)

        return obs, self._reward, done, self.info

    def _after_step(self, reward, done):
        """
        Perform after step
        """
        return reward

    def _do_step(self, action):
        """
        Perform the action in the environment
        """
        return {}

    def _compute_extra_rewards(self):
        pass

    def _calc_reward(self):                
        """
        Return the reward after steps
        """
        return 0

    def _get_done(self, actions = None):
        """Return if the episode is over, True if over, False otherwise."""
        return False        

    def close(self):

        """Close the environment."""        
        if self.game is None:
            raise ValueError('Environment has already been closed.')

        self.game.close()

        self.game = None    

    def render(self, mode='rgb_array', tick=60):        
        """        
        Render the environment.
        
        Parameters:
            mode (str): the mode to render:
            - human: render to the current display
            - rgb_array: Return an numpy.ndarray representing RGB values
        
        Returns:
            rgb_array (any[]) a numpy array representing RGB values
        """
        metadata = {'render.modes': ['human', 'rgb_array']}
        a = None        
        a = self.game.render(mode, tick)        
        """
        if self.render_gym:
            if self.viewer is None:                
                self.viewer = rendering.SimpleImageViewer()        
                self.viewer.imshow(a)       
        """
        return a

    def close(self):
        if self.viewer:
            self.viewer.close()        
            self.viewer = None
    
    #discounted cumulative reward
    def total_rewards(self, rewards, discount_rate=0.99, gamma = 1):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
            discount_rate *= gamma
        return cumulative_rewards    
    
    #discounted cumulative reward
    def discount_rewards(self, rewards, discount_rate=0.99):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(self, all_rewards, discount_rate=0.99):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) /reward_std for discounted_rewards in all_discounted_rewards]        
        
    @property
    def _finished_iterations(self):
        """Return if iterations is finished."""
        return self.iterations >= self.max_iterations

    @property
    def _finished_changes(self):
        """Return if changes is finished."""
        return self.counter_changes >= self.max_changes