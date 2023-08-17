import pandas as pad
from pcgrl.Agents import *
from collections import OrderedDict
from gym import spaces
from enum import Enum
from pcgrl.Utils import *
from pcgrl.PCGRLEnv import PCGRLEnv
from collections import deque

from pcgrl.callbacks import BasePCGRLCallback
from pcgrl.Rewards import * 
from pcgrl import PCGRLPUZZLE_MAP_PATH

sign = lambda x: math.copysign(1, x)

class Experiment(Enum):
    
    AGENT_SS               = "SS"    
    AGENT_HHP              = "HHP"    
    AGENT_HHPD             = "HHPD"
    AGENT_HQHPD            = "HQHPD"            
    AGENT_HEQHP            = "HEQHP"            
    AGENT_HEQHPD           = "HEQHPD"            
    AGENT_HEQHPEX          = "HEQHPDEX"            
    
    def __str__(self):
        return self.value         

class BasePCGRLEnv(PCGRLEnv):

    def __init__(self, seed = None, 
                game = None, factor_reward = 1.0, 
                piece_size = 8, board = (3, 2), 
                show_logger = False, save_logger = False, 
                save_image_level = False, path = ".\Results",
                rep = None, name = "BasePCGRLEnv",
                action_change = False,
                action_rotate = False,
                env_rewards = False,
                rendered = False,
                agent = None,
                max_changes = 61,
                reward_change_penalty = None,
                path_models = None, extra_actions = {}, callback = BasePCGRLCallback(),
                reward_function = RewardFunction()):
        
        self.action_change  = action_change
        self.action_rotate  = action_rotate
        self.agent_behavior = None
        self.segment        = 0
        self.max_segment    = 0
        self.representation = rep  
        self.is_render      = rendered
        self.piece_size     = piece_size            
        self.board = board        
        self.path_models = path_models        
        self.extra_actions = extra_actions
        super(BasePCGRLEnv, self).__init__(name = name, 
                                           seed = seed, 
                                           game = game, 
                                           max_changes = max_changes,
                                           save_logger = save_logger, 
                                           show_logger=show_logger, 
                                           path=path, 
                                           callback = callback)                
        self.factor_reward    =  factor_reward
        self.reward_game      = 0        
        self.exp_rpg          = 0.01
        self.max_exp_rpg      = 0.80        
        self.experience_inc   = 0.0002
        self.counter_done     = 0
        self.save_image_level = save_image_level                                              
        self._reward = 0
        self.reward_best_done_bonus = 50
        self.reward_medium_done_bonus = 10
        self.reward_low_done_bonus = 0
        self.reward_entropy_penalty = 0
        self._cumulative_reward = 0        
        self.last_pieces = []        
        self._last_rewards = 0
        self.reward_change_penalty = reward_change_penalty
        
        #obs = round(entropy(np.arange(board[0] * board[1]).reshape(board[1], board[0])), 2)
        self.max_entropy = calc_entropy(size=board, n_repetead=0) #round(entropy(obs), 2)          
        self.entropy_min = calc_entropy(n_repetead = 2, size=board)                    

        self.max_segment = 6
        self.exp = agent
        self.current_piece  = []
        self.env_rewards     = env_rewards
        self.reward_function = reward_function

        if not self.reward_function is None:
            self.reward_function.env = self
    
    def create_action_space(self):           
        path_piece = os.path.join(PCGRLPUZZLE_MAP_PATH, self.path_models)            
        
        self.agent_behavior  = LevelDesignerAgentBehavior(env = self, 
                            piece_size=(self.piece_size, self.piece_size), 
                            rep = self.representation, 
                            path_pieces = path_piece, 
                            action_change=self.action_change, 
                            action_rotate=self.action_rotate,
                            extra_actions=self.extra_actions)

        self.max_cols_piece = self.agent_behavior.max_cols
        self.max_rows_piece = self.agent_behavior.max_rows            
        self.action_space   = self.agent_behavior.action_space
        self.max_segment = int( self.max_cols_piece * self.max_rows_piece )
        self._reward_agent  = 0
        return self.action_space                 

    def create_observation_space(self):                      
        width = self.game.get_cols()
        height = self.game.get_rows()
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=self.num_tiles-1, dtype=np.uint8, shape=(height, width))
        })
        return self.observation_space

    def reset(self): 
        super().reset()
        self.segment = 0
        self.game.reset(self.np_random)                          
        self.counter_changes   = 0          
        self.current_stats = self.agent_behavior.get_stats()
        obs = OrderedDict({
            "map"   : self.game.map.copy()
        })
        self._reward = 0
        self._cumulative_reward = 0
        self.agent_behavior.reset()
        self.last_pieces = self.agent_behavior.grid
        self.current_piece = []

        if not self.reward_function is None:
            self.reward_function.reset()

        return obs

    def _do_step(self, action):

        self.old_stats = self.current_stats        
        
        self._reward_agent, change, self.current_piece = self.agent_behavior.step(action)
        
        obs = self.agent_behavior.get_current_observation({})

        if change > 0:
            self.counter_changes += change            
            self.current_stats = self.agent_behavior.get_stats()
            self.segment += 1

        return obs

    def get_positions(self, tiles, map):        
        max_row = map.shape[0]
        max_col = map.shape[1]
        new_map = []
        for row in range(max_row):
            for col in range(max_col):
                id = int(map[row][col])
                if id in tiles:
                    new_map.append((row, col))
        return new_map

    def _reward_distance(self, segments):
        
        map_segments = np.array(segments)
        n_segments = map_segments.shape[1]        
        map_segments = set(map_segments.flatten())            

        reward_m = 0
        reward_e = 0

        for segment in map_segments:
            positions = self.get_positions([segment], segments)
            #print(positions)        
            if len(positions) > 1:    
                pos_init = positions[0]
                for row, col in positions:                    
                    reward_e += (n_segments - euclidean_distance(pos_init, (row, col)))        
        return -reward_e

    def  _compute_extra_rewards(self):
        self.reward_game, rewards_info = self.game.compute_reward(self.current_stats, self.old_stats)                

    def _after_step(self, reward, done):   
        
        if self.is_done_success:

            self.game.render_map()
            self.last_counter_changes = self.counter_changes
            filename_png = "{}{}-{}.png".format("MapEnvTraining", str(self.counter_done), self.exp)
            
            if (self._reward > 0):                                
                self.counter_done += 1
                if (not self.path is None):
                    if self.save_image_level:
                        path = self.path + "/best/"+filename_png
                        self.game.save_screen(path)
            else:
                self.counter_done += 1
                if (not self.path is None):
                    if self.save_image_level:
                        path = self.path + "/worst/"+filename_png
                        self.game.save_screen(path)

            if (not self.path is None):
                if self.save_image_level:                                
                    df = pad.DataFrame(self.game.map)
                    #if (self._reward > 0):
                    df.to_csv(self.path + "/map/best/Map"+str(self.counter_done)+".csv", header=False, index=False)
                    #else:
                    # df.to_csv(self.path + "/map/worst/Map"+str(self.counter_done)+".csv", header=False, index=False)
        
        self.info["counter_changes"] = self.counter_changes+1
        self.info["counter_done"]    = self.counter_done
        self.info["representation"]  = self.representation
        if self.is_render:
            self.game.render_map()

    def _get_done(self, actions = None):
        
        done_game = self.game.is_done(self.current_stats)
        self.is_done_success = done_game and self.agent_behavior.is_done()
        print("Done: ", self.is_done_success)
        done = self.is_done_success

        if (self._finished_iterations):
            self.counter_done_interations += 1

        if (self._finished_changes):
            self.counter_done_max_changes += 1                                    
        
        if (self._finished_changes) and self.use_done_max_changes and not self.is_done_success:
            done = True            
            
        self.info["counter_done_interations"] = self.counter_done_interations
        self.info["counter_done_max_changes"] = self.counter_done_max_changes
        self.info["is_done_success"] = self.is_done_success
        self.info["agent"] = self.agent_behavior.get_info()

        if (self.is_done_success):
            self.last_map = self.game.map            
            self.last_pieces = self.agent_behavior.grid
        
        self.info["segments"] = self.agent_behavior.grid.flatten()       
        
        if (self.is_done_success or done):
            self.info["entropy"] = entropy(self.agent_behavior.grid)
            self.info["entropy_map"] = entropy(self.game.map)
            
        return done
    
    @property
    def _reward_change_entropy_penalty(self):
        reward = 0
                        
        if self.reward_change_penalty is None:
            reward = self._changes_entropy_penalty * self.factor_reward
        else:            
            if (sign(self._changes_entropy_penalty) == 1):
                reward = -self._changes_entropy_penalty            
            else:    
                reward = self.reward_change_penalty * self.factor_reward
                
        return reward        

    @property
    def _reward_entropy_penalty(self):
        
        reward = 0
        
        if (sign(self.reward_entropy_penalty) == 1):
            reward = -self.reward_entropy_penalty * self.factor_reward
        elif (sign(self.reward_entropy_penalty) == 0):
            reward = -self.max_entropy * self.factor_reward
        else:
            reward = self.reward_entropy_penalty * self.factor_reward
        
        return reward

    def _calc_reward_newtest(self):  

        reward = 0

        kwargs = {
            'segments'      : self.agent_behavior.grid,
            'entropy_min'   : self.entropy_min,
            'factor_reward' : self.factor_reward,
            'agent_reward'  : self._reward_agent,
            'max_entropy'   : self.max_entropy,
            'entropy_min'   : self.entropy_min
        }

        if (not self.is_done_success):
            reward = self.change_penalty()
        else: 
            reward = self.reward_function.compute_reward(**kwargs)

        return reward
    
    def _calc_reward(self):  
        
        reward = 0
        kwargs = {
            'segments'      : self.agent_behavior.grid,
            'entropy_min'   : self.entropy_min,
            'factor_reward' : self.factor_reward,
            'agent_reward'  : self._reward_agent
        }

        if (self.exp == Experiment.AGENT_SS.value):            
            if (not self.is_done_success):
                reward = self.change_penalty()
            else:   
                simpleReward = SimpleReward(env = self)
                reward = simpleReward.compute_reward(**kwargs)                
                #reward = self.max_entropy * self.factor_reward
        elif (self.exp == Experiment.AGENT_HHP.value):
            if (not self.is_done_success):
                reward = self.change_penalty()
            else:
                entropyQuality = Entropy(env = self)
                reward = entropyQuality.compute_reward(**kwargs)                
                #reward = entropy(self.agent_behavior.grid) * self.factor_reward
        elif (self.exp == Experiment.AGENT_HHPD.value):            
            if (not self.is_done_success):
                reward = self.change_penalty()
            else:
                reward = (entropy(self.agent_behavior.grid) + self._done_bonus) * self.factor_reward
        elif (self.exp == Experiment.AGENT_HEQHP.value):
            if (not self.is_done_success):
                reward = self.change_penalty()
            else:
                entropyQuality = EntropyQuality(env = self)
                reward = entropyQuality.compute_reward(**kwargs)
                #x = math.pi
                #e = entropy(self.agent_behavior.grid)
                #r = (e**x - self.entropy_min**x)                
                #f = 1
                #reward = (((r + sign(r)) * f)) * self.factor_reward                    
        elif (self.exp == Experiment.AGENT_HEQHPEX.value):
            if (not self.is_done_success):
                reward = self.change_penalty()
            else: 
                entropyQuality = EntropyQuality()
                reward = entropyQuality.compute_reward(**kwargs)

            reward += self._reward_agent
        """
        elif (self.exp == Experiment.AGENT_HEQHPD.value):
            if (not self.is_done_success):
                reward = self.change_penalty()
            else: 
                x = math.pi
                e = entropy(self.agent_behavior.grid)                
                r = (e**x - self.entropy_min**x)                
                f = 10
                reward = (((r + sign(r)) * f)) * self.factor_reward
                reward = reward + self._reward_distance(self.agent_behavior.grid)
        """        
        if (self.env_rewards):
            self._compute_extra_rewards()
            reward += self.reward_game
        
        print("Agent: ", self.exp)
        print("Reward: ", reward)   
        print("Max Entropy", self.max_entropy)     
        print("Min Entropy", self.entropy_min)     
        print("Entropy", entropy(self.agent_behavior.grid))
        print("Pieces: ", self.agent_behavior.grid)
        print("Action change: ", self.action_change)
        print("Changes: ", self.counter_changes)
        print("Piece Penalty: ", self._segment_penalty)                                                            
        rd = self._reward_distance(self.agent_behavior.grid)
        rnei = self._reward_agent#self.reward_neighbors(self.agent_behavior.grid)
        print("Rewards Distance: " , rd)
        print("Reward neighbors: " , rnei)

        self.info["reward_game"] = self.reward_game
        self.info["reward"] = reward        
        self.info["bonus_factor"] = self._bonus_factor
        exp = self._experience_bonus
        self.info["experience_bonus"] = exp
        self.info["reward_experience_bonus"] = exp * reward
        self.info["done_bonus"] = self._done_bonus        
        #self.info["changes_penalty"] = self._changes_penalty
        self.info["piece_penalty"] = self._segment_penalty
        self.info["reward_distance"] = rd
        self.info["reward_neighbors"] = rnei
                
        self.hist['rewards'].append(reward)
                
        if (self.is_done_success):
            discount_rewards = self.discount_rewards(self.hist['rewards'], discount_rate = 0.99)
            self.info['discount_rewards'] = discount_rewards.mean()
                                
        self.add_reward(reward)

        return reward 

    def change_penalty(self):
        reward = 0
        if self.reward_change_penalty is None:
            reward = self._changes_entropy_penalty * self.factor_reward
        else:
            reward = self.reward_change_penalty * self.factor_reward        

        return reward

    def reward_neighbors(self, segments):
        n, m = segments.shape
        map_segments = np.array(segments)        
        map_segments = list(map_segments.flatten())                    
        positions = self.get_positions(map_segments, segments)

        reward = 0
        
        for row, col in positions:
            segment = segments[row][col]
            nei = neighbors(row, col, n-1, m-1)                        
            for r, c in nei:                
                if (segments[r][c] != -1) and segments[r][c] == segment and (row != r or col != c):
                    reward += -2

        return reward
        
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

    @property
    def _finished_segments(self):
        return self.segment >= self.max_segment

    @property
    def _segment_penalty(self):        
        r = 0
        """
        c = 0                                
        for p in range(len(self.agent.pieces)-1):
            js = js_divergence(self.agent.pieces[p], self.current_piece)        
            if (js <= 0):            
                c += 1

        if (c > 0):
            a = self.max_segment
            b = c + 1 + self.entropy_min
            r = -(1 / math.sqrt( a / b ) * b )
        else:
            r = 0
        """
        return r    
    
    @property
    def _experience_bonus(self):
        """Return the experience bonus earned to multiple the reward earned."""
        bonus = 1.0 
        if (self.exp_rpg < self.max_exp_rpg) and (self.is_done_success):
            self.exp_rpg += self.experience_inc
        bonus = (1 + self.exp_rpg)
        return bonus
    """
    @property
    def _changes_penalty(self):
        Return the reward for changes.
        _reward = (self.counter_changes / self.max_changes)                        
        return -_reward    
    """

    @property
    def _changes_entropy_penalty(self):
        """Return the reward for changes."""        
        _reward = (self.max_entropy / self.max_changes)        
        return -_reward
            
    @property
    def _bonus_factor(self):
        """Return the bonus factor earned to multiple the reward earned."""
        bonus = 1.0
        total_pieces = min( ( (self.max_cols_piece * self.max_rows_piece) * 2) + 5, self.max_changes)
        if self.is_done_success and self.counter_changes <= total_pieces:
            bonus = 1 / math.sqrt((self.counter_changes / self.max_changes))
        return bonus
        
    @property
    def _done_bonus(self):
        """Return the reward earned by done if not goal objective."""
        r = 0.0
        #total_pieces = min( ((self.max_cols_piece * self.max_rows_piece) * 2) + 5, self.max_changes)
        if self.is_done_success: #and self.counter_changes <= total_pieces:            
            e = round(entropy(self.agent_behavior.grid), 2)
            if e >= self.max_entropy:
                r += self.reward_best_done_bonus
            elif e >= self.entropy_min:
                r += self.reward_medium_done_bonus
            else:
                r += self.reward_low_done_bonus
                                 
        return r   
    
    def _entropy_distance(self, m, w = 0.5):
        """Calculate the value that measure de penalty of entropy 

        Args:
            m (_type_): The value of entropy
            w (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        return (math.pi / (m + w) ) * self.entropy_min