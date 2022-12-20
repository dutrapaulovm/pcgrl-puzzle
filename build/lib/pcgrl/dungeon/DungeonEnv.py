from time import sleep
from gym.utils import seeding
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.dungeon import DungeonGameProblem
from pcgrl.dungeon import *
from pcgrl.Utils import *
from pcgrl.callbacks import BasePCGRLCallback

class DungeonEnv(BasePCGRLEnv):
    
    def __init__(self, 
                seed = None, 
                show_logger = False, 
                save_logger = False,
                save_image_level = False,
                path = "", 
                rep = None, 
                action_change = False, 
                piece_size = 8, 
                board = (3,2),
                env_rewards=False,
                path_models = "dungeon/",
                callback = BasePCGRLCallback()):        

        self.cols = board[0] * min(8, piece_size * 2) #Col
        self.rows = board[1] * min(8, piece_size * 2) #Row  
        
        game = DungeonGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale = 2
        self.action_change = action_change
        super(DungeonEnv, self).__init__(seed = seed, game = game, env_rewards=env_rewards, save_image_level = save_image_level, save_logger = save_logger, show_logger=show_logger, rep=rep, path=path, piece_size = piece_size, board = board, path_models = path_models, callback=callback)
        
        self.current_reward = 0        
        self.counter_done    = 0        
        self.info = {}        
        self.name = "DungeonEnv"
    """
    def create_action_space(self):        
        path_piece = os.path.join(PCGRLPUZZLE_MAP_PATH, "dungeon/")        
        self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(self.piece_size, self.piece_size), rep = self.representation, path_pieces = path_piece, action_change=self.action_change)
        self.max_cols_piece = self.agent.max_cols
        self.max_rows_piece = self.agent.max_rows            
        self.action_space   = self.agent.action_space
        self.max_segment = int( self.max_cols_piece * self.max_rows_piece )
        self._reward_agent = 0
        return self.action_space
    """