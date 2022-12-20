from time import sleep
from gym.utils import seeding
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pcgrl.BasePCGRLEnv import *
from pcgrl.combat import CombatGameProblem
from pcgrl.combat import *
from pcgrl.Utils import *

class CombatEnv(BasePCGRLEnv):
    def __init__(self, seed = None, save_logger = False, save_image_level = False, path = "", rep = None):
        
        self.rep = rep  
        game = CombatGameProblem(cols = 24, rows = 16, border = True)
        game.scale = 2
        
        super(CombatEnv, self).__init__(seed = seed, game = game, save_image_level=save_image_level, save_logger=save_logger)
        
        self.save_logger = save_logger
        self.current_reward = 0
        self.use_RPG = False
        self.exp_rpg        = 0.01
        self.max_exp_rpg    = 0.80        
        self.experience_inc = 0.002
        self.counter_done    = 0
        self.path = path
        self.save_image_level = save_image_level
        self.info = {}

    def create_action_space(self):
        path_piece = os.path.abspath(os.path.join("combat", os.pardir))
        path_piece = os.path.join(path_piece, "pcgrl/maps/combat")
        self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(8, 8), rep = self.rep, path_pieces = path_piece)        
        self.max_cols_piece = self.agent.generator.max_cols
        self.max_rows_piece = self.agent.generator.max_rows            
        self.action_space = self.agent.action_space
        self._reward_agent = 0
        return self.action_space