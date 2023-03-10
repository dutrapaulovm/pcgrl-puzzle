from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.mazecoin import MazeCoinGameProblem
from pcgrl.mazecoin import *
from pcgrl.Utils import *
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.callbacks import BasePCGRLCallback

class MazeCoinLowMapsEnv(BasePCGRLEnv):
    def __init__(self, 
                seed = None, 
                show_logger = False, 
                save_logger = False, 
                save_image_level = False, 
                rendered = False,
                path = "", 
                rep = None, 
                action_change = False, 
                action_rotate = False, 
                agent = None,
                reward_change_penalty = None,
                piece_size = 8, 
                board = (3,2), 
                env_rewards = False,
                path_models = "mazecoin-lowmodels/",
                callback = BasePCGRLCallback()):        
        
        self.rep = rep          
        self.cols = board[0] * min(piece_size, piece_size * 2) #Col
        self.rows = board[1] * min(piece_size, piece_size * 2) #Row               
        game = MazeCoinGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale    = 2            
        self.action_change = action_change
        self.action_rotate = action_rotate
        super(MazeCoinLowMapsEnv, self).__init__(seed = seed, game = game, 
                                                 env_rewards=env_rewards,
                                                 save_image_level = save_image_level,
                                                 save_logger=save_logger, 
                                                 show_logger=show_logger, rep=rep,
                                                 path=path, piece_size = piece_size, 
                                                 board = board,   
                                                 action_change=action_change,    
                                                 action_rotate=action_rotate,                                          
                                                 rendered=rendered,
                                                 agent=agent,
                                                 reward_change_penalty = reward_change_penalty,
                                                 path_models = path_models, callback = callback)
        self.current_reward   = 0
        self.counter_done     = 0
        self.name = "MazeCoinLowMapsEnv"              