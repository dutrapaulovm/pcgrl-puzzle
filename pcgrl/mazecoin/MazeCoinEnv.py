from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.mazecoin import MazeCoinGameProblem
from pcgrl.mazecoin import *
from pcgrl.Utils import *
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.callbacks import BasePCGRLCallback

class MazeCoinEnv(BasePCGRLEnv):
    def __init__(self, 
                seed = None, 
                show_logger = False, 
                save_logger = False, 
                save_image_level = False, 
                path = "", 
                rep = None, 
                action_change = False, 
                piece_size = 8, 
                board = (3, 2), 
                #tile_size = 64,
                env_rewards = False,
                path_models = "mazecoin/",
                callback = BasePCGRLCallback()):        
        
        self.rep = rep          
        self.cols = board[0] * min(8, piece_size * 2) #Col
        self.rows = board[1] * min(8, piece_size * 2) #Row                
        game = MazeCoinGameProblem(cols = self.cols, rows = self.rows, border = True)#, tile_size=tile_size)
        game.scale    = 2            
        self.action_change = action_change
        super(MazeCoinEnv, self).__init__(seed = seed, game = game, env_rewards=env_rewards,
                                          save_image_level = save_image_level, 
                                          save_logger=save_logger, 
                                          show_logger=show_logger, 
                                          rep=rep, path=path, 
                                          piece_size = piece_size, 
                                          board = board, 
                                          path_models = path_models, callback=callback)
        self.current_reward   = 0        
        self.counter_done     = 0        
        self.info = {}  
        self.name = "MazeCoinEnv"