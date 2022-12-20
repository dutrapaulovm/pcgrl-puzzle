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
                path = "", 
                rep = None, 
                action_change = False, 
                piece_size = 8, 
                board = (3,2), 
                env_rewards = False,
                path_models = "mazecoin-lowmodels/",
                callback = BasePCGRLCallback()):        
        
        self.rep = rep          
        self.cols = board[0] * min(8, piece_size * 2) #Col
        self.rows = board[1] * min(8, piece_size * 2) #Row               
        game = MazeCoinGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale    = 2            
        self.action_change = action_change
        super(MazeCoinLowMapsEnv, self).__init__(seed = seed, game = game, 
                                                 env_rewards=env_rewards,
                                                 save_image_level = save_image_level,
                                                 save_logger=save_logger, 
                                                 show_logger=show_logger, rep=rep,
                                                 path=path, piece_size = piece_size, 
                                                 board = board,
                                                path_models = path_models, callback = callback)
        self.current_reward   = 0        
        self.counter_done     = 0        
        #self.cols = 24
        #self.rows = 16 
        self.name = "MazeCoinLowMapsEnv"               
    """
    def create_action_space(self):
        #path_piece = os.path.abspath(os.path.join("mazecoin", os.pardir))
        #path_piece = os.path.join(path_piece, "pcgrl/maps/mazecoin")
        path_piece = os.path.join(PCGRLPUZZLE_MAP_PATH, "mazecoin-lowmodels/")        
        #self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(8, 8), rep = self.representation, path_pieces = path_piece, action_change=self.action_change)
        self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(self.piece_size, self.piece_size), rep = self.representation, path_pieces = path_piece, action_change=self.action_change)
        self.max_cols_piece = self.agent.max_cols
        self.max_rows_piece = self.agent.max_rows            
        self.action_space   = self.agent.action_space
        self.max_segment = int( self.max_cols_piece * self.max_rows_piece )
        self._reward_agent  = 0
        return self.action_space
    """