from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.zelda import ZeldaGameProblem
from pcgrl.zelda import *
from pcgrl.Utils import *
from pcgrl.callbacks import BasePCGRLCallback

from pcgrl import PCGRLPUZZLE_MAP_PATH
class ZeldaLowMapsEnv(BasePCGRLEnv):
    def __init__(self, 
                 seed = None, 
                 show_logger = False, 
                 save_logger = False, 
                 save_image_level = False, 
                 path = "", 
                 rep = None, 
                 rendered = False,
                 action_change = False, 
                 action_rotate = False,
                 agent = None,
                 max_changes = 61,
                 reward_change_penalty = None,                 
                 piece_size = 8, 
                 board = (3, 2), 
                 path_models = "zelda-lowmodels/",
                 env_rewards = False,
                 callback = BasePCGRLCallback()):        
        
        self.rep = rep
        self.rows = board[0] * min(piece_size, piece_size * 2) #Row               
        self.cols = board[1] * min(piece_size, piece_size * 2) #Col
        game = ZeldaGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale = 2            
        self.action_change = action_change
        self.action_rotate = action_rotate        
        super(ZeldaLowMapsEnv, self).__init__(seed = seed, game = game, 
                                       env_rewards = env_rewards,
                                       save_image_level = save_image_level,
                                       action_change = action_change,
                                       action_rotate = action_rotate,                                          
                                       reward_change_penalty = reward_change_penalty,
                                       agent = agent,
                                       max_changes=max_changes,
                                       rendered = rendered,
                                       save_logger = save_logger, 
                                       show_logger = show_logger, rep=rep,
                                       path = path, piece_size = piece_size, 
                                       board = board,
                                       path_models = path_models, callback = callback)
        self.current_reward   = 0
        self.counter_done     = 0        
        self.info = {}         
        self.name = "ZeldaEnv"
        
    """
    def create_action_space(self):
        #path_piece = os.path.abspath(os.path.join("zelda", os.pardir))
        #path_piece = os.path.join(path_piece, "pcgrl/maps/zelda")              
        path_piece = os.path.join(PCGRLPUZZLE_MAP_PATH, "zelda/")
        extra_actions = {}
        #self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(8, 8), rep = self.representation, path_pieces = path_piece, action_change=self.action_change, extra_actions = extra_actions)
        self.agent  = LevelDesignerAgentBehavior(env = self, piece_size=(self.piece_size, self.piece_size), rep = self.representation, path_pieces = path_piece, action_change=self.action_change, extra_actions = extra_actions)
        self.max_cols_piece = self.agent.max_cols
        self.max_rows_piece = self.agent.max_rows            
        self.action_space   = self.agent.action_space
        self.max_segment = int( self.max_cols_piece * self.max_rows_piece )
        self._reward_agent  = 0
        return self.action_space
    """        
    """
    def _do_step(self, action):
    
        self.old_stats = self.current_stats        

        self._reward_agent, change, self.current_piece = self.agent.step(action)
    """
        
    """
        if (self.agent.is_done()):
            
            tiles = [Coin.ID, Key.ID, Enemy.ID, Weapon.ID]
            map_locations = self.game.get_tile_positions(tiles, self.game.map)
            for row, col in map_locations:
                self.game.change_tile(col * TILE_SIZE, row * TILE_SIZE, Ground.ID)

            self.game.place_objects(Coin.ID, 6)            

            if (self.agent.representation == Behaviors.NARROW_PUZZLE.value): 
                self.game.place_objects(Enemy.ID, action[2])            
            elif (self.agent.representation == Behaviors.WIDE_PUZZLE.value): 
                self.game.place_objects(Enemy.ID, action[5])            
            elif (self.agent.representation == Behaviors.MULTI_PIECE.value):                 
                self.game.place_objects(Enemy.ID, action[7])

            self.game.place_objects(Key.ID, 1)

            self.game.place_objects(Weapon.ID, 1)
    """    
    """
        obs = self.agent.get_current_observation({})

        if change > 0:
            self.counter_changes += change            
            self.current_stats = self.agent.get_stats()    
            self.segment += 1

        return obs
    """
    """
    def _do_step(self, action):

        self.old_stats = self.current_stats

        obs, self._reward_agent, change = self.agent.step(action)

        if (self.agent.is_done()):
            if action[1] == 1:
                self.game.place_objects(Coin.ID, 6)

            if action[2] == 1:
                self.game.place_objects(Key.ID, 1)

        obs = self.agent.get_observation()

        if change > 0:
            self.counter_changes += change            
            self.current_stats = self.agent.get_stats()    
            self.segment += 1

        return obs
    """