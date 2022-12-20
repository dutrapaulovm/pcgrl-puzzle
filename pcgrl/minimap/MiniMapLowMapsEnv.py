from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.minimap import MiniMapGameProblem
from pcgrl.minimap.MiniMapLevelObjects import *
from pcgrl.minimap import *
from pcgrl.Utils import *
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.callbacks import BasePCGRLCallback

class MiniMapLowMapsEnv(BasePCGRLEnv):
    
    def __init__(self, 
                seed = None, 
                show_logger      = False, 
                save_logger      = False, 
                save_image_level = False, 
                rendered         = False,
                path             = "", 
                rep              = None, 
                action_change    = False, 
                action_rotate    = False, 
                agent = None,
                reward_change_penalty = None,
                piece_size = 8, 
                board = (3,2), 
                env_rewards = False,
                path_models = "minimap-lowmodels/",
                callback = BasePCGRLCallback()):        
        
        self.rep = rep          
        self.cols = board[0] * min(piece_size, piece_size * 2) #Col
        self.rows = board[1] * min(piece_size, piece_size * 2) #Row               
        game = MiniMapGameProblem(cols = self.cols, rows = self.rows)
        game.scale    = 2            
        self.action_change = action_change
        self.action_rotate = action_rotate
        super(MiniMapLowMapsEnv, self).__init__(seed = seed, game = game, 
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
        self.name = "MinimapLowMapsEnv" 
    
    """
    def _do_step(self, action):
        
        obs = super()._do_step(action)

        obs = self.agent.get_current_observation({})

        if (self.agent.is_done()):

            tiles = [Grass.ID]
            map_locations = self.game.get_tile_positions(tiles, self.game.map)

            minperson = self.game._range_person[0]
            maxperson = self.game._range_person[1]

            n = self.game.np_random.randint(minperson, maxperson)
            print(minperson)
            print(maxperson)
            print(n)
            time.sleep(10)

            #for row, col in n:
            #    self.game.change_tile(col * TILE_SIZE, row * TILE_SIZE, Person.ID)            

        return obs      
    """    