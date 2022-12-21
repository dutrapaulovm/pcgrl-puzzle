import distutils.version
import os
import sys
import warnings

PCGRLPUZZLE_RESOURCES_PATH = os.path.dirname(__file__) + "/resources/"
PCGRLPUZZLE_MAP_PATH = os.path.dirname(__file__) + "/maps/"

from pcgrl.version import VERSION as __version__

from . import minimap
from . import maze
from . import dungeon
from . import zelda

from pcgrl import AgentBehavior
from pcgrl import Agents
from pcgrl import Entity
from pcgrl import log

from pcgrl.BasePCGRLEnv import Experiment
from pcgrl.minimap.MiniMapGameProblem import MiniMapGameProblem
from pcgrl.minimap.MiniMapEnv import MiniMapEnv
from pcgrl.minimap.MiniMapLowMapsEnv import MiniMapLowMapsEnv

from pcgrl.maze.MazeGameProblem import MazeGameProblem
from pcgrl.maze.MazeEnv import MazeEnv

from pcgrl.dungeon.DungeonGameProblem import DungeonGameProblem
from pcgrl.dungeon.DungeonEnv import DungeonEnv

from pcgrl.mazecoin.MazeCoinGameProblem import MazeCoinGameProblem
from pcgrl.mazecoin.MazeCoinEnv import MazeCoinEnv
from pcgrl.mazecoin.MazeCoinLowMapsEnv import MazeCoinLowMapsEnv

from pcgrl.zelda.ZeldaGameProblem import ZeldaGameProblem
from pcgrl.zelda.ZeldaEnv import ZeldaEnv

from pcgrl.smb.SMBGameProblem import SMBGameProblem
from pcgrl.smb.SMBEnv import SMBEnv

from pcgrl.Agents import *
from pcgrl.GameProblem import GameProblem
from pcgrl import Sprite
from pcgrl import SpriteSheet
from pcgrl import Utils
from pcgrl import Generator
from pcgrl import PCGRLEnv
from pcgrl import wrappers
from pcgrl import Grid

from pcgrl.wrappers import RGBToGrayScaleObservationWrapper
from pcgrl.wrappers import MapWrapper
from pcgrl.wrappers import SegmentWrapper
from pcgrl.wrappers import OneHotEncodingWrapper
from pcgrl.wrappers import ActionRepeatWrapper
from pcgrl.wrappers import EnvInfo
from pcgrl.wrappers import ExperimentMonitor
from pcgrl.wrappers import RenderMonitor
from pcgrl.wrappers import make_env

from gym.envs.registration import register


register(
    id='mazecoin-narrow-puzzle-2x3-v0',
    entry_point='pcgrl.mazecoin.MazeCoinLowMapsEnv:MazeCoinLowMapsEnv',
    kwargs={"seed" : 42,    
            "rep" : Behaviors.NARROW_PUZZLE.value,
            "path" : None,
            "save_logger" : False,
            "save_image_level" : False,
            "show_logger" : False,
            "rendered" : True,
            "action_change" : False,
            "action_rotate" : False,
            "agent" : Experiment.AGENT_HEQHP.value,
            "reward_change_penalty" : -1,            
            "board" : (3, 2),
            "piece_size" : 8,
            "path_models" : "mazecoin-lowmodels/"}
)

register(
    id='mazecoin-narrow-puzzle-2x3-v1',
    entry_point='pcgrl.mazecoin.MazeCoinLowMapsEnv:MazeCoinLowMapsEnv',
    kwargs={"seed" : 42,
            "rep" : Behaviors.NARROW_PUZZLE.value,
            "path" : None,
            "save_logger" : False,
            "save_image_level" : False,
            "show_logger" : False,
            "rendered" : True,
            "action_change" : True,
            "action_rotate" : True,
            "agent" : Experiment.AGENT_HEQHP.value,
            "reward_change_penalty" : -1,            
            "board" : (3, 2),
            "piece_size" : 8,
            "path_models" : "mazecoin-lowmodels/"}
)

register(
    id='dungeon-narrow-puzzle-2x3-v0',
    entry_point='pcgrl.dungeon.DungeonEnv:DungeonEnv',
        kwargs={"seed" : 42,
            "rep" : Behaviors.NARROW_PUZZLE.value,
            "path" : None,
            "save_logger" : True,
            "save_image_level" : False,
            "show_logger" : False,
            "rendered" : True,
            "action_change" : False,
            "action_rotate" : False,
            "agent" : Experiment.AGENT_HEQHP.value,
            "reward_change_penalty" : -1,            
            "board" : (3, 2),
            "piece_size" : 8}
)

register(
    id='zelda-narrow-puzzle-2x3-v0',
    entry_point='pcgrl.zelda.ZeldaEnv:ZeldaEnv',
    kwargs={"seed" : 42,
            "rep" : Behaviors.NARROW_PUZZLE.value,
            "path" : None,
            "save_logger" : True,
            "save_image_level" : False,
            "show_logger" : False,
            "rendered" : True,
            "action_change" : False,
            "action_rotate" : False,
            "agent" : Experiment.AGENT_HEQHP.value,
            "reward_change_penalty" : -1,            
            "board" : (3, 2),
            "piece_size" : 8}
)

class Game(Enum):                 
    MAZE                           = "Maze"
    MAZECOIN                       = "MazeCoin"                  
    MAZECOINLOWMAPS                = "MazeCoinLowMaps"     
    COMBAT                         = "Combat"        
    DUNGEON                        = "Dungeon"
    ZELDA                          = "Zelda"
    SMB                            = "SMB"    
    ZELDAV3                        = "ZeldaV3"
    MINIMAP                        = "MiniMap"
    MINIMAPLOWMODELS               = "MiniMapLowModels"              
    def __str__(self):
        return self.value
            
from pcgrl.utils.experiment import ExperimentManager