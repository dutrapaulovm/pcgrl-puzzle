from os import mkdir
import time     
import random
from pygame import color
import pylab
from gym.utils import seeding
from gym import spaces
from gym import wrappers
from pcgrl import Game
from pcgrl.maze.MazeGameProblem import MazeGameProblem
from pcgrl.mazecoin.MazeCoinGameProblem import MazeCoinGameProblem
from pcgrl.zelda.ZeldaGameProblem import ZeldaGameProblem
from pcgrl.dungeon.DungeonGameProblem import DungeonGameProblem
from pcgrl.smb.SMBGameProblem import SMBGameProblem

from pcgrl.wrappers import *
from pcgrl import Generator
from pcgrl.utils.utils import *

from pcgrl.minimap.MiniMapGameProblem import *
from pcgrl.minimap.MiniMapLevelObjects import *
from numpy.core.fromnumeric import choose
from pygame.cursors import thickarrow_strings
from pygame.image import tostring
from model import *
SAVE_PATH = os.path.dirname(__file__) + "/levels/"           

def save_levels(map, name_game):
    path = os.path.abspath(os.path.join("pcgrl-puzzle", os.pardir))
    path = os.path.join(path, "pcgrl/maps/{}".format(map))

    save_file = mk_dir(SAVE_PATH, "segments")      
    path_save_file = mk_dir(save_file, name_game)
    
    game_piece = None
    if name_game == Game.ZELDA.value:
        game_piece = ZeldaGameProblem(cols = 8, rows = 8, border = False, tile_size=16)    
    elif name_game == Game.MAZECOINLOWMAPS.value:
        game_piece = MazeCoinGameProblem(cols = 8, rows = 8, border = False, tile_size=16)  
    elif name_game == Game.DUNGEON.value:
        game_piece = DungeonGameProblem(cols = 8, rows = 8, border = False, tile_size=16)  
    elif name_game == Game.SMB.value:
        game_piece = SMBGameProblem(cols = 8, rows = 8, border = False)            
    elif name_game == Game.MINIMAPLOWMODELS.value:
        game_piece = MiniMapGameProblem(cols = 8, rows = 8, tile_size=16)

    generator =  Generator(path=path, piece_size=(8,8)) #, dim=dim, n_models = n_models)   
    for p in range(generator.count()):        
        game_piece.blocked = False
        game_piece.clear_layers()
        piece = generator.get_piece(p)
        piece = np.array(piece).astype("int") 
        print(piece)
        game_piece.update_map(piece)
        game_piece.render()        
        save_file = os.path.join(path_save_file, "pieces-{}-{}{}".format(name_game, p, ".png") )
        
        print("Salvar em: ", save_file)        
        game_piece.save_screen(save_file)


    #path = os.path.join(path, "test/map/best/{}".format(map))
    if name_game == Game.ZELDA.value:
        game = ZeldaGameProblem(cols = 24, rows = 16, border = False, tile_size=16)    
    elif name_game == Game.SMB.value:
        game = SMBGameProblem(cols = 48, rows = 8, border = False)            
    elif name_game == Game.MAZECOINLOWMAPS.value:
        game = MazeCoinGameProblem(cols = 32, rows = 16, border = False, tile_size=16)            
    elif name_game == Game.DUNGEON.value:
        game = DungeonGameProblem(cols = 32, rows = 16, border = False, tile_size=16)    
    elif name_game == Game.MINIMAPLOWMODELS.value:
        game = MiniMapGameProblem(cols = 24, rows = 16, tile_size=16)
    elif name_game == Game.MAZECOIN.value:
        game = MazeCoinGameProblem(cols = 32, rows = 16, border = False, tile_size=16)
    
    print("Caminho principal", SAVE_PATH)
    for file in os.listdir(path):
        if file[-3:] in {'csv'}:
            pathfile = os.path.join(path, file)    
            print("Caminho: ", pathfile)            
            game.clear_layers()
            game.load_map(pathfile)                    
            game.render()
            f = file
            save_file = os.path.join(SAVE_PATH, "{}-{}".format(name_game, f.replace(".csv", ".png")) )
            print("Salvar em: ", save_file)
            game.save_screen(save_file)        
        
if __name__ == '__main__':
    #games = [Game.SMB.value, Game.ZELDA.value, Game.MAZECOINLOWMAPS.value,  Game.DUNGEON.value]    
    #maps = ["smb", "zelda", "mazecoin-lowmodels", "dungeon"]        
    games = [Game.SMB.value, Game.ZELDA.value, Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.MINIMAPLOWMODELS.value]    
    maps = ["smb", "zelda", "dungeon", "mazecoin-lowmodels", "minimap-lowmodels"]            
    for i in range(len(games)):
        save_levels(name_game=games[i], map=maps[i])