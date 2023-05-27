from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.zelda import ZeldaGameProblem
from pcgrl.zelda import *
from pcgrl.Utils import *
from pcgrl.callbacks import BasePCGRLCallback

from pcgrl import PCGRLPUZZLE_MAP_PATH
class ZeldaV2Env(BasePCGRLEnv):
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
                 path_models = "dungeon/",
                 env_rewards = False,
                 callback = BasePCGRLCallback()):        
        
        self.rep = rep
        self.rows = board[0] * min(piece_size, piece_size * 2) #Row               
        self.cols = board[1] * min(piece_size, piece_size * 2) #Col
        game = ZeldaGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale = 2            
        self.action_change = action_change
        self.action_rotate = action_rotate        
        self.extra_actions = {"A": 11, "B": 11, "C": 11, "D": 11, "E": 11, "F": 11, "noisy" : 10 }
        super(ZeldaV2Env, self).__init__(seed = seed, game = game, 
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
                                       path_models = path_models, callback = callback, extra_actions=self.extra_actions)
        self.current_reward   = 0
        self.counter_done     = 0        
        self.info = {}         
        self.name = "ZeldaV2Env"


    def blend_tiles(self, choices, tiles):
        """
        Given a list of states (True if ruled out, False if not) for each tile,
        and a list of tiles, return a blend of all the tiles that haven't been
        ruled out.
        """
        to_blend = [tiles[i] for i in range(len(choices)) if choices[i]]
        return to_blend    
    
    def show_state(self, potential, tiles):
        """
        Given a list of states for each tile for each position of the image, return
        an image representing the state of the global image.
        """
        rows = []
        for row in potential:
            rows.append([np.asarray(self.blend_tiles(t, tiles)) for t in row])

        #rows = np.array(rows)
        #n_rows, n_cols, tile_height, tile_width, _ = rows.shape
        #images = np.swapaxes(rows, 1, 2)
        return rows#images#Image.fromarray(images.reshape(n_rows*tile_height, n_cols*tile_width, 4))    
    
    def get_int_prob(self, prob, tiles):
        """
        Retorna um dic contendo os valores de probabilidade de cada tile que será utilizado
        para gerar o mapa gerando um id para cada para cada tile
        """
        string_to_int = dict((s, i) for i, s in enumerate(tiles))
        result = {}
        total = 0.0
        for t in tiles:
            result[string_to_int[t]] = prob[t]
            total += prob[t]
        for i in result:
            result[i] /= total
        
        return result

    def gen_random_map(self, random, _map, width, height, prob):
        map = random.choice( list(prob.keys()), size=(height, width), p=list(prob.values())).astype(np.uint8)
        return map    

    def _do_step(self, action):

        done = super()._get_done(action)
        stats = self.game.get_map_stats()
        
        regions = stats["regions_ground"]
        agent_done = self.agent_behavior.is_done()

        if regions == 1 and agent_done:

            prob = {Ground.ID : 0.80, Key.ID: action[3]/1000, Coin.ID : action[4]/1000, Enemy.ID: action[5]/1000, Weapon.ID : action[6]/1000}
            tiles = [Ground.ID, Key.ID, Coin.ID, Enemy.ID, Weapon.ID]

            d = prob #self.get_int_prob(prob, tiles)
            #print(d)

            map_locations = np.array(self.game.get_tile_positions([Ground.ID], self.game.map))
            #map = self.gen_random_map(self.np_random, map_locations, self.game.map.shape[0], self.game.map.shape[1], d)
            elements = random.choices( list(d.keys()), list(d.values()), k = len(map_locations))
            print(elements)

            map_copy = self.game.map.copy()

            for pos in range(len(map_locations)):
                y, x = map_locations[pos]
                map_copy[y, x] = elements[pos]

            self.game.update_map(map_copy)
            self.render()
            time.sleep(0.5)

            """
            #tiles = [Ground.ID]            
            
            #new_action = np.random.rand((40))*2
            #print(self.game.map)
            #time.sleep(1000)

            #ptable = np.arange(len(tiles) * noisy(action[1]+1), dtype=int)

            # shuffle our numbers in the table
            #np.random.shuffle(ptable)

            # create a 2d array and then turn it one dimensional
            # so that we can apply our dot product interpolations easily
            #ptable = np.stack([ptable, ptable]).flatten()
            #print(ptable)
            #print("Noisy")
            #s = noisy(action[1]+1)
            #potential = np.full((s, s, len(tiles)), True)
            #print(self.show_state(map_locations, tiles))

            #print(self.blend_tiles(action, tiles))
            #print("\t",noisy(action))
            
            #for row, col in map_locations:
            #    self.game.change_tile(col * TILE_SIZE, row * TILE_SIZE, Ground.ID)

            #self.game.place_objects(Coin.ID, 6)
            #if (self.agent.representation == Behaviors.NARROW_PUZZLE.value): 
            #    self.game.place_objects(Enemy.ID, action[2])            
            #elif (self.agent.representation == Behaviors.WIDE_PUZZLE.value): 
            #    self.game.place_objects(Enemy.ID, action[5])            
            #elif (self.agent.representation == Behaviors.MULTI_PIECE.value):                 
            #    self.game.place_objects(Enemy.ID, action[7])

            #self.game.place_objects(Key.ID, 1)

            #self.game.place_objects(Weapon.ID, 1)
            """ 
        return done
        
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