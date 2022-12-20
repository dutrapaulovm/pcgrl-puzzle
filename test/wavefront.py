from pcgrl import Game
from pcgrl.Utils import *
from pcgrl.BasePCGRLEnv import Experiment

import matplotlib.pyplot as plt
import csv

#Usar para o Zelda
def build_mapZelda(map, entropies, index, save_map = True):
    
    _map = np.array(map)
    _map = list(_map.flatten())    
    _map = collections.Counter(_map)        
    _sum = sum(_map[e] for e in _map.keys())
    _dist = {}
    
    for key, value in _map.items():    
        _dist[key] = value / _sum

    #print(_dist)
    t = 1
    map_linearity = 0
    liniency = 0
    
    locations = array_el_locations(map)
    segments  = np.random.randint(40, size=6)

    segments = calc_prob_dist(segments)
    segments_linearity = 0
    for k, v in segments.items():        
        segments_linearity += 1
    
    segments_linearity /= 6

    for key, value in _map.items():                     
        v =  entropies[index]
        """
        if (key == 0): #Ground
            if _dist[key] < 0.50:
                map_linearity += (0.5 * value*v)
            else:
                map_linearity += (-0.5 * (value*v))                
        
        elif (key == 1): #Blocks
            if _dist[key] < 0.60:
                map_linearity += (1 * value*v)
            else:
                map_linearity += (-1 * (value*v))                
        
        if (key == 2): #Coins
            map_linearity += (2 * value*v)
            liniency += (2* value*v)
        elif key in [2, 3, 5, 6]:
            map_linearity += 1*v
            liniency += value   
        """
    pos_player = locations[3]            
    pos_player = pos_player[0]    
    row, col = pos_player[0], pos_player[1]         

    replaces = (2, 3)
    map[map == 2] = 0    
    map[map == 3] = 0    
    h = 0 
    map, max_dist, ent, grounds = wave_front_entrace(map, row, col, h)
    if save_map:
        fig, ax = plt.subplots(figsize=(10,10))
    
    map[row][col] = 50
    coins_linearity = 0
    for p in locations[2]:
        row = p[0]
        col = p[1]
        dist = map[row][col]
        coins_linearity += dist
        map[row][col] = 100

    coins_linearity /= max_dist

    if coins_linearity >= 0.65:
        p_coins_linearity = 2
    else:
        p_coins_linearity = -2
    
    map_linearity = round((((p_coins_linearity + map_linearity) / 25) / 10), 2)

    coins_linearity1 = round(1 - der_htan(coins_linearity),2) #min(1, coins_linearity)
    coins_linearity2 = round(sigmoid(coins_linearity), 2) #min(1, coins_linearity)
    coins_linearity3 = round(htan(coins_linearity), 2) #min(1, coins_linearity)
    tiles = 24 * 16
    if save_map:
        # Loop over data dimensions and create text annotations.
        for _i in range(len(map)):
            for _j in range(len(map[0])):
                text = ax.text(_j, _i, map[_i, _j],
                        ha="center", va="center", color="w")
    
    linearity = htan(map_linearity + entropies[index])
    liniency  = round( sigmoid(liniency), 2) 
    entropy_linearity = sigmoid(entropies[index])
    
    tiles_line = grounds / tiles
    
    liniency = round(tiles_line, 2) 
    entrance_linearity = round((ent / grounds), 2)
    coins_linearity = round(coins_linearity2, 2)
    """
    title ="Linearity: {}, Map Linearity: {}, Leniency: {}, Max Dist: {}\nCoins Linearity 1(der_htan): {} \
              \nCoins Linearity 2(sigmoid): {} \
              \nCoins Linearity 3(htan): {} \
              \nCoins Linearity D: {} \
              \nEntradas: {} \
              \nGrounds: {} \
              \nLinearity Entradas: {} \
              \nEntropy{}, {} \
               ".format(linearity, map_linearity , liniency, max_dist, coins_linearity1, coins_linearity2,
                        coins_linearity3, coins_linearity, ent, grounds, (ent / grounds), entropies[index], entropy_linearity)
    """
    title =" Leniency: {}, Max Dist: {} \
              \nCoins Linearity (sigmoid): {} \
              \nEntradas: {} \
              \nGrounds: {} \
              \nEntrance Linearity: {} \
              \nEntropy{}, Entropy Linearity {} \
               ".format(liniency, max_dist, 
                        coins_linearity, 
                        ent, 
                        grounds, 
                        entrance_linearity, 
                        entropies[index], entropy_linearity)                        

    print (title)
    if save_map:
        plt.tight_layout() 
        plt.title(title)
        plt.imshow(map)              
        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("Map"+str(index), ".png") )
        plt.savefig(save_file)
        plt.close()   

    
    return liniency , entrance_linearity, coins_linearity

def get_tile_positions(tiles, map):        
    max_row = map.shape[0]
    max_col = map.shape[1]
    new_map = []
    for row in range(max_row):
        for col in range(max_col):
            id = int(map[row][col])
            if id in tiles:
                new_map.append((row, col))
    return new_map

def build_map(map, entropies, index, segments, save_map = True):
    
    _map = np.array(map)
    _map = list(_map.flatten())    
    _map = collections.Counter(_map)        
    _sum = sum(_map[e] for e in _map.keys())
    _dist = {}
    
    for key, value in _map.items():    
        _dist[key] = value / _sum

    #print(_dist)
    t = 1
    map_linearity = 0
    liniency = 0
    
    locations = array_el_locations(map)    
    level_segments = segments[index]
    level_segments = level_segments.replace("'", "").replace("[", "").replace("]", "")
    
    level_segments = level_segments.split()        
    level_segments = np.array(level_segments).astype(int)      
    n_segments = len(level_segments)
    level_segments = calc_prob_dist(level_segments)
    #print(level_segments)
    #time.sleep(5)
    segments_linearity = 0
    for k, v in level_segments.items():        
        segments_linearity += 1

    #segments_linearity /= n_segments

    pos_player = locations[3]            
    pos_player = pos_player[0]    
    row, col   = pos_player[0], pos_player[1]         
    
    map[map == 2] = 0
    map[map == 3] = 0
    h = 0 
    map, max_dist, ent, grounds = wave_front_entrace(map, row, col, h)
    if save_map:
        fig, ax = plt.subplots(figsize=(10,10))

    map[row][col] = 50
    coins_linearity = 0
    n_coins = len(locations)

    for p in locations[2]:
        row = p[0]
        col = p[1]
        dist = map[row][col]
        coins_linearity += dist
        map[row][col] = 100
    
    x = (coins_linearity / max_dist) * (n_coins**2)
    coins_linearity = x
    
    tiles = 24 * 16
    """
    if save_map:
        # Loop over data dimensions and create text annotations.
        for _i in range(len(map)):
            for _j in range(len(map[0])):
                text = ax.text(_j, _i, map[_i, _j],
                        ha="center", va="center", color="w")
    """    
    tiles_line = segments_linearity * (grounds / tiles)
    entropy_value = 1 #entropies[index]
    liniency            = round(tiles_line, 2)  #round( sigmoid( tiles_line * entropy_value ), 2) 
    entrance_linearity  = round((ent / grounds) , 2) #round( sigmoid( (ent / grounds) * entropy_value), 2)
    #coins_linearity     = round( sigmoid(coins_linearity), 2)#round( sigmoid( coins_linearity * entropy_value), 2)
    coins_linearity     = round(coins_linearity, 2)#round( sigmoid( coins_linearity * entropy_value), 2)
    entropy_linearity   = round(entropies[index], 2) #round( sigmoid( entropies[index]), 2)
    segments_linearity  = round(segments_linearity, 2)
    """
    title =" Leniency: {}, Max Dist: {} \
              \nCoins Linearity: {} \
              \nEntradas: {} \
              \nGrounds: {} \
              \nEntrance Linearity: {} \
              \nEntropy{}, Entropy Linearity {} \
              \nSegments linearity {}, {} \
               ".format(liniency, max_dist, 
                        coins_linearity, 
                        ent,
                        grounds, 
                        entrance_linearity, 
                        entropies[index], entropy_linearity, segments_linearity, segments[index])                        
    
    print (title)
    if save_map:
        plt.tight_layout() 
        plt.title(title)
        plt.imshow(map)              
        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("Map"+str(index), ".png") )
        plt.savefig(save_file)
        plt.close()
    """
    return liniency , entrance_linearity, coins_linearity, entropy_linearity, segments_linearity, max_dist, map

map = []
dleniency = []
dentrance_linearity = []

envs = [Game.DUNGEON.value, Game.MAZECOINLOWMAPS.value, Game.ZELDA.value]
agents = [Experiment.AGENT_SS.value, Experiment.AGENT_HHP.value, Experiment.AGENT_HEQHP.value]

for env_name in envs:
    for agent in agents:
        
        path_envs = "map/"+f"{env_name}/{agent}"

        path = f"{os.path.dirname(__file__)}/{path_envs}/" 
        info_file = os.path.dirname(__file__) + "/{}/{}".format(path_envs, "info.csv")

        entropies = []
        df = pd.read_csv(info_file)
        data = df["entropy"] 
        entropies = np.array(data).astype("float")  
        data = df["segments"] 
        segments = np.array(data)
        i = 0
        c_map = 0
        max_maps = 1
        save_map = False

        index = 0
        dleniency = []
        dentrance_linearity = []
        dobjective_leniency = []
        dentropies_linearity = []
        dsegments_repeated = []
        maps = []

        for file in os.listdir(path):    
            map = []
            if file[-3:] in {'csv'}:
                pathfile = os.path.join(path, file)          
                with open(pathfile, newline='') as csvfile:
                    data = list(csv.reader(csvfile))            
                    
                map = np.array(data).astype("int")            
                
            if (len(map) > 0):
                #print(map)
                #print(index)
                locations = array_el_locations(map)
                
                leniency, entrance_linearity, objective_linearity, entropy_linearity, segments_repeated, max_dist, map = build_map(map, entropies, index, segments, save_map=False)
                maps.append(map)
                # Leniency mede a proporção de regiões que representam o chão por onde o jogador pode passar
                # Leniency (Mede o quanto os níveis são faceis) Quanto mais áreas com chão, mas fácil é o nível

                #Linearidade das entradas define o número de entradas que o labirinto possui. Levando em consideração também
                # um caminho onde há paredes em ambos os lados.

                #Linearidade das moedas mede o quão distante esta as moedas em relação do jogador

                #Linearidade da entropia mede o quanto o nível é linear em relação ao valor da entropia do cenário.

                #v_entropy = entropies[index]        
                #entropy_linearity = sigmoid(v_entropy[0])                

                dleniency.append(round(leniency, 2))
                dentrance_linearity.append(round(entrance_linearity, 2))       
                dobjective_leniency.append(round(objective_linearity, 2))
                dentropies_linearity.append(round(entropy_linearity, 2))
                dsegments_repeated.append(round(segments_repeated, 2))
                index += 1
                
                pos_player = locations[3]    
                pos_player = pos_player[0]
                
                start = pos_player[0], pos_player[1]   
                """
                for p in locations[2]:
                    row = p[0]
                    col = p[1]        
                    destination = row, col
                
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(map, interpolation='nearest')
                    plt.xticks([]), plt.yticks([])

                    graph = mat2graph(map)
                    #print(graph)
                    shortest_route = A_star(graph, start, destination, map)
                    print(shortest_route)

                    #plt.subplot(1, 2, 2)
                    plt.imshow(map)
                    plt.show()    
                """
        """
        fig, ax = plt.subplots()
        plt.title("Linearity")
        plt.scatter(dpath_linearity, dentrance_linearity, c='red')
        plt.xlabel('Map Linearity')
        plt.ylabel('Coins Linearity')
        plt.title('Scatter plot')
        plt.tight_layout()
        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("Map-Scatter", ".png") )
        plt.savefig(save_file)
        plt.close()            
        """

        def normalize(x):
            return (x-min(x))/(max(x)-min(x))

        #Normalized Data
        dobjective_leniency = normalize(dobjective_leniency)
        dentrance_linearity = normalize(np.array(dentrance_linearity))       
        dentropies_linearity = normalize(np.array(dentropies_linearity))
        dsegments_repeated = normalize(np.array(dsegments_repeated))
        dleniency = np.array(dleniency)
        dleniency = normalize(dleniency)
        save_map = False
        for m in range(len(maps)):    

            if save_map:
                fig, ax = plt.subplots(figsize=(10,10))
                # Loop over data dimensions and create text annotations.
                for _i in range(len(maps[m])):
                    for _j in range(len(maps[m][0])):
                        text = ax.text(_j, _i, maps[m][_i, _j],
                                ha="center", va="center", color="w")        
            title = "Linearity: {} \
                    Leniency: {}".format(round(dentropies_linearity[m],2), round(dobjective_leniency[m],2))                            
            print (title)
            if save_map:
                plt.tight_layout() 
                plt.title(title)
                plt.imshow(maps[m])              
                save_file = os.path.join(path, "mapcolors")      
                save_file = os.path.join(save_file, "{}{}".format("Map"+str(m), ".png") )
                plt.savefig(save_file)
                plt.close()

        dentrance_linearity = np.array(dentrance_linearity)
        xlim = dleniency.min(), dleniency.max()
        ylim = dentrance_linearity.min(), dentrance_linearity.max()

        _map = np.array(dleniency)
        _map = list(_map.flatten())    
        _map = np.sort(_map)
        _map = collections.Counter(_map)    
        _sum = sum(_map[e] for e in _map.keys())
        print(_map)

        cmap     = 'Greens_r' #'Spectral'
        mincnt   = 0
        gridsize = 10
        """
        fig, ax0 = plt.subplots(sharey=True)

        hb = ax0.hexbin(dleniency, dentrance_linearity, gridsize=gridsize,  cmap=cmap, mincnt=mincnt)
        ax0.set(xlim=xlim, ylim=ylim)
        #ax0.set_title("Hexagon binning")
        cb = fig.colorbar(hb, ax=ax0, label='Número de Níveis')

        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("PathXEntrance-Linearity", ".png") )
        plt.xlabel('Leniency')
        plt.ylabel('Entrance Linearity')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
        plt.close()

        dleniency = np.array(dleniency)

        dobjective_leniency = np.array(dobjective_leniency)
        xlim = dleniency.min(), dleniency.max()
        ylim = dobjective_leniency.min(), dobjective_leniency.max()

        fig, ax0 = plt.subplots(sharey=True)
        hb = ax0.hexbin(dleniency, dobjective_leniency, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        ax0.set(xlim=xlim, ylim=ylim)
        #ax0.set_title("Hexagon binning")
        cb = fig.colorbar(hb, ax=ax0, label='Número de Níveis')

        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("PathXObjective-Linearity", ".png") )
        plt.xlabel('Leniency')
        plt.ylabel('Objective Linearity')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
        plt.close()
        """

        dentropies_linearity = np.array(dentropies_linearity)

        xlim = dobjective_leniency.min(), dobjective_leniency.max()
        ylim = dentropies_linearity.min(), dentropies_linearity.max()

        fig, ax0 = plt.subplots(sharey=True)
        hb = ax0.hexbin(dobjective_leniency, dentropies_linearity, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        ax0.set(xlim=xlim, ylim=ylim)
        #ax0.set_title("Hexagon binning")
        cb = fig.colorbar(hb, ax=ax0, label='Número de Níveis')

        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("PathXEntropy-Linearity", ".png") )
        plt.xlabel('Leniency')
        plt.ylabel('Linearity')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
        plt.close()
        """
        dleniency = np.array(dleniency)
        dsegments_repeated = np.array(dsegments_repeated)

        xlim = dleniency.min(), dleniency.max()
        ylim = dsegments_repeated.min(), dsegments_repeated.max()

        fig, ax0 = plt.subplots(sharey=True)
        hb = ax0.hexbin(dleniency, dsegments_repeated, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        ax0.set(xlim=xlim, ylim=ylim)
        #ax0.set_title("Hexagon binning")
        cb = fig.colorbar(hb, ax=ax0, label='Número de Níveis')

        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("PathXSegments-Linearity", ".png") )
        plt.xlabel('Leniency')
        plt.ylabel('Segments Linearity')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
        plt.close()

        dobjective_leniency = np.array(dobjective_leniency)
        dsegments_repeated = np.array(dsegments_repeated)

        xlim = dsegments_repeated.min(), dsegments_repeated.max()
        ylim = dobjective_leniency.min(), dobjective_leniency.max()


        fig, ax0 = plt.subplots(sharey=True)
        hb = ax0.hexbin(dsegments_repeated, dobjective_leniency, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        ax0.set(xlim=xlim, ylim=ylim)
        #ax0.set_title("Hexagon binning")
        cb = fig.colorbar(hb, ax=ax0, label='Número de Níveis')

        save_file = os.path.join(path, "mapcolors")      
        save_file = os.path.join(save_file, "{}{}".format("CoinsXSegments-Linearity", ".png") )
        plt.xlabel('Segments Repetead')
        plt.ylabel('Coins Linearity')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
        plt.close()
        """

        """
        if 0 not in _map:        
            map_linearity += -100
        if 1 not in _map:        
            map_linearity += -100
        if 2 not in _map: 
            map_linearity += -50
        if 3 not in _map: 
            map_linearity += -50
        if 4 not in _map: 
            map_linearity += -50
        if 5 not in _map: 
            map_linearity += -50
        if 6 not in _map: 
            map_linearity += -50
        if 7 not in _map: 
            map_linearity += -50
        if 8 not in _map: 
            map_linearity += -50
        _entropy = 2.75
        map_linearity = (map_linearity / 100)
        linearity = sigmoid(map_linearity + _entropy) 
        liniency  = sigmoid(liniency + _entropy) 
        """
        #print ("Linearity: {}, Map Linearity: {}, Leniency: {}".format(linearity, map_linearity , liniency))
        #print(map)
        #x = [linearity+0.5, linearity-0.5, linearity+0.6, linearity, linearity]
        #y = [2.75, 2.50, 2.75, 2.50, 2.2]
        """
        fig, ax = plt.subplots()
        plt.subplot(121)
        plt.scatter(y, x, c='red')
        plt.xlabel('Linearity')
        plt.ylabel('Entropy')
        plt.title('Scatter plot')
        plt.show()
        """

        """
        row = 0
        col = 0
        grid = map
        passable_values = [1, 2, 3, 4, 5, 6, 7, 8]
        n = len(grid)
        m = len(grid[0])
        old_value = grid[row][col]
        queue = Queue()
        dist = 0
        queue.put((row, col))
        grid[grid == 0] = 0
        grid[grid == 1] =  0x40000
        while not queue.empty():    
            row, col = queue.get()      
            nei = neighbors(row, col, n, m)                        
            for r, c in nei:                   
                if grid[r][c] != 0:                
                    continue
                else:
                    grid[r][c]  = grid[row][col] + 1#dist                 
                    queue.put((r , c))
        x1, y1 = 5, 5
        x2, y2 = 0, 0

        d = abs( grid[y1][x1] - grid[y2][x2] ) + abs( grid[y1][x2] - grid[y1][x2] )
        #print(d)
        """
        #fig, ax = plt.subplots()
        #plt.imshow(grid)
        #ax.grid(True)
        #plt.tight_layout()            
        #plt.show()
        #plt.close() 

        #print(grid)        
        #flood_fill_v2(map, color_grid, 0, 0, color_index + 1)
