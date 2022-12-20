# -*- coding: utf-8 -*-
from distutils.log import error
import collections, numpy
from collections import defaultdict
import math
import numpy as np
import time     
import random
from timeit import default_timer as timer
from datetime import timedelta
import heapq

# BFS approach:
from queue import Queue

import os




def normalize(x):
    
    mi = min(x)    
    mx = max(x)

    #if (mi == mx):
    #    return 1 / (x / len(x))
    #return x / np.sqrt(np.sum(x**2))
    return (x-np.min(x)) / (np.max(x)-np.min(x))
    #return 2*(x-min(x))/(max(x)-min(x))-1

def clamp(num, min_value, max_value):
    num = max(min(num, max_value), min_value)
    return num

def noisy(val):
                    
    between_val = val / 2.0
    if (val < between_val):
        return int(round(np.random.uniform(val, between_val)))
    else:
        return int(round(np.random.uniform(between_val, val)))            

def sigmoid(x, a = 0.5):
    return a / (a + math.exp(-x))

# Hyperbolic Tangent (htan) Activation Function
def htan(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# htan derivative
def der_htan(x):
  return 1 - htan(x) * htan(x)
                    
def to_power(x, n):
    res = 1
    for i in range(n):
        res *= x

    return res;    

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):       
        if i == 0:
            continue
        cum_sum[i] =  cum_sum[i] / (i + 1)
    return cum_sum

def clear_console():
    if 'google.colab' in str(get_ipython()):
        clear_output()
    else:
        command = 'clear'
        if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
            command = 'cls'
        os.system(command)

def pdf(x, epsilon = 0.001):
    x = x + epsilon
    return x / np.sum(x)

def ecdf(data):    
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return (x , y)

#https://machinelearningmastery.com/divergence-between-probability-distributions/        
# calculate the kl divergence
def kl_divergence(p, q): 
    #p = np.array(p)
    #p = p.flatten()      
    
    #q = np.array(q)
    #q = q.flatten()  

    #p = p / np.sum(p)
    #q = q / np.sum(q)
    #print(p)
    #print(q)
    #time.sleep(1)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    #return np.sum( np.where(p != 0, p[i] * np.log(p[i] / q[i]) for i in range(len(p)), 0) )   
 
# calculate the js divergence
def js_divergence(p, q, w = 0.5, epsilon: float = 1e-8):
        
    p = np.array(p)
    p = p.flatten()      
    
    q = np.array(q)
    q = q.flatten()
        
    p = p+epsilon
    q = q+epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)    
                
    m = w * (p + q)
            
    return (w * kl_divergence(p, q) + (1 - w) * kl_divergence(q, m))
    #return w * kl_divergence(p, m) + w * kl_divergence(q, p)

def decoded(state, max_cols):	
    
    col = state % max_cols
    a   = state // max_cols
    row = a % max_cols

    pos = (row, col)	
	
    return pos

def encoded(row, col, maxcols):
	return ( (maxcols * row) + col)

def encodedXY(x, y, maxcols, room_width, state_w, state_h):

	row   = 0
	col   = 0
	state = 0
	
	col = ((x) % room_width) / state_w
	
	row = ((y) % room_width) / state_h
		
	state = encoded( row, col, maxcols)
	
	return int(state)
	
def decodedXY(state, screen_width, state_width, state_height):
	
    max_cols = int(screen_width / state_width)

    col = state % max_cols
    a   = state // max_cols
    row = a % max_cols

    vec = pygame.Vector2()

    vec.y = int(row * state_height)
    vec.x = int(col * state_width)

    return vec

def neighbors(row, col, n, m, four_way=False):
    """Return indices of adjacent cells"""
    if four_way:
        return [
            (max(0, row - 1), col), (row, min(m, col + 1)),
            (min(n, row + 1), col), (row, max(0, col - 1))]
    else:        
        return [
            (max(0, row - 1), col), (max(0, row - 1), min(m, col + 1)),
            (row, min(m, col + 1)), (min(n, row + 1), min(m, col + 1)),
            (min(n, row + 1), col), (min(n, row + 1), max(0, col - 1)),
            (row, max(0, col - 1)), (max(0, row - 1), max(0, col - 1))]
        
def manhattan_distance(xy1, xy2):
  "Returns the Manhattan distance between points xy1 and xy2"
  return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

def heauristic(cell, goal):
    x1, y1 = cell
    x2, y2 = goal

    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist

def flip_coin( p ):
  r = random.random()
  return r < p

# DFS approach:
def dfs(grid, i, j, old_color, new_color):
    n = len(grid)
    m = len(grid[0])
    if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] != old_color:
        return
    else:
        grid[i][j] = new_color
        dfs(grid, i+1, j, old_color, new_color)
        dfs(grid, i-1, j, old_color, new_color)
        dfs(grid, i, j+1, old_color, new_color)
        dfs(grid, i, j-1, old_color, new_color)

def flood_fill(grid, i, j, new_color):
    old_color = grid[i][j]
    if old_color == new_color:
        return
    dfs(grid, i, j, old_color, new_color)     
 
def set_border_values(image, value):
    """Set edge values along all axes to a constant value.

    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.

    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_border_values(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    """
    for axis in range(image.ndim):
        # Index first and last element in each dimension
        sl = (slice(None),) * axis + ((0, -1),) + (...,)
        image[sl] = value
        
def fast_pad(image, value, *, order="C"):
    """Pad an array on all axes by one with a value.

    Parameters
    ----------
    image : ndarray
        Image to pad.
    value : scalar
         The value to use. Should be compatible with `image`'s dtype.
    order : "C" or "F"
        Specify the memory layout of the padded image (C or Fortran style).

    Returns
    -------
    padded_image : ndarray
        The new image.

    Notes
    -----
    The output of this function is equivalent to::

        np.pad(image, 1, mode="constant", constant_values=value)

    Up to versions < 1.17 `numpy.pad` uses concatenation to create padded
    arrays while this method needs to only allocate and copy once.
    This can result in significant speed gains if `image` has a large number of
    dimensions.
    Thus this function may be safely removed once that version is the minimum
    required by scikit-image.

    Examples
    --------
    >>> _fast_pad(np.zeros((2, 3), dtype=int), 4)
    array([[4, 4, 4, 4, 4],
           [4, 0, 0, 0, 4],
           [4, 0, 0, 0, 4],
           [4, 4, 4, 4, 4]])
    """
    # Allocate padded image
    new_shape = np.array(image.shape) + 2
    new_image = np.empty(new_shape, dtype=image.dtype, order=order)

    # Copy old image into new space
    sl = (slice(1, -1),) * image.ndim
    new_image[sl] = image
    # and set the edge values
    set_border_values(new_image, value)

    return new_image

def flood_fill_v4(grid, row, col, passable):
    num_tiles = 0
    n = len(grid)-1
    m = len(grid[0])-1
    if grid[row][col] not in passable:
        return 0
    queue = Queue()
    queue.put((row, col))
    while not queue.empty():
        row, col = queue.get()                        
        if grid[row][col] not in passable:
            continue
        else:            
            num_tiles += 1                                    
            grid[row][col] = num_tiles
            nei = neighbors(row, col, n, m, four_way=True)
            for r, c in nei:                   
                queue.put((r , c))        
    
    return num_tiles

def find_solution(grid, start, destination, passable):
    num_tiles = 0
    n = len(grid)-1
    m = len(grid[0])-1

    row = start[0]
    col = start[1]

    if grid[row][col] not in passable:
        return 2
    queue = Queue()   


    queue.put((row, col))

    row_dest = destination[0]
    col_dest = destination[1]

    while not queue.empty():                        

        row, col = queue.get()                                               
        
        if (row == row_dest and col == col_dest):
            return 1 

        if grid[row][col] not in passable:
            continue
        else:            
            num_tiles += 1                                    
            grid[row][col] = num_tiles
            nei = neighbors(row, col, n, m, four_way=True)
            for r, c in nei:                   
                queue.put((r , c))
    
    return 0


def flood_fill_v5(grid, row, col, passable, images):
    num_tiles = 0
    n = len(grid)-1
    m = len(grid[0])-1
    if grid[row][col] != passable:
        return 0
    queue = Queue()
    queue.put((row, col))

    while not queue.empty():            
        images.append(grid)      
        row, col = queue.get()                        
        if grid[row][col] != passable:
            continue
        else:            
            num_tiles += 1                                    
            grid[row][col] = num_tiles
            nei = neighbors(row, col, n, m, four_way=True)
            for r, c in nei:                   
                queue.put((r , c))        
    
    return num_tiles

def calc_entropy(n_repetead = 0, size = (3, 2)):
    
    obs = np.arange(size[0] * size[1]).reshape(size[1], size[0])            
    obs = np.array(obs).flatten()
    
    if n_repetead > 0:
        n_count = 1        
        for i in range(len(obs)):                        
            obs[i] = 0
            if n_count == n_repetead:
                break
            n_count += 1

    return round(entropy(obs), 2)

def entropy(p, b = 2, w = 1):
    """Calculate de entropy of an distribution

    Args:
        p (array): Distribution to calculate the entropy

    Returns:
        float: return the entropy
    """          
        
    _map = np.array(p)
    _map = list(_map.flatten())    
    _map = collections.Counter(_map)        
    _sum = sum(_map[e] for e in _map.keys())
    res = 0
    
    for e in _map.keys():
        p = _map[e] / _sum
        res -= w * (p*math.log(p, b))
    return w * res
    

def convert_strarray_to_dic(aStr):
    dic_array = {}
    for i in range(len(aStr)):
        tile = aStr[i]
        dic_array[tile] = i

    return dic_array

def counter_occurrences(map):
    _map = np.array(map)
    _map = list(_map.flatten())    
    _map = collections.Counter(_map)        
    return _map

def calc_prob_dist(map):
    _map = np.array(map)
    _map = list(_map.flatten())    
    _map = collections.Counter(_map)        
    _sum = sum(_map[e] for e in _map.keys())
    _dist = {}
    
    for key, value in _map.items():    
        _dist[key] = value / _sum    

    return _dist

def wave_front_entrace(grid, row, col, h = 0):                    
    n = len(grid)
    m = len(grid[0])    
    queue = Queue()    
    queue.put((row, col))    
    grid[grid == 1] = -99
    max_dist = -1
    grounds = 0
    ent = 0
    while not queue.empty():    
        row, col = queue.get()      
        nei = neighbors(row, col, n-1, m-1, four_way=True)                                        
        
        r1, cl = row, min(m, col + 1) 
        r2, cr = row, max(0, col - 1)

        r3, ct = max(0, row - 1), min(m, col) 
        r4, cb = min(n, row + 1), max(0, col)             
        if grid[row][col] != -99:
            grounds += 1

        if (grid[r1][cl] == -99 and grid[r2][cr] == -99 and grid[row][col] != 0) or \
            grid[r3][ct] == -99 and grid[r4][cb] == -99 and grid[row][col] != 0:
            ent += 1
            #grid[row][col]  = 100 #(grid[row][col] + 10) + h
        #else:
        #    grid[row][col]  = 10 #(grid[row][col] + 10) + h
            

        for r, c in nei:                                               
            if grid[r][c] != 0:
                #if grid[r][c] == -99:
                #    ent += 1
                continue
            else:
                grid[r][c]  = (grid[row][col] + 1) + h                
                queue.put((r , c))
                max_dist = max(max_dist, grid[r][c])                             

    return grid, max_dist, ent, grounds


def wave_front(grid, row, col, h = 0):                    
    n = len(grid)
    m = len(grid[0])    
    queue = Queue()    
    queue.put((row, col))    
    grid[grid == 1] = -99# 0x40000
    max_dist = -1
    ent = 0
    while not queue.empty():    
        row, col = queue.get()      
        nei = neighbors(row, col, n-1, m-1, four_way=True)                                        
        for r, c in nei:                                               
            if grid[r][c] != 0:
                continue
            else:
                grid[r][c]  = (grid[row][col] + 1) + h                
                queue.put((r , c))
                max_dist = max(max_dist, grid[r][c])                             

    return grid, max_dist   

def wave_front2(grid, row, col, replace_values = None):                    
    n = len(grid)
    m = len(grid[0])    
    queue = Queue()    
    queue.put((row, col))    
    dist = 0
    grid[grid == 1] = -1 #0x40000
    if not replace_values is None:        
        for v in replace_values:
             grid[grid == v] = 0
    while not queue.empty():    
        row, col = queue.get()              
                       
        if grid[row][col] != 0:                
            continue
        else:
            nei = neighbors(row, col, n, m)                           
            dist += 1 
            grid[row][col]  = dist #grid[row][col] + 1#dist                                                          
            for r, c in nei:                                                          
                queue.put((r , c))
    return grid    

"""
    Returns a dictionary of all position of elements.  
"""
def array_el_locations(array):

    locations = {}   

    n = array.shape[0]
    m = array.shape[1]    

    for row in range(n):
        for col in range(m):     
            key = array[row][col]            
            if key not in locations:            
                locations[key] = []

            locations[key].append((row,col))

    return locations

def reconstruct_path(came_from, current, pathMap):
    final_path = [current]
    while current in came_from:
        current = came_from[current]
        final_path.append(current)
    final_path = final_path[::-1]
    for x, y in final_path:
        pathMap[x][y] = 150
        plt.subplot(2, 2, 3)
        plt.xticks([]), plt.yticks([])
        plt.imshow(pathMap)
        plt.title('A* final path')
        plt.pause(0.1)
    return final_path


def heauristic(cell, goal):
    x1, y1 = cell
    x2, y2 = goal

    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist


def A_star(graph, start, goal, pathMap):
    pathMap[start[0]][start[1]] = 25

    closed_set = []  # nodes already evaluated

    open_set = [start]  # nodes discovered but not yet evaluated

    came_from = {}  # most efficient path to reach from

    gscore = {}  # cost to get to that node from start

    for key in graph:
        gscore[key] = 100  # intialize cost for every node to inf

    gscore[start] = 0

    fscore = {}  # cost to get to goal from start node via that node

    for key in graph:
        fscore[key] = 100

    fscore[start] = heauristic(start, goal)  # cost for start is only h(x)
    print(fscore[start])

    while open_set:
        min_val = 1000  # find node in openset with lowest fscore value
        for node in open_set:
            if fscore[node] < min_val:
                min_val = fscore[node]
                min_node = node

        current = min_node  # set that node to current
        if current == goal:
            return reconstruct_path(came_from, current, pathMap)
        open_set.remove(current)  # remove node from set to be evaluated and
        closed_set.append(current)  # add it to set of evaluated nodes

        for neighbor in graph[current]:  # check neighbors of current node
            if neighbor in closed_set:  # ignore neighbor node if its already evaluated
                continue
            if neighbor not in open_set:  # else add it to set of nodes to be evaluated
                open_set.append(neighbor)

            # dist from start to neighbor through current
            tentative_gscore = gscore[current] + 1

            # not a better path to reach neighbor
            if tentative_gscore >= gscore[neighbor]:
                continue
            came_from[neighbor] = current  # record the best path untill now
            gscore[neighbor] = tentative_gscore
            fscore[neighbor] = gscore[neighbor] + heauristic(neighbor, goal)
            pathMap[neighbor[0]][neighbor[1]] = 25

        img = pathMap                 # Display path while building
        
        plt.subplot(2, 2, 2)
        plt.xticks([]), plt.yticks([])
        plt.imshow(img)
        plt.title('A* search')
        plt.pause(0.1)
        
    return False


def mat2graph(mat):
    rows = len(mat)
    cols = len(mat[0])
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x][y] == 0:                                                
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= x+dx < rows and 0 <= y+dy < cols and mat[x+dx][y+dy] == True:
                        graph[(x, y)].append((x+dx, y+dy))
    return graph   