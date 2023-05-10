import numpy as np
import collections
import math
# -*- coding: utf-8 -*-
from distutils.log import error
from gym.utils import seeding
from IPython.display import clear_output
from IPython import get_ipython
import collections, numpy
from collections import defaultdict
import math
import numpy as np
import time     
import random
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
import heapq

array = np.array([[1, 2, 3],
                  [4, 5, 1]])


array1 = np.array([[1,2,3],
                   [4,5,1]])

max_entropy = np.array([[1,2,3],
                   [4,5,6]])

def manhattan_distance(xy1, xy2, w = 0):
  "Returns the Manhattan distance between points xy1 and xy2"
  #return math.sqrt(abs( xy1[0] - xy2[0] )**2 + abs( xy1[1] - xy2[1] )**2) 

  return w - (abs(xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1]))

def euclidean_distance(pointA, pointB):
    dist = np.linalg.norm(np.array(pointA) - np.array(pointB))
    return dist

def heauristic(cell, goal, w = 0):
    x1, y1 = cell
    x2, y2 = goal

    dist =  ( ((w**2) - ((x2-x1)**2 + (y2-y1)**2))**0.5)
    return dist

def neighbors(row, col, n, m, four_way=False):
    """Return indices of adjacent cells"""
    if four_way:
        return [ (max(0, row - 1), col), 
                 (row, min(m, col + 1)),
                 (min(n, row + 1), col), 
                 (row, max(0, col - 1))
                ]
    else:        
        return [
            (max(0, row - 1), col), (max(0, row - 1), min(m, col + 1)),
            (row, min(m, col + 1)), (min(n, row + 1), min(m, col + 1)),
            (min(n, row + 1), col), (min(n, row + 1), max(0, col - 1)),
            (row, max(0, col - 1)), (max(0, row - 1), max(0, col - 1))]
    
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

def get_positions(tiles, map):        
    max_row = map.shape[0]
    max_col = map.shape[1]
    new_map = []
    for row in range(max_row):
        for col in range(max_col):
            id = int(map[row][col])
            if id in tiles:
                new_map.append((row, col))
    return new_map

def reward_neighbors(segments):
        n, m = segments.shape
        map_segments = np.array(segments)        
        map_segments = list(map_segments.flatten())            
        #print(map_segments)
        #print(segments)
        #print(segments.shape)
        positions = get_positions(map_segments, segments)

        reward = 0

        for row, col in positions:
            segment = segments[row][col]
            nei = neighbors(row, col, n-1, m-1)                        
            #print("Segmento: {}, Row {}, Col {} ".format(segment, row, col))
            #print("\tVizinhos: ", nei)
            for r, c in nei:
                #print("\t\tVizinho: {}, Row {} e Col {}".format(segments[r][c], r, c))
                if (segments[r][c] != -1) and segments[r][c] == segment and (row != r or col != c):
                    reward += -2

        return reward

def euclidean(pointA, pointB):
    dist = np.linalg.norm(np.array(pointB) - np.array(pointA))
    return dist

def manhattan_distance(xy1, xy2, w = 0):
  "Returns the Manhattan distance between points xy1 and xy2"
  return w - (abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] ))

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


n_segments = array.shape[1] * array.shape[0]
_map = np.array(array)
_map = list(_map.flatten())
_map = collections.Counter(_map)
counter = 0
print(_map)
for e in _map.keys():    
    if _map[e] > 1:
       counter += _map[e]

print("Repetidos: ", counter)
print("NSegmentos: ", n_segments)
segments = set(array.flatten())    

print(_map)
print(segments)

reward_m = 0
reward_e = 0
reward_h = 0

for segment in segments:
    print("Segmento: ", segment)
    positions = get_tile_positions([segment], array)
    print("\t",positions)        
    if len(positions) > 1:    
        pos_init = positions[0]
        for row, col in positions:
            reward_m += (manhattan_distance(pos_init, (row, col), n_segments))
            reward_e += (euclidean_distance(pos_init, (row, col)))            
            reward_h += (heauristic(pos_init, (row, col), n_segments))            

sign = lambda x: math.copysign(1, x)
reward_e = reward_e
reward_m = reward_m**2
reward_h = reward_h**2

e = entropy(array)
maxe = entropy(max_entropy) 
entropy_min = entropy(array1)
print("Entropy min", e)
x = math.pi
r = (e**x - entropy_min**x)                
f = 1
reward = (((r + sign(r)) * f)) 

print("Euclediana:" , reward_e)# * counter * maxe)
print("Mahatam:"    , reward_m)
print("Heuristica:" , reward_h)
print("Recompensa: ", reward)
print("Recompensa Euclidiana: ", (reward + reward_e))
print("Recompensa Manhatam: "  , (reward + reward_m))
print("Neighbors: "  , (reward_neighbors(array)*-1))
print("Neighbors + Euclidiana: ", (reward_neighbors(array)*-1)+reward_e*-1)