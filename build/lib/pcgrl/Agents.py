# -*- coding: utf-8 -*-
import os

from gym.spaces import space

from pcgrl.Entity import Entity
from pcgrl.AgentBehavior import *
from pcgrl.Utils import *
from pcgrl.Generator import Generator

import numpy as np
from numpy.random.mtrand import get_state
import pandas as pd
import time     
import random
from enum import Enum

from gym.utils import seeding
from gym import spaces
from collections import OrderedDict

class Actions(Enum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STOP  = 4

    def __str__(self):
        return format(self.value)                                            
        
    def __int__(self):
        return self.value  
    
class Behaviors(Enum):                     
    NARROW_PUZZLE                  = "narrow-puzzle"
    WIDE_PUZZLE                    = "wide-puzzle"
    MULTI_PIECE                    = "multi-piece"
    
    def __str__(self):
        return self.value                                            

class BaseAgentBehavior(AgentBehavior):
    
    def __init__(self, max_iterations=None, env=None):
        super(BaseAgentBehavior, self).__init__(max_iterations=max_iterations)
        self.env = env
        self.observation_space = env.observation_space
        self.iterations = 0
        self.max_iterations = max_iterations     
        self.num_states = 1
        self.observations = [OrderedDict() for x in range(self.num_states)]                        
    
    def is_done(self):
        return False

    def get_info(self):
        return {}

    def get_current_observation(self, info):        
        
        states = self.get_stats()
        map = self.env.game.map
        
        state = states["state"]

        obs = OrderedDict({
            "map"   : map.copy(),
            "state" : state,            
            "stats" : states
        })
                        
        return obs

    def reset(self):
                
        super().reset()
        
        game = self.env.game        
        width = game.get_width()        
        state_w = game.get_state_width()
        state_h = game.get_state_height()
        
        self.current_state = self.np_random.randint(0, self.env.num_states)

        pos = decodedXY(self.current_state, width, state_w, state_h)
        self.entity.set_pos(pos.x, pos.y)
        
        self.last_action = -1
                                
        reward, posc, posd, tiles = 0, [], [], 0
                
        obs = self.get_current_observation({"posc" : posc, "posd" : posd, "reward" : reward, "action" : -1 })
        self.last_observation = obs
        self.last_stats = self.get_stats()                       
                        
        return obs
        
    def get_stats(self):        
        
        game = self.env.game
        stats = {          
            "state"     : 0,
            "map_stats" : game.get_map_stats()
        }
                                                
        return stats
                   
        
class LevelDesignerAgentBehavior(BaseAgentBehavior):
    """
    LevelDesignerAgentBehavior represents the behavior level designer agent. This agent has actions to changes the
    an environment called representations
    There are two representations: NARROW_PUZZLE e WIDE_PUZZLE
    """  
    def __init__(self, max_iterations=None, env = None, rep = None, piece_size = (4, 4), 
                       path_pieces = None, action_change = False, n_models = -1, extra_actions = {}):
        
        super(LevelDesignerAgentBehavior, self).__init__(max_iterations=max_iterations, env = env)

        offset = env.game.border_offset()
        dim = np.array(env.game.get_dim()).copy()  
        self.current_piece_index = 0             
        
        if (offset[0] > 0 and offset[1] > 0):
            dim[1] = dim[1] - (offset[0] * env.game.tile_height) * 2 
            dim[0] = dim[0] - (offset[1] * env.game.tile_width)  * 2 
        
        self.max_rows = int((dim[1] / env.game.tile_height) / piece_size[0])
        self.max_cols = int((dim[0] / env.game.tile_width)  / piece_size[1])
        board_pieces = int(self.max_cols * self.max_rows)                       
        self.total_board_pieces = board_pieces
        self.generator =  Generator(path=path_pieces, piece_size=piece_size, dim=dim, n_models = n_models)        
        self.last_piece = np.zeros(piece_size)
        self.last_action    = -100
        self.action_change = action_change        
        self.total_pieces = self.generator.action_space.n
        self.pieces = []
        for i in range(self.total_board_pieces):
            self.pieces.append([])

        if (rep == Behaviors.NARROW_PUZZLE.value):                        
            
            self.generator.max_cols = self.max_cols
            self.generator.max_rows = self.max_rows            
            
            actions = [self.generator.action_space.n]
            if (self.action_change):
                actions.append(2)            
            else:                                
                for k, v in extra_actions.items():                    
                    actions.append(v)

            self.action_space = spaces.MultiDiscrete(actions)

        elif (rep == Behaviors.WIDE_PUZZLE.value):                    
            
            self.generator.max_cols = self.max_cols
            self.generator.max_rows = self.max_rows            
            
            #actions = [self.max_cols, self.max_rows, self.generator.action_space.n, 1]
            actions = [self.max_cols, self.max_rows, self.generator.action_space.n]
            if (self.action_change):
                actions = [self.max_cols, self.max_rows, self.generator.action_space.n, 2]                
            
            for k, v in extra_actions.items():                    
                actions.append(v)                
            
            self.action_space = spaces.MultiDiscrete(actions)

        elif (rep == Behaviors.MULTI_PIECE.value):                    
            self.generator.max_cols = self.max_cols
            self.generator.max_rows = self.max_rows
            actions = []
            for _ in range(board_pieces):
                actions.append(self.generator.action_space.n)

            for k, v in extra_actions.items():                    
                actions.append(v)
                
            self.action_space = spaces.MultiDiscrete(actions)
            #action_space = gym.spaces.Box(0, generator.action_space.n, shape=(total_pieces,), dtype='int32')
            #self.action_space = gym.spaces.Box(0, self.generator.action_space.n, shape=(total_pieces,), dtype=np.int32)

        self.last_action  = -1        
        self.entity = None
        self.representation = rep
        self.grid = np.full((self.max_rows, self.max_cols), -1)
        self.grid_pieces = np.full((self.max_rows, self.max_cols), -1)
        #print(self.total_pieces)
        #time.sleep(5)

    def reset(self):
        self.generator.reset()
        self.grid = np.full((self.max_rows, self.max_cols), -1)
        self.grid_pieces = np.full((self.max_rows, self.max_cols), -1)
        self.pieces = []
        for i in range(self.total_board_pieces):
            self.pieces.append([])    
        self.current_piece_index = 0    
        return {}

    def is_done(self):
        return (-1 not in self.grid)        

    def get_info(self):
        info = {}
        info["Pieces"] = self.grid

    def step(self, action):
        
        game = self.env.game        
                
        obs     = []
        info    = []        
        reward  = -1        
        change  = 0     
        piece   = []          

        if (self.representation == Behaviors.NARROW_PUZZLE.value):
            
            print("Narrow Puzzle: ", action)
            print()

            do_change = True
            act = action[0]
            reward = 0.0

            if self.action_change:
                act = action[0]
                do_change = (action[1] == 1)

            if do_change:
                print("Alterou")
                x = self.generator.curr_col
                y = self.generator.curr_row
                piece  = self.grid[y][x]
                v = [0, 1][piece != act]
                change += v                  
                self.grid[y][x] = act
                                
                game.map, piece = self.generator.build_map(game.map, act,  offset=self.env.game.border_offset())                             
                game.clear()
                game.create_map(game.map)
                reward = 0#js_divergence(self.last_piece, piece)                
                self.last_piece = piece                                
                
                #p = (self.max_rows * self.generator.curr_row-1) + self.generator.curr_col-1                
                self.pieces[self.current_piece_index] = piece                 
              
                self.last_piece = piece
                self.last_action = act
                self.current_piece_index += 1
            else:
                print("Não Alterou")   
                
        elif (self.representation == Behaviors.WIDE_PUZZLE.value):                                              
            print("Wide Puzzle: ", action)
            print()
            #try:        
            do_change = True            
            if self.action_change:            
                do_change = (action[3] == 1)
            reward = 0.0

            if do_change:         
                print("Alterou")
                x = action[0]
                y = action[1]
                piece  = self.grid[y][x]
                v = [0, 1][piece != action[2]]
                change += v                                
                self.grid[y][x] = action[2]

                #change += 1
                game.map, piece = self.generator.build_map_xy(game.map, action[2], x, y, offset=self.env.game.border_offset())
                game.clear()
                game.create_map(game.map)            
                reward = js_divergence(self.last_piece, piece)                
                self.last_piece = piece
#                p = (self.max_rows * self.generator.curr_row-1) + self.generator.curr_col-1
                self.pieces[self.current_piece_index] = piece 
                self.last_action = action[2]
                self.current_piece_index += 1
            else:
                print("Não Alterou")   
            #except :
             #   print('Ocorreu um erro durante a criação do mapa. Representation {}'.format(self.representation))
                #print(game.map)            
        elif (self.representation == Behaviors.MULTI_PIECE.value):                                      
            #print("Multi Piece: ", action )
            #print()            
            #try:                          
            r = []
            change = 0            
            print("Multi Puzzle: ", action)
            print()

            for a in range(len(action)):
                
                x = self.generator.curr_col
                y = self.generator.curr_row
                piece  = self.grid[y][x]
                v = [0, 1][piece != action[a]]
                change += v                  
                self.grid[y][x] = action[a]
                #change += 1
                game.map, piece = self.generator.build_map(game.map, action[a], offset=self.env.game.border_offset())
                game.clear()                      
                game.create_map(game.map)                
                r.append(js_divergence(self.last_piece, piece))
                self.last_piece = piece
                
            reward = np.array(r).mean()

            #except :
             #   print('Ocorreu um erro durante a criação do mapa. Representation {}'.format(self.representation))
                #print(game.map)
                                        
        info = {"reward" : reward, "change" : change, "action" : action}        
        
        obs = self.get_current_observation(info)
        
        self.last_action = action
        self.last_observation = obs                
        self.last_stats = self.get_stats()
        
        if self.current_piece_index >= self.total_board_pieces:
            self.current_piece_index = 0
                
        self.add_reward(reward)
                
        return reward, change, piece #obs, reward, done, info   


#https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)    
            
def main_test():

    maze = [[5, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 8]]

    start = (0, 0)
    end = (9, 9)

    path = astar(maze, start, end)
    print(path)
    
class AStartAgent:
    """
    Agent to check if path is valid    
    """
    def __init__(self):        
        self.row = 5
        self.col = 5 
        
    # to find the path from
    # top left to bottom right
    def isPathV2(self, arr, row, col, blockeds = []) :
        #https://www.geeksforgeeks.org/check-possible-path-2d-matrix/
        # directions
        Dir = [ [0, 1], [0, -1], [1, 0], [-1, 0]]
        
        # queue
        q = []        
        
        # insert the top right corner.
        q.append((row, col))
        
        # until queue is empty
        while(len(q) > 0) :
            p = q[0]
            q.pop(0)
            
            # mark as visited
            arr[p[0]][p[1]] = -1
            
            # destination is reached.
            if(p == (self.row - 1, self.col - 1)) :
                return True
                
            # check all four directions
            for i in range(4) :
            
                # using the direction array
                a = p[0] + Dir[i][0]
                b = p[1] + Dir[i][1]                
                # not blocked and valid
                if(a >= 0 and b >= 0 and a < self.row and b < self.col and arr[a][b] != -1) :           
                    
                    q.append((a, b))

        return False

        """
        # Given array
        arr = [ [ 0, 0, 0, -1, 0 ],
                [ -1, 0, 0, -1, -1 ],
                [ 0, 0, 0, -1, 0 ],
                [ -1, 0, -1, 0, -1 ],
                [ 0, 0, -1, 0, 0 ] ]
        
        # path from arr[0][0] to arr[row][col]
        if (isPath(arr)) :
            print("Yes")
        else :
            print("No")        
        """                 
    
    # Python3 program to find
    # path between two cell in matrix
    
    # Method for finding and printing
    # whether the path exists or not
    def isPath(self, matrix, n):
        #https://www.geeksforgeeks.org/find-whether-path-two-cells-matrix/
        # Defining visited array to keep
        # track of already visited indexes
        visited = [[False for x in range (n)]
                        for y in range (n)]
        
        # Flag to indicate whether the
        # path exists or not
        flag = False
    
        for i in range (n):
            for j in range (n):
            
                # If matrix[i][j] is source
                # and it is not visited
                if (matrix[i][j] == 1 and not
                    visited[i][j]):
    
                    # Starting from i, j and
                    # then finding the path
                    if (self.checkPath(matrix, i, j, visited)):                    
                        # If path exists
                        flag = True
                        break
        if (flag):
            print("YES")
        else:
            print("NO")
    
    # Method for checking boundaries
    def isSafe(self, i, j, matrix):    
        if (i >= 0 and i < len(matrix) and
            j >= 0 and j < len(matrix[0])):
            return True
        return False
    
    # Returns true if there is a
    # path from a source(a
    # cell with value 1) to a
    # destination(a cell with
    # value 2)
    def checkPath(self, matrix, i, j, visited):
    
        # Checking the boundaries, walls and
        # whether the cell is unvisited
        if (self.isSafe(i, j, matrix) and
            matrix[i][j] != 0 and not
            visited[i][j]):
        
            # Make the cell visited
            visited[i][j] = True
    
            # If the cell is the required
            # destination then return true
            if (matrix[i][j] == 2):
                return True
    
            # traverse up
            up = self.checkPath(matrix, i - 1,
                        j, visited)
    
            # If path is found in up
            # direction return true
            if (up):
                return True
    
            # Traverse left
            left = self.checkPath(matrix, i, j - 1, visited)
    
            # If path is found in left
            # direction return true
            if (left):
                return True
    
            # Traverse down
            down = self.checkPath(matrix, i + 1,
                            j, visited)
    
            # If path is found in down
            # direction return true
            if (down):
                return True
    
            # Traverse right
            right = self.checkPath(matrix, i,  j + 1, visited)
    
            # If path is found in right
            # direction return true
            if (right):
                return True
        
        # No path has been found
        return False    
    
      # calling isPath method
    #isPath(matrix, 4)   