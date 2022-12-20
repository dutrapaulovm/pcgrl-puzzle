from pcgrl.Agents import * 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

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
                for dx, dy, in neighbors(x, y, rows, cols):                                           
                    graph[(x, y)].append((dx, dy))
                #for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    #if 0 <= x+dx < rows and 0 <= y+dy < cols and mat[x+dx][y+dy] == True:
                        #graph[(x, y)].append((x+dx, y+dy))
    return graph

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

agent = AStartAgent()

path = path = os.path.dirname(__file__) 
path = os.path.join(path, "map")
dt = list(pd.read_csv(path + "\Map1.csv").values)       
print(np.array(dt))
#player = get_tile_positions([6], dt)
#doorexit = get_tile_positions([3], dt)
#print(player)
#print(doorexit)
#agent.row = 2
#agent.col = 2 
#print(agent.isPathV2(dt, 5, 2, (1, 2, 3 , 4, 6)))

start = (1,1)
destination = (1, 16)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(dt, interpolation='nearest')
plt.xticks([]), plt.yticks([])
graph = mat2graph(dt)

print(find_solution(dt.copy(), (1, 1), (17,19), [0, 2]))

#print(graph)
#shortest_route = A_star(graph, start, destination, dt)
#print(shortest_route)

#plt.subplot(1, 2, 2)
#plt.imshow(dt)
#plt.show()