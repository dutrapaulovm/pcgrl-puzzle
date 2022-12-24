import random

# Constants for the maze dimensions
MAZE_WIDTH = 10
MAZE_HEIGHT = 8

# Create an empty maze
maze = []
for y in range(MAZE_HEIGHT):
    maze.append([0] * MAZE_WIDTH)

# Generate the maze
for y in range(1, MAZE_HEIGHT, 2):
    for x in range(1, MAZE_WIDTH, 2):
        maze[y][x] = 1
        neighbors = []
        if x > 1:
            neighbors.append((y, x - 2))
        if x < MAZE_WIDTH - 2:
            neighbors.append((y, x + 2))
        if y > 1:
            neighbors.append((y - 2, x))
        if y < MAZE_HEIGHT - 2:
            neighbors.append((y + 2, x))
        if neighbors:
            y2, x2 = random.choice(neighbors)
            maze[y2][x2] = 1
            maze[y2][min(MAZE_WIDTH-1, x2 + 1)] = 1
            maze[min(MAZE_HEIGHT-1, y2 + 1)][x2] = 1

# Print the maze
for row in maze:
    print(','.join([str(cell) for cell in row]))