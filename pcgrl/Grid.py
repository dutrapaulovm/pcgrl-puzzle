import numpy as np

class Node:
    """
    Node used for Grid
    """
    def __init__(self, x = 0, y = 0, value = -1, blocked = False, parent = None):
        self.x = x
        self.y = y
        self.value = value
        self.blocked = blocked
        self.parent = None        
        
class Grid:
    """
    Grid represents a world that has coordinate X and Y.    
    """
    def __init__(self, sizeX = 0, sizeY = 0, default_value = -1):
        self.sizeX  = sizeX
        self.sizeY = sizeY;        
        self.default_value = default_value
        self._create_grid()

    def create_grid(self):
        """Create a grid
        """
        self.data = np.full((self.sizeX * self.sizeY), self.default_value)               
    
    def get_neighbours(self, x, y):    
        """[Returns all neighbours of position x, y ]

        Args:
            x (int): position x in grid
            y (int): position y in grid

        Returns:
            list(Node): Returns a list with all neighbours of x, y
        """
        neighbours = np.array([])
        node = Node(x, y)

        for _x in range(-1, 1):
            for _y in range(-1, 1):
                if (_x == 0 and _y == 0):
                    continue

                checkX = node.x + _x
                checkY = node.y + _y

                if (checkX >= 0 and checkX < self.sizeX and checkY >= 0 and checkY < self.sizeY):
                    neighbours.append(self.data[checkX, checkY])

        return neighbours

    def resize(self, sizeX, sizeY, default_value = - 1):
        """[Resize this grid]

        Args:
            sizeX ([int]): Size in coordinate X
            sizeY ([int]): Size in coordinate Y
            default_value (int, optional): Defaul value value for all positions. Defaults to -1.
        """
        self.sizeX  = sizeX
        self.sizeY = sizeY        
        self.default_value = default_value
        self._create_grid()
        
    def set(self, x, y, value, blocked = False):
        """Set the value in position x and y

        Args:
            x (int): coordinate X
            y ([type]): coordiante Y
            value (any): value for coordinates X and Y
            blocked (bool, optional): Set if position is blocked. Defaults to False.
        """
        self.data[x, y] = Node(x=x, y=y, value=value, blocked=blocked)

    def get(self, x, y):        
        return self.data[x, y]