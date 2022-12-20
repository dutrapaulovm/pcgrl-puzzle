import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pcgrl.Utils import *
import csv
import imageio
# Driver code
# Driver code
screen =np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,8,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,7,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1],
[1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,7,0,0,0,0,1],
[1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,5,1,0,0,1,0,0,1],
[1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,4,1,0,0,0,4,0,1,0,0,1,4,0,1],
[1,0,0,0,0,0,6,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1],
[1,1,0,0,1,1,2,1,1,0,4,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
[1,1,0,0,1,1,0,0,1,7,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1],
[1,1,0,0,0,0,0,0,0,0,0,1,0,0,4,1,0,0,0,1,0,3,1,0,1,1],
[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
[1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,1,1],
[1,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1],
[1,1,0,0,4,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1],
[1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,7,1,1],
[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ])

passables = 0
"""
replaces_values = [3, 4, 6,  7]


if (replaces_values is not None):
    _map = screen.copy()            
    
    for tile in replaces_values:
        _map[_map == tile] = passable
                            
    screen = _map
"""
print(screen.shape)

locations = array_el_locations(screen)

data_flood_fill = screen.copy()    
images = []
locations = locations[0]
passable = 0
num_tiles   = 0
num_regions = 0
for row, col in locations:             
    num_tiles = flood_fill_v4(data_flood_fill, row, col, [0, 2, 3, 4, 5, 6, 7])
    if (num_tiles > 0):
        num_regions += 1

#print(data_flood_fill)
fig, ax = plt.subplots(figsize=(10,10))

for _i in range(len(data_flood_fill)):
    for _j in range(len(data_flood_fill[0])):
        text = ax.text(_j, _i, data_flood_fill[_i][_j],
                    ha="center", va="center", color="w")
path = os.path.dirname(__file__) 
#imageio.mimsave(f"{path}/flood.gif", [np.array(img) for i, img in enumerate(images)], fps=30)

plt.title("Num. of Regions: {}".format(num_regions))
plt.tight_layout()
plt.savefig(f"{path}/flood.png")
plt.imshow(data_flood_fill)
plt.show()