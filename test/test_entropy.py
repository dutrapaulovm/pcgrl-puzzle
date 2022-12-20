from math import ceil
import numpy as np
import sys
from pcgrl.Utils import *
"""
x1 = []
x2 = []
x1.append(2.58)
x1.append(2.58)
x2.append(1.80)
x2.append(1.80)
print("Distance: ",  2.2 - math.log2(manhattan_distance(x1, x2)) )


map = [[1, 2, 3], [4, 5, 6]]
print(entropy(map, b = 2, w = 1))

map = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(entropy(map, b = 2, w = 1))

map = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(entropy(map, b = 2, w = 1))

map = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
print(entropy(map, b = 2, w = 1))

map = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
print(entropy(map, b = 2, w = 1))
"""

obs = np.arange(3 * 2).reshape(2, 3)
print(round(entropy(obs), 2))

obs[0, 1] = 0
print(entropy(obs))
