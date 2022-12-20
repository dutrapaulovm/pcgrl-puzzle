import numpy as np
import collections
# Setting your input to an array
array = np.array([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2],
                 [0, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2]])
_map = array
_map = list(_map.flatten())    
_map = collections.Counter(_map)        
_sum = sum(_map[e] for e in _map.keys())
print(_map[3])    

# Find the unique elements and get their counts
unique, counts = np.unique(array, return_counts=True)

# Setting the numbers to get counts for as an array
search = np.array([0, 1, 2, 3])

# Gets the counts for the elements in search
search_counts = [counts[i] for i, x in enumerate(unique) if x in search]

print(search_counts)