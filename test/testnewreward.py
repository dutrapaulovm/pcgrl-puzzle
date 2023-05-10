from pcgrl.Utils import * 

grid = [1, 1, 2, 2, 1, 1]
print(grid)
_map = collections.Counter(grid)
n_seg = len(grid)
count = 0
dist  = 0
steps = 0
w = entropy(grid)
score = 0
for k in _map.keys():
    print(k)
    dist  = n_seg
    count = 0
    steps = 0
    d = 0
    for j in range(n_seg):
        if k == grid[j] and d > 0:
            score += 1        
        if k == grid[j] and d == 0:
            d += 1

#if (score == 0):
print("H = ", w)
print("Score: ", -(score * w))
#else:
#    print(score * w)