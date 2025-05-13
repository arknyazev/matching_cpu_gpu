import numpy as np
timelost = np.loadtxt('output/timelost.txt')
tmax = 1e-3
count = 0
for t in timelost:
    count += (t < tmax)

print(f'Lost particle {count=}')
print(f'This is {count/len(timelost)} of all partickes')
