import numpy as np

from time import time

lines = []

with open('matrices/sparsine.mtx') as f:
    
    lines = [list(map(float,line.rstrip().split())) for line in f]
 
for i in range(10):
    print(lines[i])

head = lines[0]
lines = np.asarray(lines[1:])
lines = lines[lines[:,0].argsort()]

with open("matrices/sparsine1.mtx", "w") as file:
    file.write("{} {} {}\n".format(int(head[0]),int(head[1]),int(head[2])))
    for tup in lines:
        file.write("{} {} {}\n".format(int(tup[0]),int(tup[1]),tup[2]))

    file.close()
print("done")

with open("matrices/sparsine1.mtx", "r") as file:
    for i in range(10):
        print(file.readline(),end="")
    file.close()
print("done")


