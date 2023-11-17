import numpy as np


N = 10
valuesAmount = 50

with(open("test-matrix.txt","w") as file):
    file.write("{} {} {}\n".format(N,N,valuesAmount))
    for i in range(valuesAmount):
        row = np.random.randint(50)
        col = np.random.randint(50)
        value = np.random.randint(10)
        file.write("{} {} {}\n".format(row,col,value))

file.close()
