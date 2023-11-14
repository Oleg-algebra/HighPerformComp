import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix

from time import time

matrix = mmread("rawData/sparsine/sparsine.mtx")
lst = []
count = 0
print(matrix.nnz)
for tup in zip(matrix.row,matrix.col,matrix.data):
    lst.append(tup)

lst = np.asarray(lst)
# print(lst[:10])
# lst = lst[lst[:,0].argsort()]
# print(lst[:10])
with(open("cmake-build-debug/matrices/newSparsine2.txt", "w") as file):
    file.write("{} {} {}\n".format(matrix.shape[0],matrix.shape[1],matrix.nnz))
    for tup in lst:
        file.write("{} {} {}\n".format(tup[0],tup[1],tup[2]))

    file.close()
print("done")