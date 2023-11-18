main:  mpi-part.o cuda-part.o
	mpicc -Wl,--copy-dt-needed-entries -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/  mpi-part.o cuda-part.o -lcudart -lstdc++ -o main
mpi: mpi-part.c
	mpicc -c mpi-part.c -o mpi-part.o
cuda: cuda-part.cu
	/usr/local/cuda-11.2/bin/nvcc -c cuda-part.cu -o cuda-part.o
run: 
	mpirun -n 8 ./main
all:
	make mpi
	make cuda
	make main
	make run
clean:
	rm *.o main
