test:  mpi-test.o cuda-test.o
	mpicc -Wl,--copy-dt-needed-entries -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/  mpi-test.o cuda-test.o -lcudart -lstdc++ -o test
main: mpi-test.c
	mpicc -c mpi-test.c -o mpi-test.o
cuda: cuda-test.cu
	/usr/local/cuda-11.2/bin/nvcc -c cuda-test.cu -o cuda-test.o
run: 
	mpirun -n 4 ./test
all:
	make main
	make cuda
	make test
	make run
clean:
	rm *.o test
