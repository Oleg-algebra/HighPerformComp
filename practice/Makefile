plain:
	g++ -o vector-add vector-add.c
#/usr/local/cuda-12.1/extras/dero suite/deviceQuery
hello:
	export PATH=/usr/local/cuda-12.1/bin:$PATH
	nvcc -gencode arch=compute_50,code=sm_50 cuda-hello.cu -o cuda-hello
cuda:
	/usr/local/cuda-12.1/bin/nvcc -gencode arch=compute_50,code=sm_50 cuda-vector-add-plain.cu o cuda-vector-add-plain
clean:
	rm vector-add cuda-hello cuda-vector-add-plain