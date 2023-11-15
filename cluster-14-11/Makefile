cuda:
	/usr/local/cuda-11.2/bin/nvcc matrix-mult-cuda-threads-v2.cu -o matrix-mult-cuda-threads-v2
	/usr/local/cuda-11.2/bin/nvprof ./matrix-mult-cuda-threads-v2
clean:
	rm matrix-mult-cuda-threads-v2 matrix-mult-cuda-threads
cudaV2:
	/usr/local/cuda-11.2/bin/nvcc matrix-mult-cuda-threads.cu -o matrix-mult-cuda-threads
	/usr/local/cuda-11.2/bin/nvprof ./matrix-mult-cuda-threads
