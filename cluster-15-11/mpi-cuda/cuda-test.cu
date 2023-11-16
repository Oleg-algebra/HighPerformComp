/* multiply.cu */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ 
void multiply(int N, float *a, float *b){
    a[0] = 1.0f;
    b[0]= 1.0f;
}

extern "C" void launch_multiply(int N, float *a, float *b)
{
    float *a_gpu, *b_gpu;
    cudaMallocManaged(&a_gpu, N*sizeof(float));
    cudaMallocManaged(&b_gpu, N*sizeof(float));
	
    for(int i = 0; i<N; i++){
		a_gpu[i] = a[i];
		b_gpu[i] = b[i];
    }
	
    printf("cuad-test: N = %d\n",N);
    multiply<<< 1 , 1 >>> (N,a_gpu, b_gpu);
    cudaDeviceSynchronize();
    
    for(int i = 0; i<N; i++){
		a[i] = a_gpu[i];
		b[i] = b_gpu[i];
	}
	cudaFree(a_gpu);
	cudaFree(b_gpu);
}