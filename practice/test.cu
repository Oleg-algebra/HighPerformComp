#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add()
{
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
  printf("gridDim.x: %d -- blockIdx.x: %d -- blockDim.x: %d -- threadIdx.x: %d\n",
            gridDim.x,blockIdx.x,blockDim.x,threadIdx.x);

}

int main(void)
{
    int N = 1<<20;

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	printf("numBlocks: %d\n",numBlocks);
	add<<<2,3>>>();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
    // maxError = fmax(maxError, fabs(y[i]-3.0f));
//   std::cout << "Max error: " << maxError << std::endl;

  // Free memory
//   cudaFree(x);
//   cudaFree(y);
  
  return 0;
}