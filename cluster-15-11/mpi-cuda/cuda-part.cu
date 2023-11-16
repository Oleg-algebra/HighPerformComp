#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <math.h>


__global__ 
void multMatrixVector(int nPoints,int* rows, int*cols ,double *vals, double *v, double *resVector){
    //TODO: rewrite function

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<nPoints;i+=stride){
        double value = vals[i] * v[cols[i]];
	
        resVector[rows[i]] = resVector[rows[i]] + value;
	if(cols[i] == 0){
            printf("block: %d -- thread: %d -- row index: %d\n",blockIdx.x,threadIdx.x,rows[i]);
            printf("value written to vector: %f\n",value);
	    printf("res[%d] = %f\n",rows[i],resVector[rows[i]]);
	}

    }

}


extern "C" void launch_multiply(int nPoints,int* rows, int*cols ,double *vals, double *v, double *resVector)
{

}