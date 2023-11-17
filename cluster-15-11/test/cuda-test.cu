/* multiply.cu */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;


__global__ 
void multMatrixVector(int nPoints,int* rows, int*cols ,double *vals, double *v, double *resVector){
    //TODO: rewrite function

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<nPoints;i+=stride){
        double value = vals[i] * v[cols[i]];
	double oldValue = resVector[rows[i]];
	double newValue = oldValue + value;

        resVector[rows[i]] = newValue;
	
	if(cols[i] == 0){
            //printf("res[16668] = %f\n",resVector[16668]);
            //printf("block: %d -- thread: %d -- row index: %d\n",blockIdx.x,threadIdx.x,rows[i]);
            //printf("value written to vector: %f  -- resVector[%d] = %f\n",value,rows[i],resVector[rows[i]]);
	    printf("col: %d -- resV[%d] = %f -- v[%d] = %f -- value: %f\n",cols[i],rows[i],resVector[rows[i]],cols[i], v[cols[i]],newValue);
	      
              	
	}

    }

}

void getData(const string& dataString, double *storage){
    stringstream ss(dataString);
    string singleData;
    int i = 0;
    while (getline(ss,singleData,' ')) {
//        cout << "Data: "<< singleData << "\n";
        storage[i] = std::stod(singleData);
        i++;
        // store token string in the vector

    }
}

void readMatrix(int *cols,int *rows, double *vals, string& fileName){

    fstream file;
    file.open(fileName,ios::in);
    int counter = 0;
    if(file.is_open()){
        string text;
        double *values = new double[3];
        getline(file,text);
        getData(text,values);
        
        while(getline(file,text)){
            getData(text,values);
            cols[counter] = (int)values[1];
	    rows[counter] = (int)values[0];
	    vals[counter] = values[2];
            counter++;
            }
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
    
}

void readHead(const string &fileName, int *headData){
    fstream file;
    file.open(fileName,ios::in);
    double *values = new double[3];
    if(file.is_open()){
        string text;
        
        getline(file,text);
        getData(text,values);
		headData[0] = values[0];
		headData[1] = values[1];
		headData[2] = values[2];
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
}

void printMatrix(int n, int* rows, int* cols, double* vals){
	for(int i = 0; i<n;i++){
		printf("col: %d -- row: %d -- val: %f\n",rows[i],cols[i],vals[i]);
	}
}

void printVector(int n, double*v){
    printf("printing vector\n");
    for(int i = 0;i<n;i++){
        printf("v[%d] = %f\n",i,v[i]);
    }
}

extern "C" void launch_multiply(int rank, double *vector, double *resVector)
{
    //printf("rank %d choosing GPU....\n",rank);
    cudaSetDevice(rank);
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    double *vector_gpu, *resVector_gpu, *vals;
    int *cols, *rows;
    
    int *head = new int[3];  
    string fileName = "chunk_" + std::to_string(rank)+".txt";
    readHead(fileName, head);
    int N = head[0];
    int nPoints = head[2];
    //printf("rank: %d -- N=%d\n",rank,N);
    
    //printf("rank %d allocating memory...\n",rank);
    cudaMallocManaged(&vector_gpu, N*sizeof(double));
    cudaMallocManaged(&resVector_gpu, N*sizeof(double));

    cudaMallocManaged(&vals, nPoints*sizeof(double));
    cudaMallocManaged(&rows, nPoints*sizeof(int));
    cudaMallocManaged(&cols, nPoints*sizeof(int));
    //printf("rank %d memory allocation finished...\n",rank);
    
    //printf("rank %d reading chunk_%d\n",rank,rank);
    readMatrix(cols,rows,vals,fileName);
    printf("rank %d data written %d\n",rank,nPoints);
    //printf("rank %d finished reading chunk_%d\n",rank,rank);
    /*
    for(int i = 0; i<5; i++){
	printf("rank: %d -- row: %d -- col: %d  -- val: %f\n",rank,rows[i],cols[i],vals[i]);
    }*/
    	
    //printf("rank %d copy data to GPU.....\n",rank);
    for(int i = 0; i<N; i++){
		vector_gpu[i] = vector[i];
		resVector_gpu[i] = resVector[i];
    }
    //printf("rank %d vec[0] = %f\n",rank,vector_gpu[0]);


    int blockSize = prop.maxThreadsDim[2];
    int numBlocks = min(prop.multiProcessorCount, (nPoints + blockSize - 1)/ blockSize);	
    cout<<"blockSize: "<<blockSize<<"\n";
    cout<<"numBlocks: "<<numBlocks<<"\n";
    //printf("rank %d starting computations\n",rank);	
    multMatrixVector<<< 2,1 >>> (nPoints,rows,cols,vals,vector_gpu, resVector_gpu);
    //printf("rank %d finished computations\n",rank);
    cudaDeviceSynchronize();
    //printf("rank %d copy data to CPU.....\n",rank);
    //printf("rank %d res[%d] = %f\n",rank,rank, resVector_gpu[rank]);
    for(int i = 0; i<N; i++){
		vector[i] = vector_gpu[i];
		resVector[i] = resVector_gpu[i];
	}
	cudaFree(vector_gpu);
	cudaFree(resVector_gpu);
}