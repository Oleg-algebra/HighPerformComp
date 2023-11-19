/* multiply.cu */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;
using namespace std::chrono;


__global__ 
void multMatrixVector(int N,int* rows, int*cols ,double *vals, double *v, double *resVector,int *rowDataNumber){
    //TODO: rewrite function

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<N;i+=stride){
	int start;
	if( i == 0){
	    start = 0;
	}else{
	    start = rowDataNumber[i-1];
	}
	for(int j = start; j<rowDataNumber[i];j++){
	    double value = vals[j] * v[cols[j]];
	    resVector[rows[j]] += value;
	}    
        
	        
	if(cols[i] == 0){
            //printf("res[1] = %f\n",resVector[1]);
            //printf("block: %d -- thread: %d -- row index: %d\n",blockIdx.x,threadIdx.x,rows[i]);
            //printf("value written to vector: %f  -- resVector[%d] = %f\n",value,rows[i],resVector[rows[i]]);
	    //printf("col: %d -- resV[%d] = %f -- v[%d] = %f -- value: %f\n",cols[i],rows[i],resVector[rows[i]],cols[i], v[cols[i]],value);
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

void readMatrix(int *cols,int *rows, double *vals, int *rowDataNumber,string& fileName){

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
            rowDataNumber[(int)values[0]] += 1;
            counter++;
            }
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
    cout<<"data written: "<<counter<<"\n";
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

    double *vector_gpu, *resVector_gpu, *vals, *validationVec;
    int *cols, *rows,*rowDataNumber;
    
    int *head = new int[3];  
    string fileName = "chunk_" + std::to_string(rank)+".txt";
    readHead(fileName, head);
    int N = head[0];
    int nPoints = head[2];
    //printf("rank: %d -- N=%d\n",rank,N);
    
    printf("rank %d allocating memory...\n",rank);
    cudaMallocManaged(&rowDataNumber, N*sizeof(int));
    cudaMallocManaged(&vector_gpu, N*sizeof(double));
    cudaMallocManaged(&resVector_gpu, N*sizeof(double));
    cudaMallocManaged(&validationVec, N*sizeof(double));

    cudaMallocManaged(&vals, nPoints*sizeof(double));
    cudaMallocManaged(&rows, nPoints*sizeof(int));
    cudaMallocManaged(&cols, nPoints*sizeof(int));
    printf("rank %d memory allocation finished...\n",rank);
    
    printf("rank %d reading chunk_%d\n",rank,rank);
    readMatrix(cols,rows,vals,rowDataNumber,fileName);
    printf("rank %d data written %d\n",rank,nPoints);
    printf("rank %d finished reading chunk_%d\n",rank,rank);
    /*
    for(int i = 0; i<5; i++){
	printf("rank: %d -- row: %d -- col: %d  -- val: %f\n",rank,rows[i],cols[i],vals[i]);
    }*/

    for(int i = 0; i<10; i++){
        printf("rank: %d before comput rowDataNumber[%d]: %d\n",rank,i,rowDataNumber[i]);
    }

	
    int nonZero = 0;
    for(int i = 0; i<N; i++){
	    if(nonZero == 0){
	        nonZero = rowDataNumber[i];
	        continue;
	    }else{
	       rowDataNumber[i] += nonZero;
	       nonZero = rowDataNumber[i];
	    }
    }

    for(int i = 0; i<10; i++){
        printf("rank: %d after comput rowDataNumber[%d]: %d\n",rank,i,rowDataNumber[i]);
    }
    	
    printf("rank %d copy data to GPU.....\n",rank);
    for(int i = 0; i<N; i++){
		vector_gpu[i] = vector[i];
		resVector_gpu[i] = 0.0;
                validationVec[i] = 0.0;
    }
    //printf("rank %d vec[0] = %f\n",rank,vector_gpu[0]);


    int blockSize = prop.maxThreadsDim[2];
    int numBlocks = min(prop.multiProcessorCount, (N + blockSize - 1)/ blockSize);	
    //cout<<"blockSize: "<<blockSize<<"\n";
    //cout<<"numBlocks: "<<numBlocks<<"\n";
    printf("rank %d starting computations\n",rank);	
    multMatrixVector<<< blockSize,numBlocks >>> (N,rows,cols,vals,vector_gpu, resVector_gpu,rowDataNumber);
    multMatrixVector<<< 1,1 >>>(N,rows,cols,vals,vector_gpu, validationVec,rowDataNumber);
    printf("rank %d finished computations\n",rank);
    cudaDeviceSynchronize();
    printf("rank %d copy data to CPU.....\n",rank);
    //printf("rank %d res[%d] = %f\n",rank,rank, resVector_gpu[rank]);
    for(int i = 0; i<N; i++){
		vector[i] = vector_gpu[i];
		resVector[i] = resVector_gpu[i];
    }

    double maxError = 0.0;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(resVector_gpu[i]-validationVec[i]));
    }
    
    printf("rank %d -- Max error = %f\n",rank,maxError);

    double eps = 1e-13;
    for(int i = 0; i<N; i++){
        if(fabs(validationVec[i])>eps || fabs(resVector_gpu[i])>eps){
            printf("rank: %d validVec[%d] = %f\n",rank,i,validationVec[i]);
            printf("rank: %d resVec_gpu[%d] = %f\n",rank,i,resVector_gpu[i]);
        }
    }

    printf("rank %d free GPU memory\n",rank);
    cudaFree(validationVec);
    cudaFree(vals);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vector_gpu);
    cudaFree(resVector_gpu);
}