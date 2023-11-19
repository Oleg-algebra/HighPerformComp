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
void multMatrixVector(int N,double ***M, double *v, double *resVector,int *colsN){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<N;i+=stride){
        for(int j = 0; j<colsN[i]; j++){
	    int col = (int) M[i][j][0];
	    double value = M[i][j][1];
	    resVector[i]+=(value * v[col]);
	}        

    }

}

__global__ 
void multMatrixVectorValid(int nPoints,int* rows, int*cols ,double *vals, double *v, double *resVector){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<nPoints;i+=stride){
        double value = vals[i] * v[cols[i]];
	resVector[rows[i]] += value;
	        
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

void readMatrix(int *cols,int *rows, double *vals, int *colsN,string& fileName){

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
            colsN[(int)values[0]] +=1;
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
/*
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %d\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %d\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %d\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i){
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i){
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    }
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %d\n",  devProp.totalConstMem);
    printf("Texture alignment:             %d\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));

}
*/

extern "C" void launch_multiply(int rank, double *vector, double *resVector, double *validVector)
{
    //printf("rank %d choosing GPU....\n",rank);
    cudaSetDevice(rank);
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    double *vector_gpu, *resVector_gpu, *vals, ***M, *valid_gpu;
    int *cols, *rows,*colsN;
    
    int *head = new int[3];  
    string fileName = "chunk_" + std::to_string(rank)+".txt";
    readHead(fileName, head);
    int N = head[0];
    int Ncols = head[1];
    int nPoints = head[2];

    //printf("rows: %d --- columns: %d\n",N,Ncols);
        
    auto start = high_resolution_clock::now();

    //printf("rank %d allocating memory...\n",rank);
    cudaMallocManaged(&vector_gpu, Ncols*sizeof(double));
    cudaMallocManaged(&resVector_gpu, Ncols*sizeof(double));
    cudaMallocManaged(&valid_gpu, Ncols*sizeof(double));
    cudaMallocManaged(&M, N*sizeof(double**));

    cudaMallocManaged(&vals, nPoints*sizeof(double));
    cudaMallocManaged(&rows, nPoints*sizeof(int));
    cudaMallocManaged(&cols, nPoints*sizeof(int));

    int memoryUsed = 0;
    memoryUsed += 3*Ncols*sizeof(double);
    memoryUsed += 2*nPoints*sizeof(int);
    memoryUsed += nPoints*sizeof(double);

    cudaMallocManaged(&colsN, N*sizeof(int));
    for(int i = 0; i<N; i++){
        colsN[i] = 0;
    }


    //printf("rank %d reading chunk_%d\n",rank,rank);
    readMatrix(cols,rows,vals,colsN,fileName);
    printf("rank %d data written %d\n",rank,nPoints);
    //printf("rank %d finished reading chunk_%d\n",rank,rank);

    for(int i = 0; i<N; i++){
	if(colsN[i] > 0){
            cudaMallocManaged(&(M[i]),colsN[i]*sizeof(double*));
            for(int j = 0; j<colsN[i]; j++){
                cudaMallocManaged(&(M[i][j]),2*sizeof(double));
                memoryUsed += 2*sizeof(double);
            }
	}else{
	    cudaMallocManaged(&(M[i]),1*sizeof(double*));
	    cudaMallocManaged(&(M[i][0]),2*sizeof(double));
            memoryUsed += 2*sizeof(double);
	    //cout<<"rank: "<<rank<<" empty row\n";	
	}
    }
    //printf("rank %d memory allocation finished...\n",rank);
    
    memoryUsed = memoryUsed / 1024;
    printf("rank: %d -- memory used: %d KB\n",rank,memoryUsed);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    cout << " rank: "<<rank<<" GPU memory allocation time: "
         << duration.count() << " milliseconds" << endl;

   
    int *colsNcopy = new int[N];
    for(int i = 0; i<N; i++){
        colsNcopy[i] =colsN[i]; 
        
    }
    //cout<<"colsNcopy created\n";
	
	//cout<<"matrix filling starting...\n";
	for(int i = 0; i< nPoints; i++){
	    int row = rows[i];
            int col = cols[i];
            double value = vals[i];
            int colInd = colsNcopy[row] - 1;
            M[row][colInd][0] = (double) col;
            M[row][colInd][1] = value;
            colsNcopy[row] = colInd;		
	}
	//cout<<"matrix filling finished\n";

    	
    //printf("rank %d copy data to GPU.....\n",rank);
    for(int i = 0; i<Ncols; i++){
		vector_gpu[i] = vector[i];
		resVector_gpu[i] = 0.0;
		valid_gpu[i] = 0.0;
    }


    int blockSize = prop.maxThreadsDim[2];
    int numBlocks = min(prop.multiProcessorCount, (nPoints + blockSize - 1)/ blockSize);	
    //cout<<"blockSize: "<<blockSize<<"\n";
    //cout<<"numBlocks: "<<numBlocks<<"\n";
    
    start = high_resolution_clock::now();
    //printf("rank %d starting computations\n",rank);	
    multMatrixVector<<<numBlocks,blockSize>>>(N,M,vector_gpu,resVector_gpu,colsN);
    //printf("rank %d finished computations\n",rank);
    cudaDeviceSynchronize();

    //printf("rank %d starting computations of validaiton vector\n",rank);	
    multMatrixVectorValid<<<1,1>>>(nPoints,rows, cols ,vals,vector_gpu,valid_gpu);
    //printf("rank %d finished computations of validaiton vector\n",rank);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);

    cout << " rank: "<<rank<<" GPU computation time: "
         << duration.count() << " milliseconds" << endl;
	
    //printf("rank %d copy data to CPU.....\n",rank);
    //printf("rank %d res[%d] = %f\n",rank,rank, resVector_gpu[rank]);
    for(int i = 0; i<Ncols; i++){
		vector[i] = vector_gpu[i];
		resVector[i] = resVector_gpu[i];
		validVector[i] = valid_gpu[i];
    }
	
    double maxError = 0.0;
    for (int i = 0; i < N; i++){
            maxError = fmax(maxError, fabs(resVector_gpu[i]-valid_gpu[i]));
    }
    
    printf("rank %d -- Max error = %f\n",rank,maxError);

    //printf("rank %d free GPU memory\n",rank);

    cudaFree(vals);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(valid_gpu);
    cudaFree(vector_gpu);
    cudaFree(resVector_gpu);
    cudaFree(M);
    cudaFree(colsN);
    delete [] colsNcopy;
}