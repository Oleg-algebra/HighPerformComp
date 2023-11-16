/* multiply.cu */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;

__global__ 
void multiply(int N, float *a, float *b){
    a[0] = 1.0f;
    b[0]= 1.0f;
}



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