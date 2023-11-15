#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>
#include <math.h>

using namespace std;
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

void start(int numBlocks, int blockSize, string path){
    
    double *v, *res, *vals;
    int *rows, *cols;
    cout<<"memory allocation starting..."<<"\n";
    int *head = new int[3];
    cout<<"reading head\n";
    readHead(path,head);
    cout<<"Head obtained\n";
    int nPoints = head[2];
    int N = head[0];
    cout<<"data points number: "<<(int)nPoints<<"\n";
    cudaMallocManaged(&cols, nPoints*sizeof(int));
    cudaMallocManaged(&rows, nPoints*sizeof(int));
    cudaMallocManaged(&vals, nPoints*sizeof(double));
    cudaMallocManaged(&v, N*sizeof(double));
    cudaMallocManaged(&res, N*sizeof(double));  
    cout<<"memory allocated"<<"\n";
    
    
    cout<<"reading data from file\n";
    readMatrix(cols,rows,vals,path);
    cout<<"Data reading Finished\n";

    cout<<"filling v and res with zeros\n";
    for(int i = 0; i<N; i++){
	v[i] = 0.0;
	res[i] = 0.0;
    }
    v[0] = 1.0;
    
    /*
    cout<<"printing matrix\n";
    printMatrix(22,rows,cols,vals);
    cout<<"printing vector v\n";
    printVector(10,v);
    cout<<"printing vector res\n";
    printVector(10,res);
    

    int blockSize = 256;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;
    */
    cout<<"BlockSize: "<<blockSize<<"\n";
    cout<<"numBlocks: "<<numBlocks<<"\n";

    cout<<"Starting computation\n";
    multMatrixVector<<<numBlocks,blockSize>>>(nPoints,rows,cols,vals,v,res);
    cout<<"Computation finished\n";
    cudaDeviceSynchronize();
    //printVector(10,res);
    double maxError = 0.0;


    int ind = 0;
    for (int i = 0; i < N; i++){
        if(cols[ind] == 0 && rows[ind] == i){	
            maxError = fmax(maxError, fabs(res[i]-vals[ind]));
            ind++;
        }
    }
    double eps = 1e-10;
    cout << "Max error: " << maxError << "\n";
    /*
    for(int i = 0; i<N;i++){
        if(fabs(res[i] - 0.0)>eps){
            printf("res[%d] = %f\n",i,res[i]);
        }
    }
    */
    if(true){
        fstream outFile;
        outFile.open("log_file.txt",ios::app);
        outFile<<"blockSize: "<<blockSize<<"\n";
        outFile<<"numBlocks: "<<numBlocks<<"\n";
	outFile<<"numberOfThreads: "<<numBlocks*blockSize<<"\n";
	outFile<<"pointsNumber: "<<nPoints<<"\n";
        outFile<<"Max error: "<< maxError <<"\n";
        outFile<<"============================\n";
        outFile.close();   
    }
    
    delete [] head;
    cudaFree(v);
    cudaFree(res);
    cudaFree(vals);
    cudaFree(rows);
    cudaFree(cols);
    cudaDeviceSynchronize();
}

void printGPUInfo(){
    int deviceCount = 0;
    cudaError_t err = cudaSuccess;
    err = cudaGetDeviceCount(&deviceCount);
    if(err == cudaSuccess){
         cout<<"deviceCount: "<<deviceCount<<"\n";
    
   
         for(int id = 0; id < deviceCount;id++){
         size_t totalDevMem, freeDevMem;
         err = cudaSetDevice(id);
         if (err == cudaSuccess) {
                     
                  cudaMemGetInfo(&freeDevMem, &totalDevMem);
                  cout << " : ";
         	  cout << "Dev " << id << " (" << (freeDevMem/1024) << " KB of " << (totalDevMem/1048576) << " MB free)\n";

             }
        }
    }
}


int main() {
    string  path;
//    path = "matrices/sparsine/sparsine2.mtx";
    //  path = "matrices/newSparsine2.txt";
   // path = "matrices/test-matrix.txt";
    path = "matrices/sparsine.mtx";

    int *head = new int[3];
    cout<<"reading head\n";
    readHead(path,head);
    cout<<"Head obtained\n";
    int nPoints = head[2];
    
    printGPUInfo();    

    cudaSetDevice(0);
    for(int bs = 32; bs<=32; bs+=32){	
        int blockSize = 256;
        int numBlocks = (nPoints + blockSize - 1) / blockSize;
	//numBlocks = nPoints / blockSize;
    
        start(numBlocks,blockSize,path);
    }
    delete [] head;
    return 0;
}
