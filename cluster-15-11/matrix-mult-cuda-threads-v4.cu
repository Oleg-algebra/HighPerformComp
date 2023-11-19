#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <math.h>

using namespace std;
__global__ 
void multMatrixVector(int N,double ***M, double *v, double *resVector,int *colsN){
    //TODO: rewrite function

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

void start(int numBlocks, int blockSize, string path){
    
    double *v, *res, *vals, ***M;
    int *rows, *cols, *colsN;
    cout<<"memory allocation starting..."<<"\n";
    int *head = new int[3];
    cout<<"reading head\n";
    readHead(path,head);
    cout<<"Head obtained\n";
    int nPoints = head[2];
    int N = head[0];
    cout<<"data points number: "<<(int)nPoints<<"\n";
	
    cudaMallocManaged(&cols, nPoints*sizeof(int));
    cudaMallocManaged(&colsN, N*sizeof(int));
    cudaMallocManaged(&rows, nPoints*sizeof(int));
    cudaMallocManaged(&vals, nPoints*sizeof(double));
    cudaMallocManaged(&v, N*sizeof(double));
    cudaMallocManaged(&res, N*sizeof(double)); 
    cudaMallocManaged(&M, N*sizeof(double**));
    cout<<"memory allocated"<<"\n";

    cout<<"filling v and res with zeros\n";
	
    for(int i = 0; i<N; i++){
	v[i] = 0.0;
	res[i] = 0.0;
        colsN[i] = 0;
    }
    v[0] = 1.0;

    cout<<"total size memory: "<<(2*nPoints*sizeof(int)+nPoints*sizeof(double)+2*N*sizeof(double))/1024<<" KB\n";
    
    cout<<"reading data from file\n";
    readMatrix(cols,rows,vals,colsN,path);
    cout<<"Data reading Finished\n";


    int *colsNcopy = new int[N];
    for(int i = 0; i<N; i++){
        colsNcopy[i] =colsN[i]; 
        
    }
	
	cout<<"matrix memory allocation starting...\n";
	for(int i = 0; i<N; i++){
		if(colsN[i] > 0){
		    cudaMallocManaged(&(M[i]),colsN[i]*sizeof(double*));
			for(int j = 0; j<colsN[i]; j++){
                cudaMallocManaged(&(M[i][j]),2*sizeof(double));
			}
		}else{
		    cudaMallocManaged(&(M[i]),1*sizeof(double*));
		    cudaMallocManaged(&(M[i][0]),2*sizeof(double));
		    cout<<"empty row\n";	
		}
	}
	cout<<"matrix memory allocation finished\n";
	
	cout<<"matrix filling starting...\n";
	for(int i = 0; i< nPoints; i++){
	    int row = rows[i];
        int col = cols[i];
        double value = vals[i];
        int colInd = colsNcopy[row] - 1;
        M[row][colInd][0] = (double) col;
        M[row][colInd][1] = value;
        colsNcopy[row] = colInd;		
	}
	
	cout<<"matrix filling finished\n";    
    

    cout<<"BlockSize: "<<blockSize<<"\n";
    cout<<"numBlocks: "<<numBlocks<<"\n";

    cout<<"Starting computation\n";
    multMatrixVector<<<numBlocks,blockSize>>>(N,M,v,res,colsN);
    cout<<"Computation finished\n";
    cudaDeviceSynchronize();

    double maxError = 0.0;
    
    int ind = 0;
    for (int i = 0; i < N; i++){
        if(cols[ind] == 0 && rows[ind] == i){	
            maxError = fmax(maxError, fabs(res[i]-vals[ind]));
            ind++;
        }
    }

    cout << "Max error: " << maxError << "\n";

    double eps = 1e-10;    
    for(int i = 0; i<N;i++){
        if(fabs(res[i] - 0.0)>eps){
            printf("res[%d] = %f\n",i,res[i]);
        }
    }
    
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

    if(true){
        fstream resFile;
        resFile.open("corr_res.txt",ios::out);
        resFile<<N<<"\n";
        for(int i = 0; i<N; i++){
	    resFile<<res[i]<<"\n";
	}
        resFile.close();   
    }

    
    delete [] head;
    delete [] colsNcopy;
    cudaFree(v);
    cudaFree(res);
    cudaFree(vals);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(colsN);
    cudaFree(M);
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
    //path = "matrices/sparsine/sparsine2.mtx";
    //path = "matrices/newSparsine2.txt";
    //path = "matrices/test-matrix.txt";
    path = "matrices/sparsine.mtx";

    int *head = new int[3];
    cout<<"reading head\n";
    readHead(path,head);
    cout<<"Head obtained\n";
    int nPoints = head[2];
    
    //printGPUInfo();    

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);
    //printDevProp(prop);
    cout << "Multiprocessor Count: " << prop.multiProcessorCount << endl;
    cout << "Thread Count: " << prop.maxThreadsDim[2] << endl;

    for(int bs = 32; bs<=32; bs+=1){	
        int blockSize = prop.maxThreadsDim[2];
        int numBlocks = min(prop.multiProcessorCount, (nPoints + blockSize - 1)/ blockSize);
    
        start(numBlocks,blockSize,path);
    }
    delete [] head;
    return 0;
}
