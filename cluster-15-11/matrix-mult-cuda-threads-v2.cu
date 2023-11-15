#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <vector>

using namespace std;
__global__ 
void multMatrixVector(int nPoints, double **M, double *v, double *resVector, double **idVector,int *conter){
    //TODO: rewrite function

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("block: %d -- thread: %d -- start index: %d\n",blockIdx.x,threadIdx.x,index);
    for(int i =index; i<nPoints;i+=stride){
        //printf("block: %d -- thread: %d -- index: %d\n",blockIdx.x,threadIdx.x,i);
        int row = (int) M[i][0];
        int col = (int) M[i][1];
	conter[0] += 1;
        
        //idVector[row][0] =row ;
	
        double val = M[i][2];
	double res = (val * v[col]);
	if((int) col == 0){
	    idVector[row][0] = row;
	    idVector[row][1] = res;
	}
        resVector[row] += res;
    }

}

vector<double> getData(const string& dataString, int expectedDataLen){
    stringstream ss(dataString);
//    cout << "Data: "<< dataString << "\n";
    string singleData;
    vector<double> dataDouble(expectedDataLen,0);
    int i = 0;
    while (getline(ss,singleData,' ')) {

//        cout << "Data: "<< singleData << "\n";
        dataDouble[i] = std::stod(singleData);
        i++;
        // store token string in the vector

    }
    return dataDouble;
}

void readMatrix(double **M,const string& fileName){

    fstream file;
    file.open(fileName,ios::in);
    int counter = 0;
    if(file.is_open()){
        string text;
        vector<double> values;
        getline(file,text);
        values = getData(text,3);
        
        while(getline(file,text)){
            values = getData(text,3);
            M[counter][0] = values[0];
            M[counter][1] = values[1];
            M[counter][2] = values[2];
            counter++;
            }
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
    cout<<"data written: "<<counter<<"\n";
}

vector<double> readHead(const string &fileName){
    fstream file;
    file.open(fileName,ios::in);
//    int counterData = 0;
    vector<double> values;
    if(file.is_open()){
        string text;
        
        getline(file,text);
        values = getData(text,3);
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
    return values;
}

void printMatrix(vector<vector<vector<double>>> &matrix){
    for(int i = 0; i<5; i++){
        vector<vector<double>> row = matrix[i];
        for(auto data : row){
            if(true) {
                cout << "row: " << i << " col: " << data[0] << " value: " << data[1] << "\n";
            }
        }
    }
}

void checkMatrix(vector<vector<vector<double>>> &matrix){
    for(int i = 0; i<matrix.size(); i++){
        vector<vector<double>> row = matrix[i];
        for(auto data : row){
            if(data[0] != i) {
                printf("row: %d contains row: %f\n",i,data[0]);
                break;
            }
        }
    }
}

void printVector(int n, double*v){
    printf("printing vector\n");
    for(int i = 0;i<n;i++){
        printf("v[%d] = %f\n",i,v[i]);
    }
}
int main() {
    string  path;
//    path = "matrices/sparsine/sparsine2.mtx";
      path = "matrices/newSparsine2.txt";
   // path = "matrices/test-matrix.txt";
    int column = 0;
    double *v, *res;
    double **M;
    double** debugVector;
    cout<<"memory allocation starting..."<<"\n";
    int *conter;	    
    vector<double> parameters = readHead(path);
    double nPoints = parameters[2];
    int N = parameters[0];
    cout<<"data points number: "<<(int)nPoints<<"\n";
    cudaMallocManaged(&M, nPoints*sizeof(double*));
    cudaMallocManaged(&v, N*sizeof(double));
    cudaMallocManaged(&res, N*sizeof(double));
    cudaMallocManaged(&debugVector, N*sizeof(double*));
    cudaMallocManaged(&conter,1*sizeof(int));
    conter[0] = 0;    

    for(int i = 0; i<nPoints;i++){
        cudaMallocManaged(&(M[i]),3*sizeof(double));
    }
    for(int i = 0; i<N;i++){
    	cudaMallocManaged(&(debugVector[i]),2*sizeof(double));
    }

    for(int i = 0; i<N;i++){
    	debugVector[i][0] = 0.0;
	debugVector[i][1] = 0.0;
    }

    cout<<"memory allocated"<<"\n";
    cout<<"vectors filling"<<"\n";
    for(int i = 0; i<N;i++){
        // cout<<"i: "<<i<<"\n";
        v[i] = 0.0;
        res[i] = 0.0;
    }
    v[column] = 1.0;
    cout<<"vectors finished"<<"\n";
    cout<<"Matrix filling"<<"\n";
    readMatrix(M,path);
    cout<<"Matrix filled"<<"\n";
	
    //for(int i = 0; i<10;i++){
    //printf("row: %f -- col: %f -- value: %f\n",M[i][0],M[i][1],M[i][2]);       
    //}
    cout << "Matrix filled"<<"\n";
    int blockSize = 288;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;
    //numBlocks = nPoints/blockSize;
     
    cout<<"blockSize: "<<blockSize<<"\n";
    cout<<"numBlocks: "<<numBlocks<<"\n";
    cout << "Starting computation" << "\n";
    multMatrixVector<<<numBlocks,blockSize>>>(nPoints,M,v,res,debugVector,conter);
    cudaDeviceSynchronize();
  
    cout << "Computation finished" << "\n";
    double rowSum = 0.0;
    double resSum = 0.0;

    for(int i = 0; i<nPoints;i++){
        if(M[i][1]<column){
            continue;
        }else if(((int)M[i][1])==column){
            //printf("M[%f,%f] = %f\n",M[i][0],M[i][1],M[i][2]);
            int row = (int) M[i][0];
            //if(idVector[row] == 1){
            //    printf("row: %d used\n",row);
            //}
            rowSum+=M[i][2];
        }else{
            break;
        }
        
    }
    for(int i = 0; i<N ; i++){
        if(res[i] != 0.0){
            //printf("res[%d] = %f\n",i,res[i]);
           // resSum+=res[i];
        }
	resSum+=res[i];
    }
    //printf("res[2] = %f\n",res[2]);
    double checkSum = 0.0;
    for(int i = 0; i<N ; i++){
	if((int)debugVector[i][0] != 0 || i==0){	    
    	    //printf("res[%f] = %f\n",debugVector[i][0],debugVector[i][1]);
	    checkSum += debugVector[i][1];
	}
    }
    printf("checkSum: %f\n",checkSum);
    printf("rowSum: %f\n",rowSum);
    printf("resSum: %f\n",resSum);
    printf("counter: %d\n",conter[0]);
    double error = rowSum - resSum;
    if(error > 0){
        cout<<"Error: "<< error<<"\n";
    }else{
        cout<<"Error: "<< -1 * error<<"\n";
    }
    fstream outFile;
    outFile.open("log_file.txt",ios::app);
    outFile<<"blockSize: "<<blockSize<<"\n";
    outFile<<"numBlocks: "<<numBlocks<<"\n";
    outFile<<"Error: "<< error <<"\n";
    outFile<<"============================\n";
    outFile.close();    
    
    // matrix.clear();
    cudaFree(debugVector);
    cudaFree(conter);
    cudaFree(M);
    cudaFree(v);
    cudaFree(res);
    return 0;
}
