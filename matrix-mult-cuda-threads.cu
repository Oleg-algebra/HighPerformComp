#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;
//__global__
//void pointWiseProd(vector<vector<double>> &row, double *v2, int i, double *resVector){
//    //TODO: rewrite function, replace arrays by vectors
//    double res = 0.0;
//    int n = row.size();
//    for(int i = 0; i<n;i++){
//        int ind = (int) row.at(i)[0];
//        double value = row.at(i)[1];
//        res += (value * v2[ind]);
//    }
//    resVector[i] = res;
//}
__global__
void multMatrixVector(int n, double ***M, double *v, double *resVector, int *colsN){
    //TODO: rewrite function, replace arrays by vectors
    // int index = threadIdx.x;
    // int stride = blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i =index; i<n;i+=stride){
        double res = 0.0;
        int m = colsN[i];
        for(int j = 0; j<m;j++){
            int ind = (int) M[i][j][0];
            res+= (M[i][j][1]*v[ind]);
        }
        resVector[i] = res;
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


vector<vector<vector<double>>> readMatrix(const string& fileName){

    fstream file;
    file.open(fileName,ios::in);
    vector<vector<vector<double>>> matrix;
//    int counterData = 0;
    if(file.is_open()){
        string text;
        vector<double> values;
        getline(file,text);
        values = getData(text,2);
        vector<vector<double>> row;
        long currentRow = 0;
        while(getline(file,text)){
//            counterData++;
            values = getData(text,3);
            vector<double> data(2,0);
            if(values[0] == currentRow){
//                printf("%f, %f, %f\n",values[0],values[1],values[2]);

                data[0] = values[1];
                data[1] = values[2];
                row.push_back(data);
            }else{
                if(values[0] - currentRow > 1){
                    cout<< "empty row"<<"\n";
                }
                matrix.push_back(row);
                row.clear();
                data[0] = values[1];
                data[1] = values[2];
                row.push_back(data);
                currentRow++;
            }
        }
        matrix.push_back(row);
        file.close();
    } else{
        cout << "file closed"<<"\n";
    }
//    cout<<"Data read: "<<counterData<<"\n";
    return matrix;
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
int main() {
    string  path;
//    path = "matrices/sparsine/sparsine.mtx";
    path = "/content/drive/MyDrive/HighPerformComput/3-cuda-matrix-mult/cmake-build-debug/matrices/newSparsine.txt";
    vector<vector<vector<double>>> matrix;
    matrix = readMatrix(path);
    cout<<"matrix size: "<<matrix.size()<<"\n";


    double *v, *res;
    int *colsN;
//    vector<vector<vector<double>>> *pMatrix;
    double ***M;
    cout<<"memory allocation starting..."<<"\n";
    int N = matrix.size();
    cudaMallocManaged(&M, N*sizeof(double**));
    cudaMallocManaged(&v, N*sizeof(double));
    cudaMallocManaged(&res, N*sizeof(double));
    cudaMallocManaged(&colsN,N*sizeof(int));

    for(int i = 0; i<matrix.size();i++){
        double cols = matrix[i].size();
        colsN[i] = (int)cols;
        cudaMallocManaged(&(M[i]),cols*sizeof(double*));
        for(int j = 0; j<cols; j++){
            cudaMallocManaged(&(M[i][j]),2*sizeof(double));
        }
    }

    cout<<"memory allocated"<<"\n";
    cout<<"vectors filling"<<"\n";
    for(int i = 0; i<matrix.size();i++){
        // cout<<"i: "<<i<<"\n";
        v[i] = 0.0;
        res[i] = 0.0;
    }
    v[0] = 1.0;
    cout<<"vectors finished"<<"\n";
	cout<<"Matrix filling"<<"\n";
	for(int i = 0; i<N;i++){
        // cout<<"i: "<<i<<"\n";
        
		for(int j = 0; j<colsN[i];j++){
            // cout<<"j: "<<j<<"\n";
            vector<double> data = matrix[i][j];
			M[i][j][0] = data[0];
            M[i][j][1] = data[1];
            // cout<<"<--"<<"\n";
		}
	}
	cout<<"Matrix filled"<<"\n";
	
	// for(int i = 0; i<10;i++){
	// 	double cols = matrix[i].size();
	// 	for(int j = 0;j<cols;j++){
    //         printf("row: %d --- col: %f --- value: %f\n",i,M[i][j][0],M[i][j][1]);
    //     }
	// }
   cout << "Matrix filled"<<"\n";
    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;
   cout << "Starting computation" << "\n";
   multMatrixVector<<<numBlocks,blockSize>>>(matrix.size(),M,v,res,colsN);
   cudaDeviceSynchronize();
   cout << "Computation finished" << "\n";
   for(int i = 0; i<10;i++){
       cout << res[i] << " ";
   }
    cout<<"\n";
    matrix.clear();
    cudaFree(M);
    cudaFree(v);
    cudaFree(res);
    return 0;
}
