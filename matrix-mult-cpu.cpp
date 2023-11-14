#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;

double pointWiseProd(vector<vector<double>> &row, double *v2){
    //TODO: rewrite function, replace arrays by vectors
    double res = 0.0;
    int n = row.size();
    for(int i = 0; i<n;i++){
        int ind = (int) row.at(i)[0];
        double value = row.at(i)[1];
        res += (value * v2[ind]);
    }
    return res;
}

void multMatrixVector(int n, vector<vector<vector<double>>> &M, double *v, double *resVector){
    //TODO: rewrite function, replace arrays by vectors
    for(int i =0; i<n;i++){
        vector<vector<double>> row = M[i];
        resVector[i] = pointWiseProd(row,v);
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
int main1() {
    string  path;
//    path = "matrices/sparsine/sparsine.mtx";
    path = "matrices/newSparsine.txt";
    vector<vector<vector<double>>> matrix;
    matrix = readMatrix(path);
    cout<<"matrix size: "<<matrix.size()<<"\n";


    auto *v = new double[matrix.size()];
    auto *res = new double[matrix.size()];

    for(int i = 0; i<matrix.size();i++){
        v[i] = 0.0;
        res[i] = 0.0;
    }
    v[0] = 1.0;
    cout << "Starting computation" << "\n";
    multMatrixVector(matrix.size(),matrix,v,res);
    cout << "Computation finished" << "\n";
    for(int i = 0; i<10;i++){
        cout << res[i] << " ";
    }
    cout<<"\n";
    return 0;
}
