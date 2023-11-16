#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;

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
	int N = head[0];
    cout<<"data points number: "<<(int)nPoints<<"\n";
	
    int *rows = new int[nPoints];
    int *cols = new int[nPoints];
    double *vals = new double[nPoints];
	
	cout<<"reading data from file\n";
    readMatrix(cols,rows,vals,path);
    
    cout<<"Data reading Finished\n";
    
    int nproces = 8;
    for(int i = 0; i < nproces; i++){
	fstream outFile;
	string fileName = "chunk_" + std::to_string(i)+".txt";
	outFile.open(fileName,ios::out);
	int dataNumber = (nPoints-i)/nproces;
	outFile<<N<<" "<<N<<" "<<dataNumber<<"\n";
    for(int j = i; j<nPoints; j+=nproces){
	//	cout<<cols[j]<<" "<<rows[j]<<" "<<vals[j]<<"\n";
		outFile<<cols[j]<<" "<<rows[j]<<" "<<vals[j]<<"\n";
	}
	outFile.close();
    }    
	
    
	delete [] rows;
	delete [] cols;
	delete [] vals; 
    delete [] head;
    return 0;
}
