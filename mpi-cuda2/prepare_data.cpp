#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;
using namespace std::chrono;

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
    auto start = high_resolution_clock::now();
    string  path;
    path = "matrices/sparsine.mtx";
    //path = "matrices/TF18.mtx";
    //path  = "matrices/rail_79841.mtx";
    //path = "matrices/m_t1.mtx";
    
    cout<<"File: "<<path<<"\n";

    int *head = new int[3];
    cout<<"reading head\n";
    readHead(path,head);
    cout<<"Head obtained\n";
    int nPoints = head[2];
    int N = head[0];
    int M = head[1]; 
    cout<<"rows: "<<N<<"\n";
    cout<<"columns: "<<M<<"\n";
    cout<<"data points number: "<<(int)nPoints<<"\n";
	
    int *rows = new int[nPoints];
    int *cols = new int[nPoints];
    double *vals = new double[nPoints];
	
	cout<<"reading data from file\n";
    readMatrix(cols,rows,vals,path);
    
    cout<<"Data reading Finished\n";
    cout<<"creating chunks...\n";
    int nproces = 8;
    for(int i = 0; i < nproces; i++){
	fstream outFile;
	string fileName = "chunk_" + std::to_string(i)+".txt";
        //fileName = path;
	outFile.open(fileName,ios::out);
	int dataNumber = 0;
        for(int j = i; j<nPoints; j+=nproces){
            dataNumber ++;
	}

	outFile<<N<<" "<<M<<" "<<dataNumber<<"\n";
        for(int j = i; j<nPoints; j+=nproces){
	//	cout<<rows[j]<<" "<<cols[j]<<" "<<vals[j]<<"\n";
		outFile<<rows[j]<<" "<<cols[j]<<" "<<vals[j]<<"\n";
	}
	outFile.close();
    }
    cout<<"chunks created\n"; 

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    
    cout <<" data preparation time: "
         << duration.count() << " milliseconds" << endl;
   
	
    
    delete [] rows;
    delete [] cols;
    delete [] vals; 
    delete [] head;
    return 0;
}
