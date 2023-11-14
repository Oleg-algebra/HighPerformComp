//
// Created by User on 13.11.2023.
//
#include <iostream>

using namespace std;
int main(){
    int (*array)[4] = new int [3][4];
    array[0][0] = 98;
    array[0][2] = 2;
    array[1][2] = 5;

    int n = 9;
    int* pn = &n;
    cout<<&n<<"\n";
    cout<<&pn<<"\n";

    int *m = new int[3];
    m[0] = 30;
    m[1] = 45;
    m[2] = 50;
    cout<< m<<"\n";
    cout<< *&m<< "\n";
    return 0;
}