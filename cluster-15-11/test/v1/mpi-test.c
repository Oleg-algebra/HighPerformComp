/* main.c */
#include <mpi.h>
#include <stdio.h>

int readDim(int *row, int *col){
    printf("opening file\n");
    FILE* ptr = fopen("chunk_0.txt", "r");
    if (ptr == NULL) {
        printf("no such file.");
        return 0;
    }
 
    while (1){
        if(fscanf(ptr, "%d", row) == 1){
            printf("row: %d \n",*row);
	}
	if(fscanf(ptr, "%d", col) == 1){
            printf("col: %d \n",*col);
	}
	break;
        
    }
    printf("reading finished\n");
    fclose(ptr);	


}

void launch_multiply(int rank, double *vector, double *resVector);

int main (int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
	int row = 0;
	int col = 0;
    if(rank == 0){
	printf("rank: %d preparing data\n",rank);
	readDim(&row,&col);
	printf("row: %d, col: %d\n",row,col);
    }
    //printf("rank %d waiting...\n",rank);
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //printf("rank %d passed barrier\n",rank);
    //printf("rank %d row: %d col: %d\n",rank,row,col);

    double vector[col];
    double resVector[col];
    
    for(int i = 0; i<col; i++){
        vector[i] = 0.0;
        resVector[i] = 0.0;
    }
    vector[0] = 1.0;		

    //printf("rank: %d of %d\n",rank,nprocs);
     	 
    double *pVector = vector;
    double *pResVector = resVector;
    for(int i = 0; i<1;i++){    
         //printf("repeating");
         launch_multiply (rank, pVector, pResVector);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int tag = 1;
    MPI_Status status;
    if(rank != 0){
        /*
        printf("rank %d result\n",rank);
	for(int i = 0; i<10; i++){
	    printf("pResVector[%d] = %f\n",i,pResVector[i]);
        }*/
        MPI_Send(pResVector,col,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
	printf("message sent from rank %d to rank 0\n",rank);
    }else{
        double res[col];
        double *resP = res;

        for(int id  = 1; id < nprocs; id++){
            MPI_Recv(resP, col, MPI_DOUBLE, id, tag, MPI_COMM_WORLD, &status);
            printf("message recieved from rank %d \n",id);
            for(int i = 0; i<col; i++){
	        pResVector[i] += resP[i];
	    }
	}
        /*
	for(int i = 0; i<col; i++){
	    if(pResVector[i] != 0.0){
                 printf("resVector[%d] = %f\n",i,pResVector[i]);
	    }
        }*/

    }
    
    MPI_Finalize();
    return 0;
}
