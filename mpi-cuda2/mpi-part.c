/* main.c */
#include <mpi.h>
#include <stdio.h>
#include <time.h>

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

void launch_multiply(int rank, double *vector, double *resVector, double *validVector);


int main (int argc, char **argv)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();

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
    double validVector[col];
    
    for(int i = 0; i<col; i++){
        vector[i] = 1.0;
        resVector[i] = 0.0;
	validVector[i] = 0.0;
    }
    vector[0] = 1.0;
    vector[1] = 1.0;		

    //printf("rank: %d of %d\n",rank,nprocs);
     	 
    double *pVector = vector;
    double *pResVector = resVector;
    double *pValidVector = validVector;
    for(int i = 0; i<1;i++){    
         //printf("repeating");
         launch_multiply (rank, pVector, pResVector, pValidVector);
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
	    MPI_Send(pValidVector,col,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
	//printf("message sent from rank %d to rank 0\n",rank);
    }else{
        double res[col];
        double *resP = res;
        double valid[col];
        double *pValid = valid;

        for(int id  = 1; id < nprocs; id++){
            MPI_Recv(resP, col, MPI_DOUBLE, id, tag, MPI_COMM_WORLD, &status);
	        MPI_Recv(pValid, col, MPI_DOUBLE, id, tag, MPI_COMM_WORLD, &status);
            //printf("message recieved from rank %d \n",id);
            for(int i = 0; i<col; i++){
	            pResVector[i] += resP[i];
                pValidVector[i] += pValid[i];
	    }
	}
        double eps = 1e-13;
        double maxError = 0.0;
	for(int i = 0; i<col; i++){
	    double error = pResVector[i] - pValidVector[i];
            if(error < -eps){
	        error = -error;
	    } 
            if(error - maxError > eps){
	        maxError = error;
	    }
	    
        }
        /*
        int count = 0;
	for(int i = 0; i<col; i++){
            if(pResVector[i] > eps || pResVector[i] < -eps){
                printf("resVec[%d] = %f  -- validVec[%d] = %f\n",i,pResVector[i],i,pValidVector[i]);
                count++;
            }
        }
        printf("nonZeros: %d\n",count);
        */
	printf("Max error: %f\n",maxError);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Total execution time: %f seconds\n", cpu_time_used);
    }
    
    MPI_Finalize();
    return 0;
}
