/* main.c */
#include <mpi.h>
#include <stdio.h>

//void launch_multiply(int N, float *a, float *b);

int main (int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    if(rank == 0){
	printf("rank: %d preparing data\n",rank);
    }
    printf("rank %d waiting...\n",rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank %d passed barrier\n",rank);

    int N = 10;
    float a[N];
    float b[N];
    
    for(int i = 0; i<N; i++){
        a[i] = -1.0f;
        b[i] = -2.0f;
    }
	
    float *pA = a;
    float *pB = b;
	

    printf("rank: %d of %d\n",rank,nprocs);
    /*	
    launch_multiply (N,pA, pB);
    for(int i = 0; i < N; i++){
        printf("a[%d] = %f\n",i,a[i]);
        printf("b[%d] = %f\n",i,b[i]);
    }*/	
    MPI_Finalize();
    return 0;
}
