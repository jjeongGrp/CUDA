#include <stdio.h>
#include "mpi.h"
void printhello(int rank, int procs)
{
    printf("Hello World on CPU %d/%d\n",rank,procs);
}
int main(int argc, char **argv)
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    printhello(myrank, nprocs);
    MPI_Finalize();
    return 0;
}

