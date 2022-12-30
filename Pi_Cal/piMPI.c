#include <stdio.h>
#include <math.h>
#include "mpi.h"
int main(int argc, char *argv[])
{
	int i, myrank, nprocs;
	const long num_step = 50000000;
	double mypi, x, pi, h, sum;
	double st, et;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	h=1.0/(double)num_step;
	sum = 0.0;
	st = MPI_Wtime();
	for(i=myrank;i<num_step;i+=nprocs)
	{
		x = h*((double)i-0.5);
		sum += 4.0/(1.0+x*x);
	}
	mypi= h*sum;
	MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	et=MPI_Wtime();
	if(myrank==0){
		printf("PI= %.15f (Error = %e)\n",pi, fabs(acos(-1)-pi));
		printf("Elapsed Time = %f, [sec]\n", et-st);
		printf("----------------------------------------\n");
	}
	MPI_Finalize();
	return 0;
}
