#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
inline double cpuTimer()
{
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}
int main()
{
	double iStart, ElapsedTime;
	const long num_step = 50000000;
	long i;
	double sum, step, pi, x;
	step = (1.0/(double)num_step);
	sum = 0.0;
	iStart=cpuTimer();
	printf("-------------------------------------\n");
	omp_set_num_threads(4);
#pragma omp parallel for reduction(+:sum), private(x)
	for(i=1;i<=num_step;i++)
	{
		x = ((double)i-0.5)*step;
		sum += 4.0/(1.0+x*x);
	}
	pi = step*sum;
	ElapsedTime= cpuTimer() - iStart;
	printf("PI= %.15f (Error = %e)\n",pi, fabs(acos(-1)-pi));
	printf("Elapsed Time = %f, [sec]\n", ElapsedTime);
	printf("----------------------------------------\n");
	return 0;
}
