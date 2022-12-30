#include <stdio.h>
#include <math.h>
#include <sys/time.h>
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
	step = (1.0/(double)num_step);	// Del_x
	sum = 0.0;
	iStart=cpuTimer();
	printf("-------------------------------------\n");
#pragma acc kenels
	for(i=1;i<=num_step;i++)
	{
		x = ((double)i-0.5)*step;	// x_k
		sum += 4.0/(1.0+x*x);		// f(x_k)
	}
    pi = step*sum;  // sum{f(x_k)}*Del_x
	ElapsedTime=cpuTimer() - iStart;

	printf("PI= %.15f (Error = %e)\n",pi, fabs(acos(-1)-pi));
	printf("Elapsed Time = %f, [sec]\n", ElapsedTime);
	printf("----------------------------------------\n");
	return 0;
}
