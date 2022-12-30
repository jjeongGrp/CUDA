#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

inline void CHECK(const cudaError_t error)
{
	if(error !=cudaSuccess)
	{
		fprintf(stderr, "Error: %s:%d, ",__FILE__,__LINE__);
		fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}
double cpuTimer()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *arr, const int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for(int i=0;i<size;i++)
		arr[i]= (float)(rand())/RAND_MAX;
}
void MatMulOnCPU(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
	float sum;
	for(int i=0;i<Arows;i++)
	{
		for(int j=0;j<Bcols;j++)
		{
			sum = 0.0f;
			for(int k=0;k<Acols;k++)
			{
				sum += A[i*Acols+k]*B[k*Bcols+j];
			}
			C[i*Bcols+j]=sum;
		}
	}
}
__global__ void MatMultOnGPU(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
	/***********************************/
	// TODO //
	/***********************************/
}

void checkResult(float *host, float *gpu, const int N)
{
	double epsilon = 1.0e-8;
	bool match = 1;
	for(int i=0;i<N;i++)
	{
		if(abs(host[i]-gpu[i])>epsilon)
		{
			match = 0;
			printf("Matrices do not match!\n");
			printf("host %10.7f, gpu %10.7f at current %d\n", host[i], gpu[i], i);
			break;
		}
	}
	if(match)printf("Matrices match.\n");
}
int main(int argc, char **argv)
{
	double Start, ElapsedTime;
	float ETime;	
	float *MatA, *MatB, *MatC, *gpu_MatC;
	int Arows=300, Acols=200, Bcols=400;
	int threads_x=32, threads_y=32;
	if(argc>1) Arows=atoi(argv[1]);
	if(argc>2) Acols=atoi(argv[2]);
	if(argc>3) Bcols=atoi(argv[3]);
	if(argc>4) threads_x = atoi(argv[4]);
	if(argc>5) threads_y = atoi(argv[5]);
	/************ ON CPU **************/
	MatA=(float*)malloc(Arows*Acols*sizeof(float));
	MatB=(float*)malloc(Acols*Bcols*sizeof(float));
	MatC=(float*)malloc(Arows*Bcols*sizeof(float));
	gpu_MatC=(float*)malloc(Arows*Bcols*sizeof(float));
	
	initialData(MatA, Arows*Acols);
	initialData(MatB, Acols*Bcols);
	
	Start=cpuTimer();
	MatMulOnCPU(MatA, MatB, MatC, Arows, Acols, Bcols);
	ElapsedTime=cpuTimer()-Start;
	printf("Elapsed Time on CPU : %f sec\n",ElapsedTime);
	/**********************************/

	/************ ON GPU **************/
	float *d_MatA, *d_MatB, *d_MatC;
	/* Allocate d_MatA, d_MatB, d_MatC on Device */
    // TODO //
	/*********************************************/
	// create two events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    /* Data Upload (d_MatA, d_MatB )  */
    // TODO //
    /**********************************/
	dim3 block(threads_x,threads_y,1);
	dim3 grid((Bcols+block.x-1)/block.x, (Arows+block.y-1)/block.y, 1);
	MatMultOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Arows, Acols, Bcols);
    /* Data Download (d_MatC)         */
    // TODO //
    /**********************************/
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ETime, start, stop);
	printf("Elapsed Time on GPU : %f sec\n",ETime*1e-3);
	/**********************************/
	checkResult(MatC, gpu_MatC, Arows*Bcols);

	free(MatA),	free(MatB),	free(MatC),	free(gpu_MatC);
    /* Deallocate Device Memory  */
    // TODO //
    /*****************************/

	CHECK(cudaDeviceReset());
	return 0;
}
