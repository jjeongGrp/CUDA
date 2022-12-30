#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define N	(1024*1024*16)

__device__ int tmpC[N];

inline void CHECK(const cudaError_t error)
{
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
		fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}
double cpuTimer()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
void initialData(int *arr, int size)
{
	float tmp;
	time_t t;
	srand((unsigned)time(&t));  // seed
	for(int i=0;i<size;i++){
		tmp=(float)(10.0f*rand()/RAND_MAX);
		arr[i]=(int)(tmp);
	}
}
int VecProdOnCPU(int *A, int *B, int const size)
{
	int tmp=0;
	for(int i=0;i<size;i++)
		tmp += A[i]*B[i];
	return tmp;
}

__global__ void VecProdOnGPU(int *A, int *B, int *g_odata, int const size)
{
	/*******************************/
	// TODO //
	/*******************************/
}
int main(void)
{
	int cpu_result, gpu_result;

	// initialize
	int size = N;
	printf("vector length : %d\n", size);

	// execution configuration
	int blocksize=512;
	dim3 block(blocksize, 1);
	dim3 grid((size+block.x-1)/block.x,1);

	// allocate host memory
	size_t bytes=size*sizeof(int);
	int *h_A = (int*)malloc(bytes);
	int *h_B = (int*)malloc(bytes);
	int *tmp_A = (int*)malloc(bytes);
	int *tmp_B = (int*)malloc(bytes);
	int *h_AB=(int*)malloc(grid.x*sizeof(int));

	// allocate device memory
	int *d_A, *d_B, *d_AB;
	// TODO //
	/**************************/

	initialData(h_A, size);
	initialData(h_B, size);
	memcpy(tmp_A, h_A, bytes);
	memcpy(tmp_B, h_B, bytes);
	// Data Upload (h_a -> d_A, h_B -> d_B)
	// TODO //
	/*************************************/

	// cpu calculate
	double iStart=cpuTimer();
	cpu_result=VecProdOnCPU(tmp_A, tmp_B, size);
	double ElapsedTime=cpuTimer()-iStart;
	printf(" Result on CPU : %d, Elapsed Time %f sec\n", cpu_result,ElapsedTime);

	/********** GPU ***********/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	VecProdOnGPU<<<grid, block>>>(d_A,d_B,d_AB,size);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ETime;
	cudaEventElapsedTime(&ETime, start, stop);
	// Data Dowload (d_AB -> h_AB)
	// TODO //
	/***********************************/

	gpu_result=0;
	for(int i=0;i<grid.x;i++) gpu_result += h_AB[i];
	printf(" Result on GPU : %d, Elapsed Time %f sec\n", gpu_result,ETime*1e-3f);

	cudaEventDestroy(start),  cudaEventDestroy(stop);

	free(h_A), free(h_B), free(tmp_A), free(tmp_B), free(h_AB);
	cudaFree(d_A), cudaFree(d_B), cudaFree(d_AB);

	return 0;
		
}


