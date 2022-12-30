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
void MatMulOnCPU(float *A, float *B, float *C, const int nrows, const int ncols)
{
	float sum;
	for(int i=0;i<nrows;i++)
	{
		for(int j=0;j<nrows;j++)
		{
			sum = 0.0f;
			for(int k=0;k<ncols;k++)
			{
				sum += A[i*ncols+k]*B[k*nrows+j];
			}
			C[i*nrows+j]=sum;
		}
	}
}
__global__ void MatMultOnGPU(float *A, float *B, float *C, const int nrows, const int ncols)
{
	int tx = blockDim.x*blockIdx.x + threadIdx.x;	// col of C 
	int ty = blockDim.y*blockIdx.y + threadIdx.y;	// row of C 
	int tid = ty*nrows+tx;


	float sum=0.0f;
	if(tx < nrows && ty <nrows )
	{
		for(int i=0;i<ncols;i++)
		{
			sum += A[ty*ncols + i]*B[i*nrows+tx];
		}
		C[tid]=sum;
	}
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
	float *MatA, *MatB, *MatC, *gpu_MatC;
	int nrows=300, ncols=200;
	int threads_x=32, threads_y=32;
	int size;
	if(argc>1) nrows = atoi(argv[1]);
	if(argc>2) ncols = atoi(argv[2]);
	if(argc>3) threads_x = atoi(argv[3]);
	if(argc>4) threads_y = atoi(argv[4]);
	size = nrows*ncols;
	/************ ON CPU **************/
	MatA=(float*)malloc(size*sizeof(float));
	MatB=(float*)malloc(size*sizeof(float));
	MatC=(float*)malloc(nrows*nrows*sizeof(float));
	gpu_MatC=(float*)malloc(nrows*nrows*sizeof(float));
	
	initialData(MatA, size);
	initialData(MatB, size);
	
	Start=cpuTimer();
	MatMulOnCPU(MatA, MatB, MatC, nrows, ncols);
	ElapsedTime=cpuTimer()-Start;
	printf("Elapsed Time on CPU : %f\n",ElapsedTime);
	/**********************************/

	/************ ON GPU **************/
	float *d_MatA, *d_MatB, *d_MatC;
	CHECK(cudaMalloc((float**)&d_MatA, size*sizeof(float)));
	CHECK(cudaMalloc((float**)&d_MatB, size*sizeof(float)));
	CHECK(cudaMalloc((float**)&d_MatC, nrows*nrows*sizeof(float)));

	Start=cpuTimer();
	CHECK(cudaMemcpy(d_MatA,MatA, size*sizeof(float),cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_MatB,MatB, size*sizeof(float),cudaMemcpyHostToDevice));
	dim3 block(threads_x,threads_y,1);
	dim3 grid((nrows+block.x-1)/block.x, (nrows+block.y-1)/block.y, 1);
	MatMultOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nrows, ncols);

	CHECK(cudaMemcpy(gpu_MatC, d_MatC, nrows*nrows*sizeof(float), cudaMemcpyDeviceToHost));
	ElapsedTime=cpuTimer()-Start;
	printf("Elapsed Time on GPU : %f\n",ElapsedTime);
	/**********************************/
	checkResult(MatC, gpu_MatC, nrows*nrows);

	free(MatA),	free(MatB),	free(MatC),	free(gpu_MatC);
	CHECK(cudaFree(d_MatA)), CHECK(cudaFree(d_MatB)), CHECK(cudaFree(d_MatC));

	CHECK(cudaDeviceReset());
	return 0;
}
