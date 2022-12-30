#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "cublas_v2.h"
#define R2C(r, c, nrows) ((c)*(nrows)+(r))

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
void checkResult(float *host, float *gpu, const int N)
{
	double epsilon = 1.0e-8;
	bool match = 1;
	float host_tmp[100*100];
	int cnt=0;
	for(int i=0;i<100;i++)
	{
		for(int j=0;j<100;j++)
		{
			host_tmp[R2C(i,j,100)]=host[cnt];
			cnt++;
		}
	}
	for(int i=0;i<10000;i++)
		host[i]=host_tmp[i];

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
	float *MatA, *MatB;
	float *d_MatA, *d_MatB, *d_MatC;
	int Arows=300, Acols=200, Bcols=400;

	/*********** On CPU **************/
	 MatA=(float*)malloc(Arows*Acols*sizeof(float));
     MatB=(float*)malloc(Acols*Bcols*sizeof(float));

	 initialData(MatA, Arows*Acols);
     initialData(MatB, Acols*Bcols);


	 /******** On GPU **********/
  	cublasHandle_t handle = 0;
	cudaMalloc((void**)&d_MatA, sizeof(float)*Arows*Acols);
	cudaMalloc((void**)&d_MatB, sizeof(float)*Acols*Bcols);
	cudaMalloc((void**)&d_MatC, sizeof(float)*Arows*Bcols);
	cublasSetMatrix(Arows, Acols, sizeof(float), MatA, Arows, d_MatA,Acols);
	cublasSetMatrix(Acols, Bcols, sizeof(float), MatB, Acols, d_MatB,Bcols);

	float alpha=1.0f, beta=0.0f;
	Start=cpuTimer();
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,Arows,Bcols,Acols,&alpha, d_MatA, Arows, d_MatB, Acols, &beta, d_MatC, Arows);
	ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on CPU : %f\n",ElapsedTime);
	/**********************************/

	  free(MatA), free(MatB);
	  cudaFree(d_MatA), cudaFree(d_MatB), cudaFree(d_MatC);
	  return 0;

}





