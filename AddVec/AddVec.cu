#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
/*
#define CHECK(call)                                                    \
{                                                                      \
	const cudaError_t error = call;                                    \
	if (error != cudaSuccess)                                          \
	{																   \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
		fprintf(stderr, "code: %d, reason: %s\n", error,               \
		cudaGetErrorString(error));                                    \
		exit(1);                                                       \
	}                                                                  \
}
*/
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

void initialData(float *arr, int size)
{
	time_t t;
	srand((unsigned)time(&t));  // seed
	for(int i=0;i<size;i++)
		arr[i]=(float)(rand())/RAND_MAX;
}
void AddVecOnHost(float *A, float *B, float *C, const int size)
{
#pragma omp parallel for
	for(int i=0;i<size;i++)
		C[i] = A[i] + B[i];
}

__global__ void AddVecOnGPU(float *A, float *B, float *C, const int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<size) C[idx] = A[idx] + B[idx];
}

void checkResult(float *host, float *gpu, const int N)
{
	double epsilon = 1.0e-8;
	bool match = 1;
	for(int i=0;i<N;i++)
	{
		if(abs(host[i] - gpu[i]) > epsilon)
		{
			match = 0;
			printf("Vector do not match!\n");
			printf("host %5.2f, gpu %5.2f at current %d\n", host[i], gpu[i], i);
			break;
		}
	}
	if(match) printf("Vectors match.\n");
}
int main(int argc, char **argv)
{
//	int nSize = 1<<23;   //16M
	int nSize = 1<<24;   //16M
	printf("Vector size : %d\n", nSize);
/*********** on HOST *******************/
	// malloc host memory
	size_t nBytes = nSize*sizeof(float);

	float *h_A, *h_B, *hostResult, *gpuResult;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostResult = (float*)malloc(nBytes);
	gpuResult = (float*)malloc(nBytes);

	double iStart, iEnd;
	double ElapsedTime;

	initialData(h_A, nSize);
	initialData(h_B, nSize);

	memset(hostResult, 0, nBytes);
	memset(gpuResult, 0, nBytes);

	iStart=cpuTimer();
	AddVecOnHost(h_A, h_B, hostResult, nSize);
	iEnd = cpuTimer();
	ElapsedTime = iEnd - iStart;
	printf("Elapsed Time in AddVecOnHost : %f\n",ElapsedTime);
/*****************************************/

/********** ON GPU **********************/
	// malloc device global memory
	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((float**)&d_A, nBytes));
	CHECK(cudaMalloc((float**)&d_B, nBytes));
	CHECK(cudaMalloc((float**)&d_C, nBytes));

	// Data transfer : Host --> Device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	// dimension of thread block and grid
	dim3 block(256);
	dim3 grid((nSize+block.x-1)/block.x);

	// create tow events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	iStart = cpuTimer();
	float Etime;
	cudaEventRecord(start);
	AddVecOnGPU<<<grid, block>>>(d_A, d_B, d_C, nSize);
	CHECK(cudaDeviceSynchronize());
//	ElapsedTime = cpuTimer() - iStart;
	cudaEventRecord(stop);
	ElapsedTime = cpuTimer() - iStart;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&Etime, start, stop);
	printf("Elapsed Time in AddVecOnGPU<<<%d, %d>>> : %f ms\n", grid.x, block.x, Etime);
//	printf("GPU Timer : %f ms , CPU Timer : %f ms\n",Etime, ElapsedTime*1000.0);

	CHECK(cudaMemcpy(gpuResult, d_C, nBytes, cudaMemcpyDeviceToHost));
/****************************************/	

	// check results
	checkResult(hostResult, gpuResult, nSize);
	
	// memory deallocate
	free(h_A),	free(h_B),	free(hostResult), 	free(gpuResult);
	CHECK(cudaFree(d_A)),	CHECK(cudaFree(d_B)),	CHECK(cudaFree(d_C));
	return 0;
}
