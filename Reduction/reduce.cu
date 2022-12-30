#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

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

__global__ void reduceInterleaved(float *g_idata, float *g_odata, unsigned int n)
{
	unsigned int tid=threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

	// convert global data pointer to the local pointer of this block
	float *idata = g_idata + blockIdx.x*blockDim.x;

	// boundary check
	if(idx>=n) return;

	// in-place reduction in global memory
	for(int stride=blockDim.x/2;stride>0;stride>>=1)
	{
		if(tid<stride)
			idata[tid]+= idata[tid+stride];
		__syncthreads();
	}
	// write result for this block to global mem
	if(tid==0) g_odata[blockIdx.x]=idata[0];
}

__global__ void reduceNeighbored(float *g_idata, float *g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

	// convert global data pointer to the local pointer of this block
	float *idata = g_idata + blockIdx.x*blockDim.x;

	// boundary check
	if(idx>=n) return;

	// in-place reduction in global memory
	for(int stride=1;stride<blockDim.x; stride *=2)
	{
		if((tid%(2*stride))==0)
			idata[tid] += idata[tid+stride];
		// synchrnoize within block
		__syncthreads();
	}
	// write result for this block to global mem
	if(tid==0) g_odata[blockIdx.x] = idata[0];
}

float recursiveReduce(float *data, int const size)
{
	// terminate check
	if(size==1) return data[0];

	// renew the stride
	int const stride = size / 2;

	// in-place reduction
	for(int i=0;i<stride;i++)
		data[i] += data[i+stride];
	// call recursively
	return recursiveReduce(data, stride);
}

int main(void)
{
	bool bResult = false;
	float cpu_sum, gpu_sum;
	
	// initialize
	int size = 1<<24;	// 16M
	printf("array size %d\n", size);

	// execution configuration
	int blocksize = 512;
	dim3 block(blocksize,1);
	dim3 grid((size+block.x-1)/block.x,1);

	// allocate host memory
	size_t bytes = size*sizeof(float);
	float *h_idata = (float *)malloc(bytes);
	float *h_odata = (float *)malloc(grid.x*sizeof(float));

	// allocate device memory
    float *d_idata, *d_odata;
    cudaMalloc((float**)&d_idata, bytes);
    cudaMalloc((float**)&d_odata, grid.x*sizeof(float));

	initialData(h_idata, size);
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

	// cpu reduction
	double iStart = cpuTimer();
	cpu_sum = recursiveReduce(h_idata, size);
	double ElapsedTime = cpuTimer()-iStart;
	printf("CPU reduction : %f, Elapsed Time : %f sec\n", cpu_sum, ElapsedTime);

	/********** GPU **************/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
//	reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
	reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_odata, d_odata, grid.x*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ETime;
	cudaEventElapsedTime(&ETime, start, stop);
	printf("gpu reduction : %f, Elapsed Time : %f sec\n", gpu_sum, ETime*1e-3f);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;

}


