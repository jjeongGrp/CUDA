#include <cuda_runtime.h>
#include <stdio.h>

__global__ void Kernel1(float *c)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	float a, b;
	a=b=0.0f;
	if(tid%2==0){
		a=1.0f;
		b=2.0f;
	}else{
		a=2.0f;
		b=1.0f;
	}
	c[tid] = a+b;
}
__global__ void Kernel2(float *c)
{
	int tid=blockIdx.x*blockDim.x + threadIdx.x;
	float a, b;
	a=b=0.0f;
	if((tid/warpSize)%2==0){
		a=1.0f;
		b=2.0f;
	}else{
		a=2.0f;
		b=1.0f;
	}
	c[tid]=a+b;
}
int main()
{
	int size=64;
	int blocksize=64;
	float *d_C;
	cudaMalloc((float**)&d_C, size*sizeof(float));
	dim3 block(blocksize);
	dim3 grid((size+block.x-1)/block.x);
	printf("Execution configure (block %d grid %d)\n",block.x, grid.x);
	Kernel1<<<grid,block>>>(d_C);

	cudaDeviceSynchronize();

	Kernel2<<<grid,block>>>(d_C);
	cudaDeviceSynchronize();

	cudaFree(d_C);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
