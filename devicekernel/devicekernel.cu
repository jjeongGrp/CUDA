#include <stdio.h>
__global__ void helloFromHost();
__device__ int helloFromDevice(int tid);
int main()
{
	helloFromHost<<<1,5>>>();
	cudaDeviceReset();
	return 0;
}
__global__ void helloFromHost()
{
	int tid=threadIdx.x;
	printf("Hello world From __global__ kernel: %d\n",tid);
    int tid1=helloFromDevice(tid);
    printf("tid1 : %d\n",tid1);
}

__device__ int helloFromDevice(int tid)
{
	printf("Hello world Form __device__ kernel: %d\n",tid);
	return tid+1;
}
