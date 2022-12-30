#include <stdio.h>
__host__ __device__ void Print()
{
	printf("Hello World\n");
}
__global__ void Wrapper()
{
	Print();
}
int main()
{
	Print();
	printf("==================\n");
	Wrapper<<<1,5>>>();
	cudaDeviceReset();
	return 0;
}
