#include <cuda_runtime.h>
#include <stdio.h>
int main(void)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device : \"%s\"\n", deviceProp.name);

	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("driverVersion : %d\n", driverVersion);
	printf("runtimeVersion : %d\n", runtimeVersion);
	printf("\tCUDA Driver Version / Runtime Version  %d.%d / %d.%d\n",
			driverVersion/1000, (driverVersion%100)/10,
			runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("\tCUDA Capability Major/Minor version number : %d.%d\n",
			deviceProp.major, deviceProp.minor);
	printf("\tTotal amount of global memory : %.2f GBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
			(unsigned long long) deviceProp.totalGlobalMem);
	printf("\tGPU Clock rate :\t%.0f MHz(%0.2f GHz)\n",
			deviceProp.clockRate*1e-3f, deviceProp.clockRate*1e-6f);
	printf("\tMemory Clock rate :\t%.0f Mhz\n", deviceProp.memoryClockRate*1e-3f);
	printf("\tMemory Bus Width :\t%d-bit\n", deviceProp.memoryBusWidth);
	if(deviceProp.l2CacheSize)
		printf("\tL2 Cache Size:\t%d bytes\n",deviceProp.l2CacheSize);
	printf("\tTotal amount of constant memory:\t%lu bytes\n",deviceProp.totalConstMem);
	printf("\tTotal amount of shared memory per block:\t%lu bytes\n",deviceProp.sharedMemPerBlock);
	printf("\tTotal number of registers available per block:\t%d\n",deviceProp.regsPerBlock);
	printf("\tWarp Size:\t%d\n",deviceProp.warpSize);
	printf("\tMaximum number of threads per multiprocessor:\t%d\n",deviceProp.maxThreadsPerMultiProcessor);
	printf("\tMaximum number of thread per block:\t%d\n",deviceProp.maxThreadsPerBlock);
	printf("\tMaximum sizes of each dimension of a block:\t%d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("\tMaximum sizes of each dimension of a grid:\t%d x %d x %d\n",
			deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
	exit(EXIT_SUCCESS);


	return 0;
	
}
