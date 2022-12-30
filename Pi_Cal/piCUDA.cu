#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
//#define NUM_STEP 500000000
#define NUM_STEP 5000000

__device__ double tmpC[NUM_STEP];

__global__ void PiCal(double *g_odata)
{
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        double step = (1.0/(double)NUM_STEP);

        double x =((double)idx +0.5)*step;
        tmpC[idx] = 4.0/(1.0+x*x);
        __syncthreads();

        double *idata = tmpC + blockIdx.x * blockDim.x;

        // boundary check
        if(idx >= NUM_STEP) return;

        // in-place reduction in global mem
        for(int stride = blockDim.x/2; stride >0; stride >>= 1)
        {
          if(tid<stride)
            idata[tid] += idata[tid+stride];
          __syncthreads();
        }
        // write result for this block to global mem
        if(tid==0) g_odata[blockIdx.x] = idata[0];
	/*************************************/
	// TODO //
	/*************************************/
}

int main()
{
	const long num_step=NUM_STEP;
    double step;
	step=(1.0/(double)num_step);
	int blocksize=512;
	dim3 block(blocksize,1);
	dim3 grid((num_step+block.x-1)/block.x,1);
	double *d_tmp, *h_tmp;

	h_tmp=(double*)malloc(grid.x*sizeof(double));
        
        cudaMalloc((void**)&d_tmp, grid.x*sizeof(double));
	// Allocate Device memory 
	// TODO //
	/********************************/

	PiCal<<<grid, block>>>(d_tmp);
	cudaDeviceSynchronize();
        cudaMemcpy(h_tmp, d_tmp, grid.x*sizeof(double), cudaMemcpyDeviceToHost);
	/* Data Download (d_tmp -> h_tmp) */
	// TODO //
	/**********************************/
	double pi=0.0;
	for(int i=0;i<grid.x;i++)pi += h_tmp[i];
	pi=pi*step;
	printf("PI= %.15f (Error = %e)\n", pi, fabs(acos(-1)-pi));
	return 0;
}

