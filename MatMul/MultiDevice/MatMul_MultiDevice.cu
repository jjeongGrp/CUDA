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
__global__ void MatMultOnGPU(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
	int tx = blockDim.x*blockIdx.x + threadIdx.x;	// col of C 
	int ty = blockDim.y*blockIdx.y + threadIdx.y;	// row of C 
	int tid = ty*Bcols+tx;


	float sum=0.0f;
	if(tx < Bcols && ty <Arows )
	{
		for(int i=0;i<Acols;i++)
		{
			sum += A[ty*Acols + i]*B[i*Bcols+tx];
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
	int Arows=300, Acols=300, Bcols=400;
	int threads_x=32, threads_y=32;
	if(argc>1) threads_x = atoi(argv[3]);
	if(argc>2) threads_y = atoi(argv[4]);

	int ngpus;
	CHECK(cudaGetDeviceCount(&ngpus));
	printf(" CUDA-capable device : %d\n", ngpus);

	CHECK(cudaMallocHost((float**)&MatA, Arows*Acols*sizeof(float)));
	CHECK(cudaMallocHost((float**)&MatB, Acols*Bcols*sizeof(float)));
	CHECK(cudaMallocHost((float**)&MatC, Arows*Bcols*sizeof(float)));
	CHECK(cudaMallocHost((float**)&gpu_MatC, Arows*Bcols*sizeof(float)));

	
	initialData(MatA, Arows*Acols);
	initialData(MatB, Acols*Bcols);
	
	Start=cpuTimer();
	MatMulOnCPU(MatA, MatB, MatC, Arows, Acols, Bcols);
	ElapsedTime=cpuTimer()-Start;
	printf("Elapsed Time on CPU : %f\n",ElapsedTime);
	/**********************************/

	/************ ON GPU **************/
	float **d_MatA=(float**)malloc(sizeof(float*)*ngpus);
	float **d_MatB=(float**)malloc(sizeof(float*)*ngpus);
	float **d_MatC=(float**)malloc(sizeof(float*)*ngpus);
	cudaStream_t *stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);

	for(int i=0;i<ngpus;i++)
	{
		CHECK(cudaSetDevice(i));
		CHECK(cudaMalloc((float**)&d_MatA[i], (Arows/ngpus)*Acols*sizeof(float)));
		CHECK(cudaMalloc((float**)&d_MatB[i], Acols*Bcols*sizeof(float)));
		CHECK(cudaMalloc((float**)&d_MatC[i], (Arows/ngpus)*Bcols*sizeof(float)));
		CHECK(cudaStreamCreate(&stream[i]));
	}

	Start=cpuTimer();
	for(int i=0;i<ngpus;i++)
	{
		CHECK(cudaSetDevice(i));
		CHECK(cudaMemcpyAsync(d_MatA[i],MatA+i*(Arows/ngpus)*Acols, (Arows/ngpus)*Acols*sizeof(float),cudaMemcpyHostToDevice,stream[i]));
		CHECK(cudaMemcpyAsync(d_MatB[i],MatB, Acols*Bcols*sizeof(float),cudaMemcpyHostToDevice,stream[i]));
		dim3 block(threads_x,threads_y,1);
		dim3 grid((Bcols+block.x-1)/block.x, (Arows/ngpus+block.y-1)/block.y, 1);
		MatMultOnGPU<<<grid, block,0,stream[i]>>>(d_MatA[i], d_MatB[i], d_MatC[i], Arows/ngpus, Acols, Bcols);

		CHECK(cudaMemcpyAsync(gpu_MatC+i*(Arows/ngpus)*Bcols, d_MatC[i], (Arows/ngpus)*Bcols*sizeof(float), cudaMemcpyDeviceToHost,stream[i]));
	}
	for(int i=0;i<ngpus;i++)
	{
		cudaSetDevice(i);
		cudaStreamSynchronize(stream[i]);
	}
	ElapsedTime=cpuTimer()-Start;
	printf("Elapsed Time on GPU : %f\n",ElapsedTime);
	/**********************************/
	checkResult(MatC, gpu_MatC, Arows*Bcols);

	CHECK(cudaFreeHost(MatA)),	CHECK(cudaFreeHost(MatB)),	CHECK(cudaFreeHost(MatC)),	CHECK(cudaFreeHost(gpu_MatC));
	for(int i=0;i<ngpus;i++)
		CHECK(cudaFree(d_MatA[i])), CHECK(cudaFree(d_MatB[i])), CHECK(cudaFree(d_MatC[i]));
	free(d_MatA), free(d_MatB), free(d_MatC);

	CHECK(cudaDeviceReset());
	return 0;
}
