#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
inline void CHECK(const cudaError_t error)
{
    if(error !=cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__,__LINE__);
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
    srand((unsigned int)time(&t));
    for(int i=0;i<size;i++)
        arr[i]=(float)(rand())/RAND_MAX;
}
void checkResult(float *host, float *gpu, const int N)
{
    double epsilon = 1.0e-8;
    bool match=1;
    for(int i=0;i<N;i++)
    {
        if(abs(host[i]-gpu[i])>epsilon)
        {
            match=0;
            printf("Matrices do not match!\n");
            printf("host %10.7f, gpu %10.7f at current %d\n", host[i], gpu[i], i);
            break;
        }
    }
    if(match)printf("Matrices match.\n");
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

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;	
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA*BLOCK_SIZE*by; 
    int aEnd = aBegin +wA-1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE*bx;
    // Step size used to integrate through the sub-matrix of B
    int bStep = BLOCK_SIZE*wB;
    float Csub = 0.0f;
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for(int a = aBegin, b=bBegin; a<=aEnd;  a+=aStep, b+=bStep)
    {

//        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  //      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx]=A[a+wA*ty+tx]; 
        Bs[ty][tx]=B[b+wB*ty+tx];
        __syncthreads();

#pragma unroll
        for(int k=0;k<BLOCK_SIZE;++k)
            Csub += As[ty][k]*Bs[k][tx];
        __syncthreads();
    }

    int c = wB*BLOCK_SIZE*by + BLOCK_SIZE*bx;
    C[c+wB*ty + tx] =Csub;
}
int main()
{   double Start, ElapsedTime;
    int block_size=32;
    int nrows=10*block_size;
    int ncols=20*block_size;
    float *MatA, *MatB, *MatC, *gpu_MatC;
    float *d_MatA, *d_MatB, *d_MatC;
    int size = nrows*ncols;

    /********** ON CPU **************/
    MatA=(float*)malloc(size*sizeof(float));
    MatB=(float*)malloc(size*sizeof(float));
    MatC=(float*)malloc(nrows*nrows*sizeof(float));
    gpu_MatC=(float*)malloc(nrows*nrows*sizeof(float));
    initialData(MatA, size);
    initialData(MatB, size);
    Start=cpuTimer();
    MatMulOnCPU(MatA, MatB, MatC, nrows, ncols);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on CPU: %f\n",ElapsedTime);
    /********************************/
    /********** ON GPU *****************/
    cudaMalloc((void**)&d_MatA, size*sizeof(float));
    cudaMalloc((void**)&d_MatB, size*sizeof(float));
    cudaMalloc((void**)&d_MatC, nrows*nrows*sizeof(float));
    Start=cpuTimer();
    // copy host memory to device
    cudaMemcpy(d_MatA, MatA, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, MatB, size*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(block_size, block_size); // sub-matrix dimension
    dim3 grid((nrows+block.x-1)/block.x, (nrows+block.y-1)/block.y);    // submatrix dimension of C
    matrixMulCUDA<32><<<grid,block>>>(d_MatC, d_MatA, d_MatB, ncols, nrows);
    cudaMemcpy(gpu_MatC,d_MatC, nrows*nrows*sizeof(float), cudaMemcpyDeviceToHost);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on GPU : %f\n",ElapsedTime);
    checkResult(MatC, gpu_MatC,nrows*nrows);
    free(MatA), free(MatB), free(MatC), free(gpu_MatC);
    cudaFree(d_MatA),   cudaFree(d_MatB),   cudaFree(d_MatC);

    cudaDeviceReset();
    return 0;
}

