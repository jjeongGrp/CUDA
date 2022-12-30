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
ta(float *arr, const int size)
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
__global__ void MatMulOnGPUGlobal(float *A, float *B, float *C, const int Arows, const int Acols, const int Bcols)
{
    int tx = blockDim.x*blockIdx.x + threadIdx.x;   // col of C
    int ty = blockDim.y*blockIdx.y + threadIdx.y;   // row of C
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
template <int BLOCK_SIZE> __global__ void
MatMulOnGPUShared(float *C, float *A, float *B, int wA, int wB)
{
    // 스레드가 속한 블록 인덱스
    int bx = blockIdx.x;	
    int by = blockIdx.y;
    // 블록 내의 스레드 인덱스
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 스레드 블록에의해 처리되는 행렬 A의 첫 번째 sub-matrix index
    int aBegin = wA*BLOCK_SIZE*by; 
    // 스레드 블록에 의해 처리되는 행렬 A의 마지막 sub-matrix index 
    int aEnd = aBegin +wA-1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first-submatrix of B processed by the block
    int bBegin = BLOCK_SIZE*bx;
    // Step size used to integrate through the sub-matrix of B
    int bStep = BLOCK_SIZE*wB;
    float Csub = 0.0f;
    for(int a = aBegin, b=bBegin; a<=aEnd;  a+=aStep, b+=bStep)
    {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // 공유 메모리에 해당 블록에 해당하는 행렬 값 A, B를 할당
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
nt main(int argc, char **argv)
{
    double Start, ElapsedTime;
    float *MatA, *MatB, *MatC, *gpu_MatC;
    int Arows=320, Acols=640, Bcols=320;
    int nrows=Arows, ncols=Acols;
    int threads_x=32, threads_y=32;
    if(argc>1) Arows=atoi(argv[1]);
    if(argc>2) Acols=atoi(argv[2]);
    if(argc>3) Bcols=atoi(argv[3]);
    if(argc>4) threads_x=atoi(argv[4]);
    if(argc>5) threads_y=atoi(argv[5]);

    /*************** ON CPU ******************/
    CHECK(cudaMallocHost((float**)&MatA, Arows*Acols*sizeof(float)));
    CHECK(cudaMallocHost((float**)&MatB, Acols*Bcols*sizeof(float)));
    CHECK(cudaMallocHost((float**)&MatC, Arows*Bcols*sizeof(float)));
    CHECK(cudaMallocHost((float**)&gpu_MatC, Arows*Bcols*sizeof(float)));

    initialData(MatA, Arows*Acols);
    initialData(MatB, Acols*Bcols);
    Start=cpuTimer();
    MatMulOnCPU(MatA, MatB, MatC, Arows, Acols, Bcols);
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on CPU : \t\t\t%f sec\n", ElapsedTime);

    /************ ON GPU ******************/
    /************ Global Memory ******************/
    dim3 block(threads_x, threads_y);
    dim3 grid((Bcols+block.x-1)/block.x, (Arows+block.y-1)/block.y);

    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((float**)&d_MatA, Arows*Acols*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_MatB, Acols*Bcols*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_MatC, Arows*Bcols*sizeof(float)));

    /********************/
    Start=cpuTimer();
    CHECK(cudaMemcpy(d_MatA, MatA, Arows*Acols*sizeof(float), 
              cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, MatB, Acols*Bcols*sizeof(float), 
              cudaMemcpyHostToDevice));
    MatMulOnGPUGlobal<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Arows, Acols, Bcols);
    CHECK(cudaMemcpy(gpu_MatC, d_MatC, Arows*Bcols*sizeof(float), 
              cudaMemcpyDeviceToHost));
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on GPU ( Global memory) : \t%f sec\n", ElapsedTime);
    checkResult(MatC, gpu_MatC, Arows*Bcols);
    /**************************************/
    /************ Shared Memory ************/
    Start=cpuTimer();
    CHECK(cudaMemcpy(d_MatA, MatA, Arows*Acols*sizeof(float), 
              cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, MatB, Acols*Bcols*sizeof(float), 
              cudaMemcpyHostToDevice));
    MatMulOnGPUShared<32><<<grid, block>>>(d_MatC, d_MatA, d_MatB, ncols, 
             nrows);
    CHECK(cudaMemcpy(gpu_MatC, d_MatC, nrows*nrows*sizeof(float), 
                                 cudaMemcpyDeviceToHost));
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on GPU ( Shared memory) : \t%f sec\n", ElapsedTime);
    checkResult(MatC, gpu_MatC, Arows*Bcols);
    /***************************************/
    /********** Multi Device **************/
    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("  CUDA-capable device : %d\n", ngpus);
    float **d_MatAM=(float**)malloc(sizeof(float*)*ngpus);
    float **d_MatBM=(float**)malloc(sizeof(float*)*ngpus);
    float **d_MatCM=(float**)malloc(sizeof(float*)*ngpus);
    cudaStream_t *stream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*ngpus);

    for(int i=0;i<ngpus;i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((float**)&d_MatAM[i], (Arows/ngpus)*Acols*sizeof(float)));
        CHECK(cudaMalloc((float**)&d_MatBM[i], Acols*Bcols*sizeof(float)));
        CHECK(cudaMalloc((float**)&d_MatCM[i], (Arows/ngpus)*Bcols*sizeof(float)));
        CHECK(cudaStreamCreate(&stream[i]));
    }
    Start=cpuTimer();
    for(int i=0;i<ngpus;i++){
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpyAsync(d_MatAM[i], MatA+i*(Arows/ngpus)*Acols, 
                        (Arows/ngpus)*Acols*sizeof(float), cudaMemcpyHostToDevice,stream[i]));
        CHECK(cudaMemcpyAsync(d_MatBM[i], MatB, Acols*Bcols*sizeof(float), 
                         cudaMemcpyHostToDevice,stream[i]));
        MatMulOnGPUGlobal<<<grid, block,0,stream[i]>>>(d_MatAM[i], d_MatBM[i], 
                                                                       d_MatCM[i], Arows/ngpus, Acols, Bcols);
        CHECK(cudaMemcpyAsync(gpu_MatC+i*(Arows/ngpus)*Bcols, d_MatCM[i], 
                           (Arows/ngpus)*Bcols*sizeof(float), cudaMemcpyDeviceToHost,stream[i]));
    }
    for(int i=0;i<ngpus;i++){
        cudaSetDevice(i);
        cudaStreamSynchronize(stream[i]);
    }
    ElapsedTime=cpuTimer()-Start;
    printf("Elapsed Time on GPU ( Multi-Device) : \t%f sec\n", ElapsedTime);
    checkResult(MatC, gpu_MatC, Arows*Bcols);
    /**************************************/
    CHECK(cudaFreeHost(MatA)),  CHECK(cudaFreeHost(MatB));
    CHECK(cudaFreeHost(MatC)),  CHECK(cudaFreeHost(gpu_MatC));
    for(int i=0;i<ngpus;i++){
        CHECK(cudaFree(d_MatAM[i])), CHECK(cudaFree(d_MatBM[i]));
        CHECK(cudaFree(d_MatCM[i]));
    }
    free(d_MatAM), free(d_MatBM), free(d_MatCM);
    cudaFree(d_MatA), cudaFree(d_MatB), cudaFree(d_MatC);
    CHECK(cudaDeviceReset());
    return 0;
}

