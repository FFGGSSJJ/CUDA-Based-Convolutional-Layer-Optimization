/* v4
 * Optimized version of Forward-conv using 
 * ‘Shared memory matrix multiplication and input matrix unrolling (3 points)’
 * Optimized using 3d grid
*/


#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <iostream>


#define TILE_WIDTH 8
#define BLOCK_SIZE 1024
#define IMG_PER 25
//__constant__ float Kc[10000];

// matrix multiplicarion kernel
// C = y = k*x is the output matrix with size 10*M*(H_out*W_out)
// B = x is input matrix with size 10*(C*K*K)*(H_out*W_out)
// A = k is input matrix with size M*(C*K*K)
__global__ void matrixMultiply(const float *A, const float *B, float *C,
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns,
                                int numCRows, int numCColumns)
{
    /*
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps       4   16
    C - number of input feature maps        1   4
    H - input height dimension              86  40
    W - input width dimension               86  40
    K - kernel height and width (K x K)     7   7
    */
    
  __shared__ float A_ds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_ds[IMG_PER][TILE_WIDTH][TILE_WIDTH];
  
  int img_id = blockIdx.z;
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  float Cval = 0;
  int k = numAColumns%TILE_WIDTH==0?numAColumns/TILE_WIDTH:numAColumns/TILE_WIDTH+1;
  for (int i = 0; i < k; i++) {
    if (Row < numARows && (i*TILE_WIDTH+tx < numAColumns))
      A_ds[ty][tx] = A[Row*numAColumns + i*TILE_WIDTH + tx];  // notice that shared memory is stored as (y, x)
    else  A_ds[ty][tx] = 0;
    if ((i*TILE_WIDTH + ty) < numBRows && Col < numBColumns)
      B_ds[img_id][ty][tx] = B[img_id*numBRows*numBColumns + (i*TILE_WIDTH + ty)*numBColumns + Col];
    else  B_ds[img_id][ty][tx] = 0;
    __syncthreads();
    for (int j = 0; j < TILE_WIDTH; j++){
      Cval += A_ds[ty][j]*B_ds[img_id][j][tx];
      //Cval += Kc[Row*numAColumns + i*TILE_WIDTH + j] * B_ds[img_id][j][tx];
    }
    __syncthreads();
    if (Row < numCRows && Col < numCColumns)  C[img_id*numCRows*numCColumns + Row*numCColumns + Col] = Cval;
  }
}


// CUDA version of unrolling the X
// x_unroll is of size 10*(C*K*K)*(H_out*W_out)
// x is of size 10*(C*H*W)
__global__ void unroll_kernel(float *x_unroll, const float *x, const int C, const int H, const int W, const int K)
{
    int img_id = blockIdx.y;    // index of the image
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;   // size of 

    if (idx < C * W_unroll) {
        int c = idx/W_unroll;   // current index of channels
        int s = idx%W_unroll;   // 
        int h_out = s/W_out;    // row index
        int w_out = s%W_out;    // col index
        int w_unroll = h_out*W_out + w_out;
        int w_base = c*K*K;     // start of the channel
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p*K + q;
                x_unroll[img_id*(C*K*K*W_unroll) + h_unroll*W_unroll + w_unroll] = x[img_id*(C*H*W) + c*H*W + (h_out+p) * W + (w_out+q)];
            }
        }
    }
}

// function used to call cuda kernel unroll_kernel to genrate x_unroll
// x_unroll is output matrix with size 10*(C*K*K)*(H_out*W_out)
// x is input with size 10*C*H*W
void unroll_gpu(float *x_unroll, const float *x, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    dim3 GridDim(ceil(1.0*C*H_out*W_out/BLOCK_SIZE), IMG_PER);
    //int block_num = ceil(1.0*C*H_out*W_out/BLOCK_SIZE);
    // std::cout << "H_out: " << H_out << " W_out: " << W_out << " C: " << C << "\n";
    // std::cout << "unroll block: " << block_num << "\n";
    
    unroll_kernel<<<GridDim, BLOCK_SIZE>>>(x_unroll, x, C, H, W, K);
}

// function used to call cuda kernel forward_multi
// y = k*x is the output matrix with size 10*M*(H_out*W_out)
// x is input matrix with size 10*(C*K*K)*(H_out*W_out)
// k is input matrix with size M*(C*K*K)
void gemm(float *y, const float *x, const float *k, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H-K+1; int W_out = W-K+1;
    int numxRows = C*K*K; int numxColumns = H_out*W_out;
    int numkRows = M; int numkColumns = C*K*K;
    int numyRows = M; int numyColumns = H_out*W_out;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil(1.0*numyColumns/TILE_WIDTH), ceil(1.0*numyRows/TILE_WIDTH), IMG_PER);
    //for (int i = 0; i < M*(C*K*K); i++) std::cout << k[i] << " ";
    matrixMultiply<<<dimGrid, dimBlock>>>(k, x, y, numkRows, numkColumns, numxRows, numxColumns, numyRows, numyColumns);
}







__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    cudaMalloc((void**)device_y_ptr, B*M*H_out*W_out*sizeof(float));
    cudaMalloc((void**)device_x_ptr, B*C*H*W*sizeof(float));
    cudaMalloc((void**)device_k_ptr, M*C*K*K*sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(Kc, host_k, sizeof(float) * M * C * K * K);

    // conv_forward_gpu(*device_y_ptr, *device_x_ptr, *device_k_ptr, B, M, C, H, W, K);
    // conv_forward_gpu_epilog(host_y, *device_y_ptr, *device_x_ptr, *device_k_ptr, B, M, C, H, W, K);


    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    // int H_grid = ceil(1.*H_out/TILE_WIDTH);
    // int W_grid = ceil(1.*W_out/TILE_WIDTH);
    // int Z = H_grid*W_grid;
    //get_device_properties();
    float* X_unroll;
    cudaMalloc(&X_unroll, sizeof(float)*IMG_PER*(C*K*K)*(H_out*W_out));
    for (int b = 0; b < B ; b += IMG_PER) {
        unroll_gpu(X_unroll, device_x + b*C*H*W, C, H, W, K);

        // int H_out = H - K + 1;
        // int W_out = W - K + 1;
        // dim3 GridDim(ceil(1.0*C*H_out*W_out/BLOCK_SIZE), IMG_PER);
        // int block_num = ceil(1.0*C*H_out*W_out/BLOCK_SIZE);
        // unroll_kernel<<<GridDim, BLOCK_SIZE>>>(x_unroll, x, C, H, W, K);
    
    
        // y = k*x is the output matrix with size M*(H_out*W_out)
        // x is input matrix with size (C*K*K)*(H_out*W_out)
        // k is input matrix with size M*(C*K*K)
        gemm(device_y+b*M*H_out*W_out, X_unroll, device_k, M ,C, H, W, K);
        // int H_out = H-K+1; int W_out = W-K+1;
        // int numxRows = C*K*K; int numxColumns = H_out*W_out;
        // int numkRows = M; int numkColumns = C*K*K;
        // int numyRows = M; int numyColumns = H_out*W_out;

        // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        // dim3 dimGrid(ceil(1.0*numyColumns/TILE_WIDTH), ceil(1.0*numyRows/TILE_WIDTH), IMG_PER);
        // //for (int i = 0; i < M*(C*K*K); i++) std::cout << k[i] << " ";
        // matrixMultiply<<<dimGrid, dimBlock>>>(k, x, y, numkRows, numkColumns, numxRows, numxColumns, numyRows, numyColumns);
    }

    cudaFree(X_unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, B*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
