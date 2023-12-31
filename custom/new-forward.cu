/* v8
 * Optimized version of Forward-conv using 
 * ‘Tiled shared memory convolution (2 points)’
 * 'Weight matrix (kernel values) in constant memory (1 point)'
 * 'Multiple kernel implementations for different layer sizes (1 point)'
 * 'Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (1 point)'
 * 'Using Streams to overlap computation with data transfer (4 point)'
 * Pinned Memory
*/



#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define TILE_WIDTH_8 8
#define TILE_WIDTH_16 8
#define IMG_ONCE 1250
__constant__ float Kc[16*4*7*7];

__global__ void convf_shared_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int TILE_WIDTH)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define kc4d(i3, i2, i1, i0) Kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    extern __shared__ float shared_mem[];
    int X_tile_width = TILE_WIDTH + K - 1;  // size of the X needed for a tile
    float* Xshared = &shared_mem[0];    // shared memory of size X_tile_width * X_tile_width

    int n = blockIdx.x;     // number of input batches B
    int m = blockIdx.y;     // number of output features M
    int H_grid = ceil(1.*H_out/TILE_WIDTH);
    int W_grid = ceil(1.*W_out/TILE_WIDTH);
    int h = (blockIdx.z/W_grid)*TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z%W_grid)*TILE_WIDTH + threadIdx.x;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_start = (blockIdx.z/W_grid)*TILE_WIDTH;
    int w_start = (blockIdx.z%W_grid)*TILE_WIDTH;

    float acc = 0.0f;
    
    for (int c = 0; c < C; c++) {
        // load X into shared memory
        for (int i = h; i < h_start+X_tile_width; i += TILE_WIDTH) {
            for (int j = w; j < w_start+X_tile_width; j += TILE_WIDTH) {
                if ((i < H) && (j < W))
                    Xshared[(i - h_start)*X_tile_width + (j - w_start)] = x4d(n, c, i, j);
                else
                    Xshared[(i - h_start)*X_tile_width + (j - w_start)] = 0;
            }
        }
        __syncthreads();
        
        // calculate the convolution
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                //if ((h0+p < X_tile_width) && (w0+q < X_tile_width))
                if (h < H_out && w < W_out) 
                    acc += Xshared[(h0+p)*X_tile_width + (w0+q)] * kc4d(m, c, p, q);
                    //acc += x4d(n, c, h+p, w+q) * kc4d(m, c, p, q);
            }
        }
        __syncthreads();
    }
    if (n<B && m<M && h<H_out && w<W_out)
        y4d(n, m, h, w) = acc;
    
   

#undef y4d
#undef x4d
#undef k4d
#undef kc4d
}

// x: B*C*H*W
// y: B*M*H_out*W_out
// k: M*C*K*K
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    cudaMalloc((void**)device_y_ptr, B*M*H_out*W_out*sizeof(float));
    cudaMalloc((void**)device_x_ptr, B*C*H*W*sizeof(float));
    cudaMalloc((void**)device_k_ptr, M*C*K*K*sizeof(float));
    
    cudaHostRegister((float*)host_x, B*C*H*W*sizeof(float), cudaHostAllocDefault);
    //cudaHostRegister((float*)host_k, M*C*K*K*sizeof(float), cudaHostAllocDefault);
    cudaHostRegister((float*)host_y, B*M*H_out*W_out*sizeof(float), cudaHostAllocDefault);
    
    cudaMemcpyToSymbol(Kc, host_k, sizeof(float) * M * C * K * K);


    int H_grid8 = ceil(1.*H_out/TILE_WIDTH_8);
    int W_grid8 = ceil(1.*W_out/TILE_WIDTH_8);
    int Z_8 = H_grid8*W_grid8;
    int H_grid16 = ceil(1.*H_out/TILE_WIDTH_16);
    int W_grid16 = ceil(1.*W_out/TILE_WIDTH_16);
    int Z_16 = H_grid16*W_grid16;

    // Creat Streams
    // cudaStream_t stream[8];
    // for (int i = 0; i < 8; i++) cudaStreamCreate(&stream[i]);
    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);
    cudaStreamCreate(&stream7);


    if (M == 4 && C == 1) {  // first layer using tile width 16
        dim3 BlockDim16(TILE_WIDTH_16, TILE_WIDTH_16, 1);
        dim3 GridDim16(IMG_ONCE, M, Z_16);
        #pragma unroll 8
            for (int b = 0; b < B; b += IMG_ONCE*8) {
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*0)*C*H*W, host_x+(b+IMG_ONCE*0)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream0);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*1)*C*H*W, host_x+(b+IMG_ONCE*1)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream1);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*2)*C*H*W, host_x+(b+IMG_ONCE*2)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream2);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*3)*C*H*W, host_x+(b+IMG_ONCE*3)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream3);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*4)*C*H*W, host_x+(b+IMG_ONCE*4)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream4);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*5)*C*H*W, host_x+(b+IMG_ONCE*5)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream5);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*6)*C*H*W, host_x+(b+IMG_ONCE*6)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream6);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*7)*C*H*W, host_x+(b+IMG_ONCE*7)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream7);

                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream0>>>(*device_y_ptr+(b+IMG_ONCE*0)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*0)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream1>>>(*device_y_ptr+(b+IMG_ONCE*1)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream2>>>(*device_y_ptr+(b+IMG_ONCE*2)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream3>>>(*device_y_ptr+(b+IMG_ONCE*3)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream4>>>(*device_y_ptr+(b+IMG_ONCE*4)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream5>>>(*device_y_ptr+(b+IMG_ONCE*5)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream6>>>(*device_y_ptr+(b+IMG_ONCE*6)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);
                convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1)), stream7>>>(*device_y_ptr+(b+IMG_ONCE*7)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_16);


                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*0)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*0)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream0);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*1)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*1)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream1);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*2)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*2)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream2);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*3)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*3)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream3);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*4)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*4)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream4);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*5)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*5)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream5);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*6)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*6)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream6);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*7)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*7)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream7);
                
            }
        
    }
    if (M == 16 && C == 4) {// second layer using tile width 8
        dim3 BlockDim8(TILE_WIDTH_8, TILE_WIDTH_8, 1);
        dim3 GridDim8(IMG_ONCE, M, Z_8);
        #pragma unroll 8
            for (int b = 0; b < B; b += IMG_ONCE*8) {
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*0)*C*H*W, host_x+(b+IMG_ONCE*0)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream0);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*1)*C*H*W, host_x+(b+IMG_ONCE*1)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream1);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*2)*C*H*W, host_x+(b+IMG_ONCE*2)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream2);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*3)*C*H*W, host_x+(b+IMG_ONCE*3)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream3);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*4)*C*H*W, host_x+(b+IMG_ONCE*4)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream4);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*5)*C*H*W, host_x+(b+IMG_ONCE*5)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream5);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*6)*C*H*W, host_x+(b+IMG_ONCE*6)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream6);
                cudaMemcpyAsync(*device_x_ptr+(b+IMG_ONCE*7)*C*H*W, host_x+(b+IMG_ONCE*7)*C*H*W, IMG_ONCE*C*H*W*sizeof(float), cudaMemcpyHostToDevice, stream7);

                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream0>>>(*device_y_ptr+(b+IMG_ONCE*0)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*0)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream1>>>(*device_y_ptr+(b+IMG_ONCE*1)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*1)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream2>>>(*device_y_ptr+(b+IMG_ONCE*2)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*2)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream3>>>(*device_y_ptr+(b+IMG_ONCE*3)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*3)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream4>>>(*device_y_ptr+(b+IMG_ONCE*4)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*4)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream5>>>(*device_y_ptr+(b+IMG_ONCE*5)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*5)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream6>>>(*device_y_ptr+(b+IMG_ONCE*6)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*6)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);
                convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1)), stream7>>>(*device_y_ptr+(b+IMG_ONCE*7)*M*H_out*W_out, *device_x_ptr+(b+IMG_ONCE*7)*C*H*W, *device_k_ptr, B, M, C, H, W, K, TILE_WIDTH_8);

                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*0)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*0)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream0);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*1)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*1)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream1);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*2)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*2)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream2);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*3)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*3)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream3);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*4)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*4)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream4);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*5)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*5)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream5);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*6)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*6)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream6);
                cudaMemcpyAsync((float*)host_y+(b+IMG_ONCE*7)*M*H_out*W_out, *device_y_ptr+(b+IMG_ONCE*7)*M*H_out*W_out, IMG_ONCE*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost, stream7);
            }
    }
    // cudaMemcpy(*device_x_ptr, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_k_ptr, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    

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
    // // Set the kernel dimensions and call the kernel
    // int H_out = H - K + 1;
    // int W_out = W - K + 1;
    // int H_grid8 = ceil(1.*H_out/TILE_WIDTH_8);
    // int W_grid8 = ceil(1.*W_out/TILE_WIDTH_8);
    // int Z_8 = H_grid8*W_grid8;
    // int H_grid16 = ceil(1.*H_out/TILE_WIDTH_16);
    // int W_grid16 = ceil(1.*W_out/TILE_WIDTH_16);
    // int Z_16 = H_grid16*W_grid16;

    // dim3 BlockDim8(TILE_WIDTH_8, TILE_WIDTH_8, 1);
    // dim3 BlockDim16(TILE_WIDTH_16, TILE_WIDTH_16, 1);
    // dim3 GridDim8(IMG_ONCE, M, Z_8);
    // dim3 GridDim16(IMG_ONCE, M, Z_16);
    // if (M == 4 && C == 1) {  // first layer
    //     for (int i = 0; i < B; i += IMG_ONCE) {

    //     }
    //     convf_shared_kernel<<<GridDim16, BlockDim16, sizeof(float)*((TILE_WIDTH_16 + K - 1)*(TILE_WIDTH_16 + K - 1))>>>(device_y, device_x, device_k, B, M, C, H, W, K, TILE_WIDTH_16);
    // }
    // if (M == 16 && C == 4) {// second layer
    //     convf_shared_kernel<<<GridDim8, BlockDim8, sizeof(float)*((TILE_WIDTH_8 + K - 1)*(TILE_WIDTH_8 + K - 1))>>>(device_y, device_x, device_k, B, M, C, H, W, K, TILE_WIDTH_8);
    // }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //cudaMemcpy(host_y, device_y, B*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost);
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
