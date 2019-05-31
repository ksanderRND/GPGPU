#include <cstdint>
#include <math.h>

#define BLOCK_SIZE 16

template<typename T>
__device__ T* get_2d_array_element(T* _2d_array_base, uint32_t row, uint32_t column, size_t pitch)
{
    T* p_result = (T*)((char*)_2d_array_base + row*pitch) + column;
    return p_result;
}


__device__ __constant__ float GX_kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__device__ __constant__ float GY_kernel[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

extern "C"
{
    __global__ void sobel_filter(const float* __restrict__ img, float* __restrict__ sobel,
        size_t order_x, size_t order_y, size_t kernel_size, size_t bytes_per_float)
 {

        int row = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        int column = BLOCK_SIZE * blockIdx.y + threadIdx.y;

        int pitch = order_y * bytes_per_float;

        __shared__ float I[BLOCK_SIZE][BLOCK_SIZE];

        if ((row >= order_x) || (column >= order_y) || (row < 0) || (column < 0))
        {
            I[threadIdx.x][threadIdx.y] = 0.0;
            return;
        }

        int xi = 0;
        int yi = 0;

        float sx = 0;
        float sy = 0;

        I[threadIdx.x][threadIdx.y] = *get_2d_array_element(img, row, column, pitch);
        __syncthreads();

        for (xi=-1; xi<2; xi++)
        {
            for (yi=-1; yi<2; yi++)
            {
                if ((row+xi >= order_x) || (column+yi >= order_y) || (row+xi < 0) || (column+yi < 0))
                {
                    sx+=0.0;
                    sy+=0.0;
                }
                else
                {
            if ((threadIdx.x + xi < 0) || (threadIdx.x + xi > BLOCK_SIZE-1) || (threadIdx.y + yi > BLOCK_SIZE-1) || (threadIdx.y + yi < 0))
                  {
                    sx+= *get_2d_array_element(img, row+xi, column+yi, pitch) * GX_kernel[xi+1][yi+1];
                    sy+= *get_2d_array_element(img, row+xi, column+yi, pitch) * GY_kernel[xi+1][yi+1];
                  }
                    else
                    {
                      sx+= I[threadIdx.x+xi][threadIdx.y+yi] * GX_kernel[xi+1][yi+1];
                      sy+= I[threadIdx.x+xi][threadIdx.y+yi] * GY_kernel[xi+1][yi+1];
                    }
                }
            }
        }

        *get_2d_array_element(sobel, row, column, pitch) = sqrt(sx*sx + sy*sy);
 }

}
