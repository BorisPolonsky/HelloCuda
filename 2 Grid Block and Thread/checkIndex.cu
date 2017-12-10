#include <stdio.h>

__global__ void checkIndex(void)
{
	printf("grid dim:\t(%d, %d, %d)\nblock dim\t(%d, %d, %d)\nblockIdx:\t(%d, %d, %d)\nthreadIdx:\t(%d, %d, %d)\n\n",
					gridDim.x, gridDim.y, gridDim.z, 
					blockDim.x, blockDim.y, blockDim.z, 
					blockIdx.x, blockIdx.y, blockIdx.z, 
					threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
	dim3 block(3, 1, 1);
	dim3 grid(2, 1, 1);
	printf("grid dim: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
	printf("block dim: (%d, %d, %d)\n\n", block.x, block.y, block.z);
	checkIndex<<<grid, block>>>();
	cudaDeviceReset();
	return 0;
}
