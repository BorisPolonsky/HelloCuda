#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello Cuda 8.0!(From GPU)\n");
}

int main()
{
	printf("Hello Cuda 8.0!(From CPU)\n");
	helloFromGPU<<<1, 10>>>();
	cudaDeviceReset();
	return 0;
}


