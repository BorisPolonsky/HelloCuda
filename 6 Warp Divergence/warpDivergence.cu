#include "stdio.h"
#include <sys/time.h>

#define FALSE 0
#define TRUE !FALSE

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void simpleWarpDivergence(double *arr, const int arrSize)
{
	int i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	if(i<arrSize)
	{	
		if(i%2==0)
			arr[i]=0;
		else
			arr[i]=1;
	}
}

__global__ void minimalWarpDivergence(double *arr, const int arrSize)
{
	int i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	if(i<arrSize)
	{	
		if(TRUE)
			arr[i]=i%2;
		else
			arr[i]=i%2;
	}
}
int main()
{
	const int arrSize=1<<16;
	double t=0;
	double *d_arr=NULL;
	if(cudaMalloc((void**)&d_arr, arrSize*sizeof(double))!=cudaSuccess)
	{
		printf("Failed to allocated enough memory on GPU.\n");
		exit(-1);
	}
	cudaMemset(d_arr, 0, arrSize*sizeof(double));
	t=cpuSecond();
	simpleWarpDivergence<<<1<<8, 1<<8>>>(d_arr,arrSize);
	t=cpuSecond()-t;
	printf("Array initialization with simple warp divergence took %lf s.\n", t);

	cudaMemset(d_arr, 0, arrSize*sizeof(double));
	t=cpuSecond();
	minimalWarpDivergence<<<1<<8, 1<<8>>>(d_arr,arrSize);
	t=cpuSecond()-t;
	printf("Array initialization with minimal warp divergence took %lf s.\n", t);
	cudaFree(d_arr);
	cudaDeviceReset();
	return 0;
}
