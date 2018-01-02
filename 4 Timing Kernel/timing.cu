#include "stdio.h"
#include <sys/time.h> 
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void cpuSum(double *x1, double *x2, double *y, const int arrSize)
{
	for(int i=0; i< arrSize; i++)
	{
		y[i] = x1[i] + x2[i];
	}
}

void arrInit(double *arr, const int arrSize)
{
	;
}

__global__ void arraySumKernel(double *x1, double *x2, double *y, double arrSize)
{
	int blockOffset = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	int threadOffset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	int i = blockOffset * blockDim.x*blockDim.y*blockDim.z + threadOffset;
	if(i<arrSize)
	{
		y[i]=x1[i]+x2[i];
	}
}

void checkGpuMalloc(cudaError_t code)
{
	if(code != cudaSuccess)
	{
		exit(-1);
		printf("CUDA ERROR occured. ");
	}
}
void gpuSum(double *h_x1, double *h_x2, double *h_y, int arrSize)
{
	double *d_x1=NULL, *d_x2=NULL, *d_y=NULL;
	//Pre-allocate memory and check status.
	checkGpuMalloc(cudaMalloc((void**)&d_x1, arrSize*sizeof(double)));
	checkGpuMalloc(cudaMalloc((void**)&d_x2, arrSize*sizeof(double)));
	checkGpuMalloc(cudaMalloc((void**)&d_x2, arrSize*sizeof(double)));
	cudaMemcpy(d_x1, h_x1, arrSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, h_x2, arrSize*sizeof(double), cudaMemcpyHostToDevice);
	arraySumKernel<<<1,arrSize>>>(d_x1, d_x2, d_y, arrSize);//One block. 
	cudaMemcpy(h_y, d_y, arrSize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_y);
}
int main()
{
	int arrSize=1<<24; //16M
	double tBegin, tEnd;
	double *h_x1=NULL, *h_x2=NULL, *h_y=NULL;
	h_x1=(double*)malloc(arrSize*sizeof(double));
	h_x2=(double*)malloc(arrSize*sizeof(double));
	h_y=(double*)malloc(arrSize*sizeof(double));
	if(h_x1==NULL||h_x2==NULL||h_y==NULL)
	{
		printf("Failed to allocate enough memory\n");
		exit(-1);
	}
	arrInit(h_x1, arrSize);
	arrInit(h_x2, arrSize);
	tBegin=cpuSecond();
	cpuSum(h_x1, h_x2, h_y, arrSize);
	tEnd=cpuSecond();
	printf("Time cost (CPU computation): %lfs\n", tEnd-tBegin);
	tBegin=cpuSecond();
	gpuSum(h_x1, h_x2, h_y, arrSize);
	tEnd=cpuSecond();
	printf("Time cost (GPU computation): %lfs\n", tEnd-tBegin);
	free(h_x1);
	free(h_x2);
	free(h_y);
	cudaDeviceReset();
	return 0;
}
