#include <stdio.h>

void setArray(float *, int);
void cpuSum(float *, float *, float *, int);
void checkCpuMalloc(float *);
void checkGpuMalloc(cudaError_t);
int arrEqual(float*, float*, int);
__global__ void gpuSum(float *, float *, float *);

int main()
{
	int num = 64;
	float *h_x1 =(float *)malloc(num * sizeof(float));
	float *h_x2 =(float *)malloc(num * sizeof(float));
	float *h_y1 =(float *)malloc(num * sizeof(float));
	float *h_y2 =(float *)malloc(num * sizeof(float));

	checkCpuMalloc(h_x1);
	checkCpuMalloc(h_x2);
	checkCpuMalloc(h_y1);
	checkCpuMalloc(h_y2);

	setArray(h_x1, num);
	setArray(h_x2, num);
	
	float *d_x1 = NULL, *d_x2 = NULL, *d_y = NULL;
	checkGpuMalloc(cudaMalloc((void**)&d_x1, num * sizeof(float)));
	checkGpuMalloc(cudaMalloc((void**)&d_x2, num * sizeof(float)));
	checkGpuMalloc(cudaMalloc((void**)&d_y, num * sizeof(float)));
	cudaMemcpy(d_x1, h_x1, num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, h_x2, num * sizeof(float), cudaMemcpyHostToDevice);
	cpuSum(h_x1, h_x2, h_y1, num);
	//Perform array sum in GPU. 
	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2);
	gpuSum<<<grid, block>>>(d_x1, d_x2, d_y);
	cudaDeviceSynchronize(); //Wait for all kernels to complete. 
	cudaMemcpy(h_y2, d_y, num * sizeof(float), cudaMemcpyDeviceToHost);
	if(arrEqual(h_y1, h_y2, num))
		printf("Result from GPU is equal to result from CPU. \n");
	else
		printf("Result from GPU is NOT equal to result from CPU. \n");
	free(h_x1);
	free(h_x2);
	free(h_y1);
	free(h_y2);
	cudaFree(d_x1);
	cudaFree(d_x2);
	cudaFree(d_y);
	cudaDeviceReset();
	return 0;
}

void setArray(float *p, int num)
{
	for(int i = 0; i < num; i++)
	{
		p[i] = 0.618 * i;	
	}
}

void cpuSum(float *x1, float *x2, float *y, int num)
{
	for(int i = 0; i < num; i++)
	{
		y[i] = x1[i] +x2[i];
	}
}

int arrEqual(float *arr1, float *arr2, int num)
{
	for(int i=0; i < num; i++)
	{
		if(arr1[i] != arr2[i])
			return 0;
	}
	return 1;
}

void checkCpuMalloc(float *p)
{
	if(p == NULL)
		exit(-1);
}

void checkGpuMalloc(cudaError_t code)
{
	if(code != cudaSuccess)
		exit(-1);
}

__global__ void gpuSum(float *x1, float*x2, float *y)
{
	int threadOffset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; //thread in block
	int blockOffset = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y; //block in grid 
	int i = blockOffset * blockDim.x * blockDim.y * blockDim.z + threadOffset; 
	printf("Adding entry no. %d in GPU\n", i);
	y[i] = x1[i] + x2[i];
}
