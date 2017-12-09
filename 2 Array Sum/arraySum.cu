#include <stdio.h>

void setArray(float *, int);
void cpuSum(float *, float *, float *, int);
void checkCpuMalloc(float *);
void checkGpuMalloc(cudaError_t);
int arrEqual(float*, float*, int);

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
	//To be implemented
	
	cudaMemcpy(h_y2, d_y, num * sizeof(float), cudaMemcpyDeviceToHost);
	if(arrEqual(h_y1, h_y2, num))
		printf("Equal\n");
	else
		printf("Not equal.\n");
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
