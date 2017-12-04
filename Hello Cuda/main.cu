/*
 * hello.cu
 *
 *  Created on: Dec 4, 2017
 *      Author: polonsky
 */

using namespace std;
#include <iostream>

float sum(float *p , int num)
{
	float ret=0;
	for(int i=0;i<num;i++)
		ret += p[i];
	return ret;
}

void set(float *p , int num)
{
	for(int i=0;i<num;i++)
		p[i] = i;
}

int main(void)
{
	cout<<"Hello Cuda 8.0!"<<endl;
	int num_of_floats = 100;

	float *cpuArr1 = (float *)malloc(num_of_floats * sizeof(float)), *cpuArr2 = (float *)malloc(num_of_floats * sizeof(float));
	if (cpuArr1 == NULL || cpuArr2 == NULL)
	{
		cout << "Unable to allocate memory in GPU." << endl;
		return 0;
	}
	set(cpuArr1, num_of_floats);
	cout << "Sum of Arr1 in CPU :" << sum(cpuArr1,num_of_floats) << endl;
	float *gpuArr1 = NULL, *gpuArr2 = NULL;
	if ((cudaMalloc((void**)&gpuArr1, num_of_floats * sizeof(float)) != cudaSuccess) || (cudaMalloc((void**)&gpuArr2, num_of_floats * sizeof(float)) != cudaSuccess))
	{
		cout << "Unable to allocate memory in GPU." << endl;
	}
	cudaMemcpy(gpuArr1, cpuArr1, num_of_floats * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuArr2, gpuArr1, num_of_floats * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cpuArr2, gpuArr2, num_of_floats * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << "GPU :" << sum(q, num_of_floats) << endl;
	cout << "Sum of Arr2 in CPU :" << sum(cpuArr2 ,num_of_floats) << endl;
	free(cpuArr1);
	free(cpuArr2);
	cudaFree(gpuArr1);
	cudaFree(gpuArr2);
	return 0;
}
