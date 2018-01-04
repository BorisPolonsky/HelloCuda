#include "stdio.h"
#define FALSE 0
#define TRUE !FALSE
double *mallocMatrix(const int row, const int column)
{
	return (double*)malloc(row*column*sizeof(double));
}

void matrixInit(double *matrix, const int row, const int column)
{
	;
}


int matEqual(double *mat1, double *mat2, const int row, const int column)
{
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<column;j++)
		{
			int k=i*column+j;
			if(mat1[k]!=mat2[k])
			{
				printf("Entry %d doens't match.\n",k);
				return FALSE;
			}
		}
	}
	return TRUE;
}

void matrixSumCpu(double *m1, double *m2, double *n, const int row, const int column)
{
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<column; j++)
		{
			int k = i * column + j;
			n[k]=m1[k]+m2[k];
		}
	}
}

__global__ void _2dGrid2dBlockMatSum(double *m1, double *m2, double *n, const int row, const int column)
{
	int rowIndex=blockIdx.x*blockDim.x+threadIdx.x;
	int columnIndex=blockIdx.y*blockDim.y+threadIdx.y;
	if(rowIndex<row&&columnIndex<column)
	{
		int i=rowIndex*column+columnIndex;//flatten
		n[i]=m1[i]+m2[i];
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

int main()
{
	int row=1<<10, column=1<<10;
	double *h_m1=NULL, *h_m2=NULL,*h_n1=NULL, *h_n2=NULL;//n=m1+m2
	h_m1=mallocMatrix(row, column);
	h_m2=mallocMatrix(row, column);
	h_n1=mallocMatrix(row, column);
	h_n2=mallocMatrix(row, column);
	if(h_m1==NULL||h_m2==NULL||h_n1==NULL||h_n2==NULL)
	{
		printf("Unable to allocate enough memory on CPU\n");
		exit(-1);
	}
	matrixInit(h_m1,row,column);
	matrixInit(h_m2,row,column);
	printf("Summing matrices on CPU...\n");
	matrixSumCpu(h_m1,h_m2,h_n1,row,column);
	double *d_m1=NULL, *d_m2=NULL, *d_n=NULL;
	checkGpuMalloc(cudaMalloc((void**)&d_m1, row*column*sizeof(double)));
	checkGpuMalloc(cudaMalloc((void**)&d_m2, row*column*sizeof(double)));
	checkGpuMalloc(cudaMalloc((void**)&d_n, row*column*sizeof(double)));
	cudaMemcpy(d_m1, h_m1, row*column*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2, h_m2, row*column*sizeof(double), cudaMemcpyHostToDevice);
	printf("Summing matrices on GPU with 2D grid and 2D blocks.\n");
	_2dGrid2dBlockMatSum<<<(1<<5,1<<5),(1<<5, 1<<5)>>>(d_m1, d_m2, d_n, row, column);
	cudaDeviceSynchronize();	
	cudaMemcpy(h_n2, d_n, row*column*sizeof(double), cudaMemcpyDeviceToHost);
	if(matEqual(h_n1, h_n2, row, column))
		printf("Matrices match.\n");
	else
		printf("Matrices don't match.\n");
	free(h_m1);
	free(h_m2);
	free(h_n1);
	free(h_n2);
	cudaFree(d_m1);
	cudaFree(d_m2);
	cudaFree(d_n);
	cudaDeviceReset();
	return 0;
}
