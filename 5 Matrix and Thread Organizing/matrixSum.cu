#include "stdio.h"
double *mallocMatrix(const int row, const int column)
{
	return (double*)malloc(row*column*sizeof(double));
}

void matrixInit(double *matrix, const int row, const int column)
{
	;
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

int main()
{
	int row=1<<10, column=1<<10;
	double *h_m1=NULL, *h_m2=NULL,*h_n=NULL;//n=m1+m2
	h_m1=mallocMatrix(row, column);
	h_m2=mallocMatrix(row, column);
	h_n=mallocMatrix(row, column);
	if(h_m1==NULL||h_m2==NULL||h_n==NULL)
	{
		printf("Unable to allocate enough memory on CPU\n");
		exit(-1);
	}
	matrixInit(h_m1,row,column);
	matrixInit(h_m2,row,column);
	matrixSumCpu(h_m1,h_m2,h_n,row,column);
	free(h_m1);
	free(h_m2);
	free(h_n);
	return 0;
}
