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

void arrInit(double *arr, const int length)
{
	;
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
	free(h_x1);
	free(h_x2);
	free(h_y);
	return 0;
}


