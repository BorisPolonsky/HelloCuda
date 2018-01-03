Solution = matrixSum
all = $(Solution) matrixSum.o
$(Solution): matrixSum.o
	nvcc matrixSum.o -arch sm_50 -o $(Solution)
matrixSum.o: matrixSum.cu
	nvcc matrixSum.cu -c -arch sm_50
.PHONY: clean
clean:
	rm $(all)
