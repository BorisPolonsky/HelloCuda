solution = checkIndex
all = $(solution) checkIndex.o
$(solution): checkIndex.o
	nvcc checkIndex.o -arch sm_50 -o checkIndex
checkIndex.o: checkIndex.cu
	nvcc -c -arch sm_50 checkIndex.cu
clean: 
	rm $(all)
.PHONY: clean
