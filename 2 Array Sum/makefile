all = arraySum arraySum.o

arraySum: arraySum.o
	nvcc arraySum.o -o arraySum -arch sm_50 
arraySum.o: arraySum.cu
	nvcc -c -arch sm_50 arraySum.cu
.PHONY: clean
clean: $(all) 
	rm $(all) 
