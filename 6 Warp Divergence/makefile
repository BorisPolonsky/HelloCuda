solution=warpDivergence
all=$(solution) warpDivergence.o
warpDivergence: warpDivergence.o
	nvcc warpDivergence.o -o warpDivergence -arch sm_50
warpDivergence.o: warpDivergence.cu
	nvcc warpDivergence.cu -c -arch sm_50
.PHONY: clean
clean:
	rm $(all)
