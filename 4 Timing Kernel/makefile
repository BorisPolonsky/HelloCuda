all = timing timing.o
timing: timing.o
	nvcc -arch sm_50 -o timing timing.o
timing.o: timing.cu
	nvcc -c -arch sm_50 timing.cu
.PHONY: clean

clean:
	rm $(all)
