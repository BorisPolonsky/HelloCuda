all = hello hello.o
.PHONY: all
all: hello	
hello : hello.o
	nvcc hello.o -o hello
hello.o: hello.cu
	nvcc -c -arch sm_20 hello.cu
.PHONY: clean
clean:
	rm $(all)
