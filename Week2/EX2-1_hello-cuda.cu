#include <stdio.h>

__global__ void hello(void) {
	printf("hello, CUDA!\n");
}

int main(void) {
	
	hello << < 1, 1 >> > ();
	
	return 0;
}