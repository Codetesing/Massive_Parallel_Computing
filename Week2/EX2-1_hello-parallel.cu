#include <stdio.h>

__global__ void hello(void) {
	printf("hello CUDA %d !\n", threadIdx.x);
}

int main(void) {
	hello << < 8, 2 >> > ();

	return 0;
}