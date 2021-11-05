#include "stdio.h"

__global__ void add_kernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	const int SIZE = 5;
	const int a[SIZE] = { 1, 2, 3, 4, 5 };
	const int b[SIZE] = { 10, 20, 30, 40, 50 };
	int c[SIZE] = { 0 };

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, SIZE * sizeof(int));

	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	add_kernel << <1, SIZE >> > (dev_c, dev_a, dev_b);

	cudaDeviceSynchronize();

	cudaError_t err = cudaPeekAtLastError();

	if (cudaSuccess != err)
	{
		printf("CUDA: ERROR: cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else
		printf("CUDA: success\n");

	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < SIZE; i++)
		printf("%d %d %d\n", a[i], b[i], c[i]);

	return 0;
}