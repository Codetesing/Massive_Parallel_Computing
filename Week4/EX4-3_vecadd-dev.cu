#include "./common.cpp"

const unsigned SIZE = 1024 * 1024;

__global__ void kernelVecAdd(float* c, const float* a, const float* b, unsigned n)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n)
		c[i] = a[i] + b[i];
}

int main(void)
{
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];

	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	float* dev_vecC = nullptr;

	srand(0);

	setNormalizedRandomData(vecA, SIZE);
	setNormalizedRandomData(vecB, SIZE);

	cudaMalloc((void**)&dev_vecA, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecB, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecC, SIZE * sizeof(float));
	CUDA_CHECK_ERROR();

	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy(dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vecB, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();

	ELAPSED_TIME_BEGIN(0);
	kernelVecAdd << < SIZE / 1024, 1024 >> > (dev_vecC, dev_vecA, dev_vecB, SIZE);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);

	CUDA_CHECK_ERROR();

	cudaMemcpy(vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();

	cudaFree(dev_vecA);
	cudaFree(dev_vecB);
	cudaFree(dev_vecC);
	CUDA_CHECK_ERROR();

	float sumA = getSum(vecA, SIZE);
	float sumB = getSum(vecB, SIZE);
	float sumC = getSum(vecC, SIZE);
	float diff = fabsf(sumC - (sumA + sumB));

	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA + sumB) = %f\n", diff);
	printf("diff(sumC, sumA + sumB) / SIZE = %f\n", diff / SIZE);

	for (int i = 0; i < 4; i++)
		printf("%d + %d = %d\n", vecA[i], vecB[i], vecC[i]);

	delete[]vecA;
	delete[]vecB;
	delete[]vecC;

	return 0;
}