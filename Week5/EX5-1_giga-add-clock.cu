#include "./common.cpp"

const unsigned SIZE = 256 * 1024 * 1024; // 256 elements

// CUDA kernel function
__global__ void kernelVecAdd(float* c, const float* a, const float* b, unsigned n, long long* times) {
	clock_t start = clock();
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		c[i] = a[i] + b[i];
	}
	clock_t end = clock();
	if (i == 0) {
		times[0] = (long long)(end - start);
	}
}


int main(void) {
	// host-side data
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];
	long long* host_times = new long long[1];
	// set random data
	srand(0);
	setNormalizedRandomData(vecA, SIZE);
	setNormalizedRandomData(vecB, SIZE);
	// device-side data
	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	float* dev_vecC = nullptr;
	long long* dev_times = nullptr;
	// allocate device memory
	cudaMalloc((void**)&dev_vecA, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecB, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecC, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_times, 1 * sizeof(long long));
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy(dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1);
	CUDA_PRINT_CONFIG(SIZE);
	ELAPSED_TIME_BEGIN(0);
	kernelVecAdd << < dimGrid, dimBlock >> > (dev_vecC, dev_vecA, dev_vecB, SIZE, dev_times);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy(vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_times, dev_times, 1 * sizeof(long long), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree(dev_vecA);
	cudaFree(dev_vecB);
	cudaFree(dev_vecC);
	cudaFree(dev_times);
	CUDA_CHECK_ERROR();
	// kernel clock calculation
	int peak_clk = 1;
	cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
	printf("num clock = %lld, peak clock rate = %dkHz, elapsed time: %f usec\n",
		host_times[0], peak_clk, host_times[0] * 1000.0f / (float)peak_clk);
	// check the result
	float sumA = getSum(vecA, SIZE);
	float sumB = getSum(vecB, SIZE);
	float sumC = getSum(vecC, SIZE);
	float diff = fabsf(sumC - (sumA + sumB));
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);
	printVec("vecA", vecA, SIZE);
	printVec("vecB", vecB, SIZE);
	printVec("vecC", vecC, SIZE);
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	delete[] host_times;
	// done
	return 0;
}
