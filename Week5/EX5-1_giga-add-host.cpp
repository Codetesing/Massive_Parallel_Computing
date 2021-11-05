#include "./common.cpp"

const unsigned SIZE = 256 * 1024 * 1024;

void setRandomData(float* dst, int size)
{
	while (size--)
		*dst++ = (rand() % 1000) / 1000.0F;
}

float getSum(float* dst, int size)
{
	register float sum = 0.0F;

	while (size--)
		sum += *dst++;

	return sum;
}

int main(void)
{
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];

	srand(0);

	setRandomData(vecA, SIZE);
	setRandomData(vecB, SIZE);

	chrono::system_clock::time_point time_begin = chrono::system_clock::now();

	for (register unsigned i = 0; i < SIZE; ++i)
		vecC[i] = vecA[i] + vecB[i];

	chrono::system_clock::time_point time_end = chrono::system_clock::now();

	chrono::microseconds time_elapsed_msec = chrono::duration_cast<chrono::microseconds>(time_end - time_begin);

	printf("elapsed wall-clock time = %ld usec\n", (long)time_elapsed_msec.count());

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
		printf("%f + %f = %f\n", vecA[i], vecB[i], vecC[i]);

	delete[]vecA;
	delete[]vecB;
	delete[]vecC;

	return 0;
}