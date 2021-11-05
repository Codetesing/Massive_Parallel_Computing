#include <stdio.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void big_Job(void)
{
	int count = 0;

	for (int i = 0; i < 10000; i++)
		for (int j = 0; j < 10000; j++)
			count++;

	printf("%d count.\n", count);
}

int main(void)
{
	system_clock::time_point chrono_begin = system_clock::now();

	big_Job();

	system_clock::time_point chrono_end = system_clock::now();

	microseconds elapsed_usec = duration_cast<microseconds>( chrono_end - chrono_begin );

	printf("elapsed time = %ld usec\n", (long)elapsed_usec.count());

	return 0;
}