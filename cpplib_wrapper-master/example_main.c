/**
 * This example program shows how to use a shared library (.so file) that
 * provides a C API for a C++ library.
 */

#include "wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

void fill_random(Ta* data, int m, int n, Ta min, Ta max)
{
	srand(time(NULL));

    for (int i = 0; i < m*n; i ++)
    {
        data[i] = (Ta) ((rand() % (max - min + 1)) + min);
    }
}

int main(int argc, char **argv)
{

	int trials = 10;
	int b_num = 5;
    int m = 1000;
    int k = 1000;
	int n = 1000;
	Ta min = -10;
	Ta max = 10;
	
	Ta* a = (Ta*) malloc(m * k * sizeof(Ta));
	Tb* b = (Tb*) malloc(k * n * sizeof(Tb));
	Tc* c_npu = (Tc*) malloc(m * n * sizeof(Tc));
	Tc* c_naive = (Tc*) malloc(m * n * sizeof(Tc));

	fill_random(a, m, k, min, max);
	fill_random(b, k, n, min, max);

    clock_t start_npu, end_npu;
    double time_npu;
     
    start_npu = clock();
    
	fill_mult_irreg(a, b, c_npu, m, k, n, 0, 1);

    end_npu = clock();
    time_npu = ((double) (end_npu - start_npu)) / CLOCKS_PER_SEC;
    
    clock_t start_cpu, end_cpu;
    double time_cpu;
    
    start_cpu = clock();

	fill_mult_naive(a, b, c_naive, m, k, n);
	
    end_cpu = clock();
    time_cpu = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    
//	print_cv(c_npu, c_naive, m, n);
	
	printf("NPU time: %f \t CPU time: %f \n", time_npu, time_cpu);

	free(a);
	free(b);
	free(c_npu);
	free(c_naive);
    return 0;
}

