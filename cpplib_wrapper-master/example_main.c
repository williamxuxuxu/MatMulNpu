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
    for (int i = 0; i < m*n; i ++)
    {
        data[i] = (Ta) ((rand() % (max - min + 1)) + min);
    }
}

int main(int argc, char **argv)
{
	srand(time(NULL));
	int trials = 5;
//	int b_num = 2;
	
    int m[] = {512, 1024, 2048, 4096};
    int k[] = {512, 1024, 2048, 4096};
	int n[] = {512, 1024, 2048, 4096};
	
	Ta min = -10;
	Ta max = 10;
	
	int best_AC;
	int best_B;
	double best_time;
	
	FILE* fptr = fopen("results.txt", "w");
	fprintf(fptr, "m \t \t k \t \t n \t \t AC \t \t B \t \t NPU time \t \t CPU time \n");

	for (int i = 0; i < sizeof(m)/sizeof(int); i++) {
		for (int j = 0; j < sizeof(k)/sizeof(int); j ++) {
			for (int l = 0; l < sizeof(n)/sizeof(int); l ++) {

				double avg_time_cpu = 0;
				double best_time = 0;
				int best_AC = 0;
				int best_B = 0;

				for (int AC_native = 0; AC_native < 2; AC_native ++) {
					for (int B_native = 0; B_native < 2; B_native ++) {
						
						double avg_time = 0;

						for (int t = 0; t < trials; t ++) {
							Ta* a = (Ta*) malloc(m[i] * k[j] * sizeof(Ta));
							Tb* b = (Tb*) malloc(k[j] * n[l] * sizeof(Tb));
							Tc* c_npu = (Tc*) malloc(m[i] * n[l] * sizeof(Tc));
							Tc* c_cpu = (Tc*) malloc(m[i] * n[l] * sizeof(Tc));

							fill_random(a, m[i], k[j], min, max);
							fill_random(b, k[j], n[l], min, max);
							
							clock_t start_npu, end_npu;
							 
							start_npu = clock();				
							
							fill_mult_irreg(a, b, c_npu, m[i], k[j], n[l], AC_native, B_native);
							
							end_npu = clock();
							
							avg_time += ((double) (end_npu - start_npu)) / CLOCKS_PER_SEC;
							
							clock_t start_cpu, end_cpu;
							 
							start_cpu = clock();				
							
							fill_mult_naive(a, b, c_cpu, m[i], k[j], n[l]);
							
							end_cpu = clock();
							
							avg_time_cpu += ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
							
							c_comp(c_npu, c_cpu, m[i], n[l]);
							
							free(a);
							free(b);
							free(c_npu);
							free(c_cpu);
						}
						avg_time = avg_time / trials;
						
						if (best_time == 0 || avg_time < best_time) {
							best_time = avg_time;
							best_AC = AC_native;
							best_B = B_native;
						}
						
						avg_time_cpu = avg_time_cpu / trials;

					}
				}
				fprintf(fptr, "%d \t \t %d \t \t %d \t \t %d \t \t %d \t\t %lf \t \t %lf\n", m[i], k[j], n[l], best_AC, best_B, best_time, avg_time_cpu);
				printf("%d \t \t %d \t \t %d \t \t %d \t \t %d \t\t %lf \t \t %lf\n", m[i], k[j], n[l], best_AC, best_B, best_time, avg_time_cpu);
			}
		}
	}
	fclose(fptr);
/*
	Ta* a_r = (Ta*) malloc(m * k * sizeof(Ta));
	Ta* a_i = (Ta*) malloc(m * k * sizeof(Ta));

	fill_random(a_r, m, k, min, max);
	fill_random(a_i, m, k, min, max);
		
	Tb** b_r_vec = (Tb**) malloc(b_num * sizeof(Tb*));
	Tb** b_i_vec = (Tb**) malloc(b_num * sizeof(Tb*));
	int* n_vec = (int*) malloc(b_num * sizeof(int));

	int n_sum = 0;
	for (int i = 0; i < b_num; i ++) {
		b_r_vec[i] = (Tb*) malloc(k * n * sizeof(Tb));
		b_i_vec[i] = (Tb*) malloc(k * n * sizeof(Tb));

		fill_random(b_r_vec[i], k, n, min, max);
		fill_random(b_i_vec[i], k, n, min, max);	
		
		n_vec[i] = n;
		n_sum += n;
	}
	
	Tc* c_r = (Tc*) malloc(m * n_sum * sizeof(Tc));
	Tc* c_i = (Tc*) malloc(m * n_sum * sizeof(Tc));

    clock_t start_npu, end_npu;
    double time_npu;
     
    start_npu = clock();
    
	fill_mult_cplx(a_r, a_i, b_r_vec, b_i_vec, c_r, c_i, m, k, n_vec, b_num, 0, 1);

    end_npu = clock();
    time_npu = ((double) (end_npu - start_npu)) / CLOCKS_PER_SEC;
    	
	printf("NPU time: %f \n", time_npu);

	free(a_r);
	free(a_i);
	for (int i = 0; i < b_num; i ++) {
		free(b_r_vec[i]);
		free(b_i_vec[i]);
	}
	free(b_r_vec);
	free(b_i_vec);
	free(n_vec);
	free(c_r);
	free(c_i);
*/
	
    return 0;
}

