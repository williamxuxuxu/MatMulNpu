#ifndef MATMUL_API_H
#define MATMUL_API_H

#if __cplusplus
	#include "matmul_C.h"
extern "C" {
#endif

void fill_mult(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_CV(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_naive(Ta* a, Tb* b, Tc* c, int m, int k, int n);

void fill_mult_irreg(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_cplx(Ta* a_r, Ta* a_i, std::vector<Tb*> b_r, std::vector<Tb*> b_i, Tc* c_r, Tc* c_i, 
				int m, int k, std::vector<int> n, bool AC_native, bool B_native);

void fill_random(T* data, int m, int n, U min, U max);

#if __cplusplus
}
#endif
#endif
	
	
