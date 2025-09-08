/**
 * c_wrapper.h
 *
 * Declares a simple C API wrapper around a C++ class.
 *
 * This wrapper is compiled into a shared library (.so)
 * which can be then called from plain old C code.
 * 
 */

#ifndef C_WRAPPER_H_
#define C_WRAPPER_H_


#ifdef __cplusplus
extern "C" {
#else
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "stdint.h"

// Matrix types 
typedef int8_t Ta;
typedef int8_t Tb;
typedef int32_t Tc;

void print_A_cv(Ta* a, int m, int n);

void print_C_cv(Tc* c1, int m, int n);

void c_comp(Tc* c, Tc* c2, int m, int n);

void fill_mult(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_CV(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_naive(Ta* a, Tb* b, Tc* c, int m, int k, int n);

void fill_mult_irreg(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native);

void fill_mult_cplx(Ta* a_r, Ta* a_i, Tb** b_r, Tb** b_i, Tc* c_r, Tc* c_i, 
				int m, int k, int* n, int n_size, bool AC_native, bool B_native);

#ifdef __cplusplus
}
#endif


#endif /* C_WRAPPER_H_ */
