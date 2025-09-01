/**
 * c_wrapper.h
 *
 * Declares a simple C API wrapper around a C++ class.
 *
 * This wrapper is compiled into a shared library (.so)
 * which can be then called from plain old C code.
 *
 * See an example program using this library in
 * applications/examples/cpplib_wrapper_example.c
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

typedef int8_t Ta;
typedef int8_t Tb;
typedef int32_t Tc;

void print_cv(Tc* c1, Tc* c2, int m, int n);

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
