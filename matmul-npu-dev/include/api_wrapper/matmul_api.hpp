#ifndef MATMUL_NPU
#define MATMUL_NPU

#include "/home/orangepi/Documents/Projects/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h"
#include <type_traits>
#include <iostream>
#include <cstring>
#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/utils/half.hpp"

// Defines the float16 and float32 types from half.hpp
using float16 = half_float::half;
typedef float float32;

/**
 * @brief Assigns the alignment parameters of the matrix multiplication
 * based on type. 
 */
 
template <typename Ti1, typename Ti2>
void type_align(int* k_align, int* n_align) 
{
	*k_align = 32;
	if (
        std::is_same<int8_t, Ti2>::value || 
        std::is_same<float16, Ti2>::value) {
		*n_align = 32;
    }
    else {
		*n_align = 64;
	}
}
/**
 * @brief convert norm layout to perf layout
 * norm layout: [M,K]
 * perf layout: [K/subK, M, subK]
 */
template <typename Ti, typename To>
void norm_layout_to_perf_layout(Ti *src, To *dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type)
{
	int outter_size = (int)std::ceil(K * 1.0f / subK);
	for (int i = 0; i < outter_size; i++)
	{
		for (int m = 0; m < M; m++)
		{
			for (int j = 0; j < subK; j++)
			{
				int ki = i * subK + j;
				if (isInt4Type)
				{
					int input_index = m * K + ki;
					int output_index = i * M * subK + m * subK + j;
					int8_t int4 = src[input_index];
					if (ki >= K)
					{
						int4 = 0;
					}
					else
					{
						int4 = int4 & 0xf;
					}
					if (output_index % 2 == 0)
					{
						dst[output_index / 2] = int4;
					}
					else
					{
						int8_t temp = dst[output_index / 2];
						int8_t result = temp | (int4 << 4);
						dst[output_index / 2] = result;
					}
				}
				else
				{
					if (ki >= K)
					{
						dst[i * M * subK + m * subK + j] = 0;
					}
					else
					{
						dst[i * M * subK + m * subK + j] = src[m * K + ki];
					}
				}
			}
		}
	}
}

template void norm_layout_to_perf_layout<int8_t, int8_t>(int8_t *src, int8_t *dst, int32_t M, int32_t K, int32_t subK,
														 bool isInt4Type);
template void norm_layout_to_perf_layout<float16, float16>(float16 *src, float16 *dst, int32_t M, int32_t K,
														   int32_t subK, bool isInt4Type);

/**
 * @brief convert norm layout to native layout
 * norm layout:  [K,N]
 * native layout: [N1, K1, subN, subK]
 *
 */
template <typename Ti, typename To>
void norm_layout_to_native_layout(Ti *src, To *dst, int32_t K, int32_t N, int32_t subN, int32_t subK, bool isInt4Type)
{
	int N_remain = (int)std::ceil(N * 1.0f / subN);
	int K_remain = (int)std::ceil(K * 1.0f / subK);
	for (int i = 0; i < N_remain; i++)
	{
		for (int j = 0; j < K_remain; j++)
		{
			for (int n = 0; n < subN; n++)
			{
				int ni = i * subN + n;
				for (int k = 0; k < subK; k++)
				{
					int ki = j * subK + k;
					if (isInt4Type)
					{
						int input_index = ki * N + ni;
						int output_index = i * (K_remain * subN * subK) + j * (subN * subK) + n * subK + k;
						int8_t int4 = src[input_index];
						if (ki < K && ni < N)
						{
							int4 = int4 & 0xf;
						}
						else
						{
							int4 = 0;
						}
						if (output_index % 2 == 0)
						{
							dst[output_index / 2] = int4 << 4;
						}
						else
						{
							int8_t temp = dst[output_index / 2];
							int8_t result = temp | int4;
							dst[output_index / 2] = result;
						}
					}
					else
					{
						if (ki < K && ni < N)
						{
							dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki * N + ni];
						}
						else
						{
							dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
						}
					}
				}
			}
		}
	}
}

template void norm_layout_to_native_layout<int8_t, int8_t>(int8_t *src, int8_t *dst, int32_t K, int32_t N, int32_t subN,
														   int32_t subK, bool isInt4Type);
template void norm_layout_to_native_layout<float16, float16>(float16 *src, float16 *dst, int32_t K, int32_t N,
															 int32_t subN, int32_t subK, bool isInt4Type);

/**
 * @brief convert perf to norm layout
 * perf layout: [K1, M, subK]
 * norm layout: [M,K]
 *
 */
template <typename Ti, typename To>
void perf_layout_to_norm_layout(Ti *src, To *dst, int32_t M, int32_t K, int32_t subK)
{
	int K_remain = (int)std::ceil(K * 1.0f / subK);
	for (int i = 0; i < K_remain; i++)
	{
		for (int j = 0; j < subK; j++)
		{
			for (int m = 0; m < M; m++)
			{
				int ki = i * subK + j;
				if (ki < K)
				{
					dst[m * K + ki] = src[i * M * subK + m * subK + j];
				}
			}
		}
	}
}

template void perf_layout_to_norm_layout<int8_t, int8_t>(int8_t *src, int8_t *dst, int32_t M, int32_t K,
														 int32_t subK);
template void perf_layout_to_norm_layout<int16_t, int16_t>(int16_t *src, int16_t *dst, int32_t M, int32_t K,
														   int32_t subK);
template void perf_layout_to_norm_layout<int32_t, int32_t>(int32_t *src, int32_t *dst, int32_t M, int32_t K,
														   int32_t subK);
template void perf_layout_to_norm_layout<float, float>(float *src, float *dst, int32_t M, int32_t K,
													   int32_t subK);
template void perf_layout_to_norm_layout<float16, float16>(float16 *src, float16 *dst, int32_t M, int32_t K,
														   int32_t subK);
														   								   
		
/**
 * @brief Utility function to choose flag from the _rknn_matmul_types
 * and assign the AC and B native parameters based on the variable type.
 * 
 * @param To    The type of the output matrix
 * @param Ti1   The type of the first input martix
 * @param Ti2   The type of the second input matrix
 * 
 * @return _rknn_matmul_ type flag for the matmul operation
 */
template<typename To, typename Ti1, typename Ti2>
_rknn_matmul_type choose_matmul_type(int* A_subK, int* B_subN, int* B_subK, 
									 int* C_subN, int* A_int4, int* B_int4) {

    if (
        std::is_same<float16, Ti1>::value && 
        std::is_same<float16, Ti2>::value && 
        std::is_same<float16, To>::value) {
			
		* A_subK = 8;
		* B_subN = 16;
		* B_subK = 32;
		* C_subN = 8;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    } else if (
        std::is_same<float16, Ti1>::value && 
        std::is_same<float16, Ti2>::value && 
        std::is_same<float32, To>::value)  {

		* A_subK = 8;
		* B_subN = 16;
		* B_subK = 32;
		* C_subN = 4;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    } else if (
        std::is_same<int16_t, Ti1>::value && 
        std::is_same<int16_t, Ti2>::value && 
        std::is_same<int32_t, To>::value)  {

		* A_subK = 8;
		* B_subN = 16;
		* B_subK = 32;
		* C_subN = 4;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    } else if (
        std::is_same<float16, Ti1>::value &&
        std::is_same<int8_t, Ti2>::value  &&
        std::is_same<float16, To>::value ) { 

		* A_subK = 8;
		* B_subN = 16;
		* B_subK = 32;
		* C_subN = 8;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_FLOAT16_MM_INT8_TO_FLOAT16; 
    } else if (
        std::is_same<int8_t, Ti1>::value && 
        std::is_same<int8_t, Ti2>::value && 
        std::is_same<int8_t, To>::value)  {
			
		* A_subK = 16;
		* B_subN = 32;
		* B_subK = 32;
		* C_subN = 16;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_INT8_MM_INT8_TO_INT8;
    } else if (
        std::is_same<int8_t, Ti1>::value && 
        std::is_same<int8_t, Ti2>::value && 
        std::is_same<int32_t, To>::value)  {
			
		* A_subK = 16;
		* B_subN = 32;
		* B_subK = 32;
		* C_subN = 4;
		* A_int4 = 0;
		* B_int4 = 0;
		
        return RKNN_INT8_MM_INT8_TO_INT32;
    } else {
        std::cout << "an unsupported combination of types:\n";
        std::cout << "please enter types from avilable types\n";
        std::cout << "1. float16, float16, float16\n";
        std::cout << "2. float16, float16, float32\n";
        std::cout << "3. float16, int8_t, float16\n";
        std::cout << "4. int8_t, int8_t, int8_t\n";
        std::cout << "5. int8_t, int8_t, int32_t\n";
        abort();
    }

}
/**
 * Struct that wraps all the built in rknn types 
 * and contains the result pointer
 * 
 * @param To The type of the output matrix
 */
struct _matmul_ctx {
    rknn_context ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_tensor_mem* matrixA;
    rknn_tensor_mem* matrixB;
    rknn_tensor_mem* matrixC;
};

/**
 * Struct that contains the result tensor of the matmul and it's context
 */
struct tensor_result {
    rknn_context ctx;
    rknn_tensor_mem* resultMatrix;

    tensor_result(rknn_context ctx, rknn_tensor_mem* resultMatrix) 
        : ctx(ctx), resultMatrix(resultMatrix) {}
};

/**
 * @brief ## __Create a matmul operation for the npu__
 * 
 * @param To The type of the output matrix
 * 
 * @return _matmul_ctx with the currect context for the rknn_matmul_run function
 */
_matmul_ctx* make_matmul(
    int32_t num_rows_a, int32_t num_cols_a, int32_t num_cols_b, _rknn_matmul_type type, bool AC_native, bool B_native
    ) {

    /* create a matmul_ctx struct */
    _matmul_ctx* matmul_ctx = (_matmul_ctx*)malloc(sizeof(_matmul_ctx));

    /* set all field to zero */
    memset(matmul_ctx, 0, sizeof(_matmul_ctx));

    matmul_ctx->info.M             = num_rows_a; /* set first matrix rows */
    matmul_ctx->info.K             = num_cols_a; /* set first matrix cols */
    matmul_ctx->info.N             = num_cols_b; /* set second matrix cols */
    matmul_ctx->info.type          = type; /* set the dtypes of the input and output matrices*/
    matmul_ctx->info.AC_layout     = AC_native; /* set the layout of matrices A and C */
    matmul_ctx->info.B_layout      = B_native; /* set the layout of matrix B */
    

    // create the matmul operation
    int ret = rknn_matmul_create(&matmul_ctx->ctx, &matmul_ctx->info, &matmul_ctx->io_attr);
    if (ret < 0) {
        printf("rknn_matmul_create fail! ret=%d\n", ret);
        abort();
    }

    // create the memory for the matrices in the npu
    matmul_ctx->matrixA = rknn_create_mem(matmul_ctx->ctx, matmul_ctx->io_attr.A.size);
    matmul_ctx->matrixB = rknn_create_mem(matmul_ctx->ctx, matmul_ctx->io_attr.B.size);
    matmul_ctx->matrixC = rknn_create_mem(matmul_ctx->ctx, matmul_ctx->io_attr.C.size);


    // set the memory in the npu
    rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->matrixA, &matmul_ctx->io_attr.A);
    rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->matrixB, &matmul_ctx->io_attr.B);
    rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->matrixC, &matmul_ctx->io_attr.C);

    return matmul_ctx;
}

/**
 * @brief Set the matrix data in the npu
 * 
 * @param Ti The type of the input matrix
 * @param ctx The context for the matmul operation
 * @param mem The information of the matrix tensor memory
 * @param attr The attributes of the matrix tensor
 */
void set_matrix_data(
    rknn_matmul_ctx* ctx, 
    rknn_tensor_mem* mem, 
    rknn_matmul_tensor_attr* attr, 
    const void* data ) {

    memcpy(mem->virt_addr, data, mem->size);
    rknn_matmul_set_io_mem(*ctx, mem, attr);
}

/**
 * @brief Free the matrices tensors 
 * 
 * @param To The type of the input matrix
 * @param ctx The context of the matmul operation
 */
void free_matmul(_matmul_ctx* ctx) {
    rknn_destroy_mem(ctx->ctx, ctx->matrixA);
    rknn_destroy_mem(ctx->ctx, ctx->matrixB);
    free(ctx);
}

/**
 * @brief Performs matrix multiplication on the npu 
 * 
 * @param To - The type of the output matrix 
 * @param Ti1 - The type of the first input matrix (inferred automatically) 
 * @param Ti2 - The type of the second input matrix (inferred automatically) 
 * @param num_rows_a The number of rows in the first input mat
 * @param num_cols_a The number of columns in the first input mat
 * @param num_cols_b The number of columns in the second input mat
 * @param a The data of the first input matrix 
 * @param b The data of the second input matrix 
 * 
 * @return _matmul_ctx<To> that has inside the pointer to the result of the matmul.
 * 
 * @note The shape of the result is (num_rows_a, num_cols_b)
 */

template<typename To, typename Ti1, typename Ti2> 
tensor_result matmul_npu(
    uint32_t m,
    uint32_t k,
    uint32_t n,
    Ti1* a,
    Ti2* b,
	bool AC_native,
	bool B_native
) {
	// Assigns the AC and B native, high performance data format parameters.
	int A_subK, B_subN, B_subK,
		C_subN, A_int4, B_int4;

	_rknn_matmul_type mul_type = choose_matmul_type<To, Ti1, Ti2>
								 (&A_subK, &B_subN, &B_subK,
								  &C_subN, &A_int4, &B_int4);

	// Initializes the intended and performance pointers for matrix A and B.
	Ti1* a_intend = a;
	Ti2* b_intend = b;
	
	Ti1* a_perf = 0;
	Ti2* b_perf = 0;
	
	// Allocates and performs the high performance data type transformation
	if (AC_native) {
		a_perf = (Ti1*) malloc(m * k * sizeof(Ti1));
		norm_layout_to_perf_layout(a, a_perf, m, k, A_subK, A_int4);
		a_intend = a_perf;
	}
	if (B_native) {
		b_perf = (Ti2*) malloc(k * n * sizeof(Ti2));
		norm_layout_to_native_layout(b, b_perf, k, n, B_subN, B_subK, B_int4);
		b_intend = b_perf;
	}
	
	// Creates and executes the matmul environment. 
    _matmul_ctx* ctx = make_matmul(
        m, k, n, mul_type, AC_native, B_native
    );

    set_matrix_data(&ctx->ctx, ctx->matrixA, &ctx->io_attr.A, a_intend);
    set_matrix_data(&ctx->ctx, ctx->matrixB, &ctx->io_attr.B, b_intend);
    rknn_matmul_run(ctx->ctx);

    tensor_result result(ctx->ctx, ctx->matrixC);
    
    // Restores C to the original matrix layout if applicable. 
	if (AC_native) {
		To* tmp = (To*) malloc(m * n * sizeof(To));
		perf_layout_to_norm_layout((To*) result.resultMatrix->virt_addr, tmp, m, n, C_subN);
		memcpy((To*) result.resultMatrix->virt_addr, tmp, m*n*sizeof(To));
		free(tmp);

	}
	
	// Frees assigned memory. 
	if (AC_native) {
		free(a_perf);
	}
	if (B_native) { 
		free(b_perf);
	}
    free_matmul(ctx);

    return result;

}


/**
 * @brief Performs matrix multiplication on the npu 
 * 
 * @param num_rows_a The number of rows in the first input mat
 * @param num_cols_a The number of columns in the first input mat
 * @param num_cols_b The number of columns in the second input mat
 * @param type The matmul type flag
 * @param a The data of the first input matrix 
 * @param b The data of the second input matrix 
 * 
 * @return _matmul_ctx<To> that has inside the pointer to the result of the matmul.
 * 
 * @note The shape of the result is (num_rows_a, num_cols_b)
 */
tensor_result matmul_npu(
    uint32_t m,
    uint32_t k,
    uint32_t n,
    _rknn_matmul_type type,
    void* a,
    void* b,
	bool AC_native,
	bool B_native
) {
	// Passes the correct types to the matmul_npu function. 
	if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16) {
		return matmul_npu<float16, float16, float16>(
                m, k, n, (float16*) a, (float16*) b, AC_native, B_native);
	} else if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
		return matmul_npu<float32, float16, float16>(
                m, k, n, (float16*) a, (float16*) b, AC_native, B_native);
	}		
	else if (type == RKNN_FLOAT16_MM_INT8_TO_FLOAT16) {
		return matmul_npu<float16, float16, int8_t>(
                m, k, n, (float16*) a, (int8_t*) b, AC_native, B_native);
	}
	else if (type == RKNN_INT8_MM_INT8_TO_INT8) {
		return matmul_npu<int8_t, int8_t, int8_t>(
                m, k, n, (int8_t*) a, (int8_t*) b, AC_native, B_native);
	}
	else if (type == RKNN_INT8_MM_INT8_TO_INT32) {
		return matmul_npu<int32_t, int8_t, int8_t>(
                m, k, n, (int8_t*) a, (int8_t*) b, AC_native, B_native);
	}
	else {
        std::cout << "an unsupported combination of types:\n";
        std::cout << "please enter types from avilable types\n";
        std::cout << "1. float16, float16, float16\n";
        std::cout << "2. float16, float16, float32\n";
        std::cout << "3. float16, int8_t, float16\n";
        std::cout << "4. int8_t, int8_t, int8_t\n";
        std::cout << "4. int8_t, int8_t, int32_t\n";
        abort();
	}
}


#endif

