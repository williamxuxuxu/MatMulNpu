#ifndef MATRIX
#define MATRIX

#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/api_wrapper/matmul_api.hpp"
#include <memory>

template <typename T>
class Matrix {
        
    private:
	
        rknn_tensor_mem* tensor_mem;
        rknn_context ctx;

    public: 

        int rows, cols;
        T* data; 

        Matrix() = default;

        Matrix(int rows, int cols, T* data) 
        : tensor_mem(nullptr), ctx(0), rows(rows), cols(cols), data(data) {}

        Matrix(rknn_tensor_mem* tensor_mem, rknn_context ctx, int rows, int cols, T* data) 
        : tensor_mem(tensor_mem), ctx(ctx), rows(rows), cols(cols), data(data) {}

        ~Matrix() {
            rknn_destroy_mem(ctx, tensor_mem);
            rknn_matmul_destroy(ctx);
        }
        
        template<typename To, typename Ti>
        Matrix<To> matmul(Matrix<Ti> mat, bool AC_native, bool B_native) {
			
			int m = this->rows;
			int k = this->cols;
			int n = mat.cols;
			T* a_intend = this->data;
			Ti* b_intend = mat.data;
			T* a_perf = 0;
			Ti* b_perf = 0;
			
			if (AC_native) {
				a_perf = (T*) malloc(m * k * sizeof(T));
				norm_layout_to_perf_layout((T*) this->data, a_perf, m, k, A_subK, A_int4);
				a_intend = a_perf;
			}
			if (B_native) {
				b_perf = (Ti*) malloc(k * n * sizeof(Ti));
				norm_layout_to_native_layout((Ti*) mat.data, b_perf, k, n, B_subN, B_subK, B_int4);
				b_intend = b_perf;
			}
			
            tensor_result result = matmul_npu<To, T, Ti>(
                m, k, n, a_intend, b_intend, AC_native, B_native
            );
			
			if (AC_native) {
				To* tmp = (To*) malloc(m * n * sizeof(To));
				perf_layout_to_norm_layout((To*) result.resultMatrix->virt_addr, tmp, m, n, C_subN);
				memcpy((To*) result.resultMatrix->virt_addr, tmp, m*n*sizeof(To));
				free(tmp);

			}
			if (AC_native) {
				free(a_perf);
			}
			if (B_native) { 
				free(b_perf);
			}
            
            return Matrix<To>(
                result.resultMatrix, 
                result.ctx,
                rows, mat.cols, 
                (To*) result.resultMatrix->virt_addr
            ); 
    } 



};

#endif
