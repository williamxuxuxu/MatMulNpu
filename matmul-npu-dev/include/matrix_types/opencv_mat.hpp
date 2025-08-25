#ifndef MATNPU
#define MATNPU

#include <opencv2/opencv.hpp>
#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/api_wrapper/matmul_api.hpp"
#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/utils/choose_type.hpp"


class MatNpu : public cv::Mat {

    private: 
        rknn_tensor_mem* tensor_mem;
        rknn_context ctx;

        MatNpu(int32_t rows, int32_t cols, int32_t type, void* data, 
                rknn_tensor_mem* tensor_mem, rknn_context ctx) 
            : cv::Mat(rows, cols, type, data), tensor_mem(tensor_mem), ctx(ctx) {}
    
    public: 

        MatNpu(int32_t rows, int32_t cols, int32_t type, void* data) 
            : cv::Mat(rows, cols, type, data), tensor_mem(nullptr), ctx(0) {}


        ~MatNpu() {
            rknn_destroy_mem(ctx, tensor_mem);
            rknn_matmul_destroy(ctx);
        }
        
        MatNpu matmul(MatNpu mat, int32_t output_type, bool AC_native, bool B_native) {
            _rknn_matmul_type mm_type = choose_matmul_type(this->type(), mat.type(), output_type);
			
			int m = rows;
			int k = cols;
			int n = mat.cols;
			Ta* a_intend = (Ta*) data;
			Tb* b_intend = (Ta*) mat.data;
			Ta* a_perf = 0;
			Tb* b_perf = 0;
			
			if (AC_native) {
				a_perf = (Ta*) malloc(m * k * sizeof(Ta));
				norm_layout_to_perf_layout((Ta*) this->data, a_perf, m, k, A_subK, A_int4);
				a_intend = a_perf;
			}
			if (B_native) {
				b_perf = (Tb*) malloc(k * n * sizeof(Tb));
				norm_layout_to_native_layout((Tb*) mat.data, b_perf, k, n, B_subN, B_subK, B_int4);
				b_intend = b_perf;
			}
			
            tensor_result result = matmul_npu(rows, cols, mat.cols, mm_type, a_intend, b_intend, AC_native, B_native);
			
			if (AC_native) {
				Tc* tmp = (Tc*) malloc(m * n * sizeof(Tc));
				perf_layout_to_norm_layout((Tc*) result.resultMatrix->virt_addr, tmp, m, n, C_subN);
				memcpy((Tc*) result.resultMatrix->virt_addr, tmp, m*n*sizeof(Tc));
				free(tmp);

			}
			if (AC_native) {
				free(a_perf);
			}
			if (B_native) { 
				free(b_perf);
			}
            
            return MatNpu(
                rows, mat.cols, output_type, result.resultMatrix->virt_addr,
                result.resultMatrix, result.ctx
            );
        }
};

#endif
