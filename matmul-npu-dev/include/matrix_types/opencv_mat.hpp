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
            tensor_result result = matmul_npu(rows, cols, mat.cols, mm_type, data, mat.data, AC_native, B_native);
            return MatNpu(
                rows, mat.cols, output_type, result.resultMatrix->virt_addr,
                result.resultMatrix, result.ctx
            );
        }
};

#endif
