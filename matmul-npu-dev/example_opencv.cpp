#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/matrix_types/opencv_mat.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {

    int rows = 8192;
    int cols = 8192;
	
	bool AC_native = 1;
	bool B_native = 1;
	
    std::vector<int8_t> a(rows*cols, 1);

    std::vector<int8_t> b(rows*cols, 2);

    MatNpu A(rows, cols, CV_8S, a.data());
    MatNpu B(rows, cols, CV_8S, b.data());


    auto start = std::chrono::high_resolution_clock::now();

    MatNpu C = A.matmul(B, CV_32S, AC_native, B_native);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Time took: " << elapsed.count() / 1.0E9f << "s\n";

    std::cout << "The first item of the matrix: ";
    std::cout << C.at<int32_t>(0, 0) << "\n";
}
