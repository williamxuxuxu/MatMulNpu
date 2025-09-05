#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/matrix_types/opencv_mat.hpp"
#include "/home/orangepi/Documents/Projects/matmul-npu-dev/include/matrix_types/matrix.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <span>
#include <cstring>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h> // Include for CBLAS functions
#include <thread>
#include <mutex>
#include <typeinfo>
#include <opencv2/opencv.hpp>

void fill_mult(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native)

{

	Matrix<Ta> A(m, k, a);
	Matrix<Tb> B(k, n, b);
	Matrix<Tc> C = A.matmul<Tc>(B, AC_native, B_native);

	memcpy(c, C.data, m*n*sizeof(Tc));

}

void fill_mult_CV(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native)
{
	MatNpu A(m, k, CV_a, a);
	MatNpu B(k, n, CV_b, b);
	MatNpu C = A.matmul(B, CV_c, AC_native, B_native);

	memcpy(c, C.data, m*n*sizeof(Tc));
}

void fill_mult_naive(Ta* a, Tb* b, Tc* c, int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Tc sum = static_cast<Tc>(0);
            for (int l = 0; l < k; l++) {
                sum += static_cast<Tc>(a[i * k + l] * b[l * n + j]);
            }
            c[i * n + j] = sum;
        }
	}
}

void fill_mult_irreg(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native)
{
	
	int k_fill = (int)std::ceil(k * 1.0f / k_align) * k_align;
	int n_fill = (int)std::ceil(n * 1.0f / n_align) * n_align;

	int k_diff = k_fill - k;
	int n_diff = n_fill - n;
		
	MatNpu A(m, k, CV_a, a);
	MatNpu B(k, n, CV_b, b);
	
	if (k_diff > 0) {
		cv::Mat A_k_zeros = cv::Mat::zeros(m, k_diff, CV_a);
		cv::hconcat(A, A_k_zeros, A);
		if (n_fill - n > 0) {
			cv::Mat B_k_zeros = cv::Mat::zeros(k_diff, n, CV_b);
			cv::Mat B_n_zeros = cv::Mat::zeros(k_fill, n_diff, CV_b);
			cv::vconcat(B, B_k_zeros, B);
			cv::hconcat(B, B_n_zeros, B);
		}
		else {
			cv::Mat B_k_zeros = cv::Mat::zeros(k_diff, n, CV_b);
			cv::vconcat(B, B_k_zeros, B);
		}
	}
	else {
		if (n_diff > 0) {
			cv::Mat B_n_zeros = cv::Mat::zeros(k_fill, n_diff, CV_b);
			cv::hconcat(B, B_n_zeros, B);
		}
	}
	
	MatNpu C_fill = A.matmul(B, CV_c, AC_native, B_native);
	if (n_diff > 0) {
		cv::Mat tmp = C_fill(cv::Rect(0, 0, n, m));
		tmp.copyTo(C_fill);
	}
	memcpy(c, C_fill.data, m*n*sizeof(Tc));

}


void fill_mult_cplx(Ta* a_r, Ta* a_i, std::vector<Tb*> b_r, std::vector<Tb*> b_i, Tc* c_r, Tc* c_i, 
				int m, int k, std::vector<int> n, bool AC_native, bool B_native)
{
	int n_sum = n[0];
	cv::Mat B_r_final = cv::Mat(k, n[0], CV_b, b_r[0]);
	cv::Mat B_i_final = cv::Mat(k, n[0], CV_b, b_i[0]);

	for (int i = 1; i < (int) n.size(); i ++) {
		n_sum += n[i];
		cv::hconcat(B_r_final, cv::Mat(k, n[i], CV_b, b_r[i]), B_r_final);
		cv::hconcat(B_i_final, cv::Mat(k, n[i], CV_b, b_i[i]), B_i_final);
	}
	
	Tc* A_rB_r = (Tc*) malloc(m * n_sum * sizeof(Tc));
	Tc* A_rB_i = (Tc*) malloc(m * n_sum * sizeof(Tc));
	Tc* A_iB_r = (Tc*) malloc(m * n_sum * sizeof(Tc));
	Tc* A_iB_i = (Tc*) malloc(m * n_sum * sizeof(Tc));

	std::vector<std::thread> thr;
	thr.emplace_back(fill_mult_irreg, a_r, (Tb*) B_r_final.data, A_rB_r, m, k, n_sum, AC_native, B_native);
	thr.emplace_back(fill_mult_irreg, a_r, (Tb*) B_i_final.data, A_rB_i, m, k, n_sum, AC_native, B_native);
	thr.emplace_back(fill_mult_irreg, a_i, (Tb*) B_r_final.data, A_iB_r, m, k, n_sum, AC_native, B_native);
	thr.emplace_back(fill_mult_irreg, a_i, (Tb*) B_i_final.data, A_iB_i, m, k, n_sum, AC_native, B_native);

	for (std::thread& t : thr) {
		t.join();
	}
	
	cv::Mat C_r_mat = cv::Mat(m, n_sum, CV_c, A_rB_r) - cv::Mat(m, n_sum, CV_c, A_iB_i);
	cv::Mat C_i_mat = cv::Mat(m, n_sum, CV_c, A_rB_i) + cv::Mat(m, n_sum, CV_c, A_iB_r);
	
	memcpy(c_r, C_r_mat.data, m * n_sum * sizeof(Tc));
	memcpy(c_i, C_i_mat.data, m * n_sum * sizeof(Tc));

	free(A_rB_r);
	free(A_rB_i);
	free(A_iB_r);
	free(A_iB_i);
	thr.clear();
}


template <typename T, typename U>
void fill_random(T* data, int m, int n, U min, U max)
{
    using Dist = std::uniform_real_distribution<double_t>;
    std::random_device rd;
    std::mt19937 gen(rd());
    Dist dis(min, max);
    for (int i = 0; i < m*n; i ++)
    {
        data[i] = static_cast<T>(dis(gen));
    }
}
