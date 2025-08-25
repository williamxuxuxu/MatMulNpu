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
/*
	Ta* a_intend = a;
	Tb* b_intend = b;
	Ta* a_perf = 0;
	Tb* b_perf = 0;
	
	if (AC_native) {
		a_perf = (Ta*) malloc(m * k * sizeof(Ta));
		norm_layout_to_perf_layout(a, a_perf, m, k, A_subK, A_int4);
		a_intend = a_perf;
	}
	if (B_native) {
		b_perf = (Tb*) malloc(k * n * sizeof(Tb));
		norm_layout_to_native_layout(b, b_perf, k, n, B_subN, B_subK, B_int4);
		b_intend = b_perf;
	}
*/
	Matrix<Ta> A(m, k, a);
	Matrix<Tb> B(k, n, b);
	Matrix<Tc> C = A.matmul<Tc>(B, AC_native, B_native);
/*
	if (AC_native) {
		perf_layout_to_norm_layout(C.data, c, m, n, C_subN);
	}
	else {
	*/
		memcpy(c, C.data, m*n*sizeof(Tc));
		/*
	}
	if (AC_native) {
		free(a_perf);
	}
	if (B_native) { 
		free(b_perf);
	}
	*/
}

void fill_mult_CV(Ta* a, Tb* b, Tc* c, int m, int k, int n, bool AC_native, bool B_native)
{
	/*
	Ta* a_intend = a;
	Tb* b_intend = b;
	Ta* a_perf = 0;
	Tb* b_perf = 0;
	
	if (AC_native) {
		a_perf = (Ta*) malloc(m * k * sizeof(Ta));
		norm_layout_to_perf_layout(a, a_perf, m, k, A_subK, A_int4);
		a_intend = a_perf;
	}
	if (B_native) {
		b_perf = (Tb*) malloc(k * n * sizeof(Tb));
		norm_layout_to_native_layout(b, b_perf, k, n, B_subN, B_subK, B_int4);
		b_intend = b_perf;
	}
*/
	MatNpu A(m, k, CV_a, a);
	MatNpu B(k, n, CV_b, b);
	MatNpu C = A.matmul(B, CV_c, AC_native, B_native);
/*
	if (AC_native) {
		perf_layout_to_norm_layout((Tc*) C.data, c, m, n, C_subN);
	}
	else {
	*/
		memcpy(c, C.data, m*n*sizeof(Tc));
		/*
	}
	if (AC_native) {
		free(a_perf);
	}
	if (B_native) { 
		free(b_perf);
	}
	*/
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

int main() {
/*
    // DGEMM parameters
    double alpha = 1.0; // Scalar for A*B
    double beta = 0.0;  // Scalar for C (set to 0.0 for C = A*B)
*/    
//	int threads = 1;
	int trials = 10;
	int b_num = 5;
    int m = 128;
    int k = 1024;
	int n = 256;
	Ta min = static_cast<Ta>(-10.0);
	Ta max = static_cast<Ta>(10.0);

/*
	auto avg_n_par = 0.0;
	auto avg_n_seq = 0.0;
	auto avg_cv_par = 0.0;
	auto avg_cv_seq = 0.0;
*/

	for (int AC_native = 0; AC_native < 2 ; AC_native ++) {
		for (int B_native = 0; B_native < 2; B_native ++) {
			auto avg_n_par = 0.0;
			for (int i = 0; i < trials; i ++) {
				
				auto a_r = (Ta*) malloc(m * k * sizeof(Ta));
				auto a_i = (Ta*) malloc(m * k * sizeof(Ta));

				fill_random(a_r, m, k, min, max);
				fill_random(a_i, m, k, min, max);

//				std::cout << "A: \n" << cv::Mat(m, k, CV_a, a_r) << "\n" << cv::Mat(m, k, CV_a, a_i) << "\n";

				std::vector<Tb*> b_r_vec(b_num);
				std::vector<Tb*> b_i_vec(b_num);
				std::vector<int> n_vec(b_num);
				
				int n_sum = 0;
				
				for (int i = 0; i < b_num; i ++) {
					b_r_vec[i] = (Tb*) malloc(k * n * sizeof(Tb));
					b_i_vec[i] = (Tb*) malloc(k * n * sizeof(Tb));
					
					fill_random(b_r_vec[i], k, n, min, max);
					fill_random(b_i_vec[i], k, n, min, max);
					n_vec[i] = n;
					n_sum += n_vec[i];
//					std::cout << "B" << i << ": \n" << cv::Mat(k, n_vec[i], CV_b, b_r_vec[i]) << "\n" << cv::Mat(k, n_vec[i], CV_b, b_i_vec[i]) << "\n";

				}

				auto c_r = (Tc*) malloc(m * n_sum * sizeof(Tc));
				auto c_i = (Tc*) malloc(m * n_sum * sizeof(Tc));

//				auto c_naive = (Tc*) malloc(m * n * sizeof(Tc));

//				fill_mult_naive(a, b, c_naive, m, k, n);

				auto start_n_par = std::chrono::high_resolution_clock::now();

				fill_mult_cplx(a_r, a_i, b_r_vec, b_i_vec, c_r, c_i, m, k, n_vec, AC_native, B_native);	
			
				auto end_n_par = std::chrono::high_resolution_clock::now();
				auto elapsed_n_par = std::chrono::duration_cast<std::chrono::nanoseconds>(end_n_par - start_n_par);
				avg_n_par += elapsed_n_par.count();

//				std::cout << "C: \n" << cv::Mat(m, n_sum, CV_c, c_r) << "\n" << cv::Mat(m, n_sum, CV_c, c_i) << "\n";

//				MatNpu C(m, n, CV_c, c);
//				MatNpu C_naive(m, n, CV_c, c_naive);

//				std::cout << cv::norm(C - C_naive, cv::NORM_L2) << "\n";
				free(a_r);
				free(a_i);
				for (int i = 0; i < b_num; i ++) {
					free(b_r_vec[i]);
					free(b_i_vec[i]);
				}
				free(c_r);
				free(c_i);
				b_r_vec.clear();
				b_i_vec.clear();
//				free(c_naive);
			}
			avg_n_par = avg_n_par/double(trials);
			std::cout << "AC Native: " << AC_native << "\t B Native: " << B_native << "\t Average time: " << avg_n_par << "\n";
		}
	}
	
/* CBLAS_DGEMM
    auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < threads; i ++) {
		cblas_gemm_s8s8s32(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a_thr[i], m, b_thr[i], k, beta, c_thr[i], m);;
	}
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "CPU: Time took in sequence: " << elapsed.count() / 1.0E9f << "s\n";

	for (int i = 0; i < threads; i ++) {
		std::cout << c_thr[i][0] << "\n";
	}


	for (int t = 0; t < trials; t ++) {
		std::vector<std::thread> thr;
		std::vector<Ta*> a_thr(threads);
		std::vector<Tb*> b_thr(threads);
		std::vector<Tc*> c_NPU_par(threads);
		std::vector<Tc*> c_NPU_seq(threads);
		std::vector<Tc*> c_CV_par(threads);
		std::vector<Tc*> c_CV_seq(threads);

		for (int i = 0; i < threads; i ++) {
			a_thr[i] = (Ta*) malloc(m * k * sizeof(Ta));
			b_thr[i] = (Tb*) malloc(k * n * sizeof(Tb));
			c_NPU_par[i] = (Tc*) malloc(m * n * sizeof(Tc));
			c_NPU_seq[i] = (Tc*) malloc(m * n * sizeof(Tc));
			c_CV_par[i] = (Tc*) malloc(m * n * sizeof(Tc));
			c_CV_seq[i] = (Tc*) malloc(m * n * sizeof(Tc));

			fill_random(a_thr[i], m, k, min, max);
			fill_random(b_thr[i], k, n, min, max);
		}

	// START OF NPU PARALLEL 
		for (int i = 0; i < threads; i ++) {
			thr.emplace_back(fill_mult, a_thr[i], b_thr[i], c_NPU_par[i], m, k, n, AC_native, B_native);
		}

		auto start_n_par = std::chrono::high_resolution_clock::now();
		for (std::thread& t : thr) {
			t.join();
		}
		auto end_n_par = std::chrono::high_resolution_clock::now();
		auto elapsed_n_par = std::chrono::duration_cast<std::chrono::nanoseconds>(end_n_par - start_n_par);
		avg_n_par += elapsed_n_par.count();

		std::cout << "NPU Parallel Time: " << elapsed_n_par.count() / 1.0E9f << "s\n";

		for (int i = 0; i < threads; i ++) {
			std::cout << c_NPU_par[i][0] << "\n";
		}

		
	// END OF NPU PARALLEL 

	// START OF NPU SEQUENCE
		auto start_n_seq = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < threads; i ++) {
			fill_mult(a_thr[i], b_thr[i], c_NPU_seq[i], m, k, n, AC_native, B_native);
		}
		auto end_n_seq = std::chrono::high_resolution_clock::now();
		auto elapsed_n_seq = std::chrono::duration_cast<std::chrono::nanoseconds>(end_n_seq - start_n_seq);
		avg_n_seq += elapsed_n_seq.count();

		std::cout << "NPU Sequence Time: " << elapsed_n_seq.count() / 1.0E9f << "s\n";

		for (int i = 0; i < threads; i ++) {
			std::cout << c_NPU_seq[i][0] << "\n";
		}

	// END OF NPU SEQUENCE 
		thr.clear();

	// START OF CV PARALLEL 

		for (int i = 0; i < threads; i ++) {
			thr.emplace_back(fill_mult_CV, a_thr[i], b_thr[i], c_CV_par[i], m, k, n, AC_native, B_native);
		}

		auto start_cv_par = std::chrono::high_resolution_clock::now();
		for (std::thread& t : thr) {
			t.join();
		}
		auto end_cv_par = std::chrono::high_resolution_clock::now();
		auto elapsed_cv_par = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cv_par - start_cv_par);

		avg_cv_par += elapsed_cv_par.count();

		std::cout << "CV Parallel Time: " << elapsed_cv_par.count() / 1.0E9f << "s\n";

		for (int i = 0; i < threads; i ++) {
			std::cout << c_CV_par[i][0] << "\n";
		}

			
	// END OF CV PARALLEL 


	// START OF CV SEQUENCE 

		auto start_cv_seq = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < threads; i ++) {
			fill_mult_CV(a_thr[i], b_thr[i], c_CV_seq[i], m, k, n, AC_native, B_native);
		}
		auto end_cv_seq = std::chrono::high_resolution_clock::now();
		auto elapsed_cv_seq = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cv_seq - start_cv_seq);

		avg_cv_seq += elapsed_cv_seq.count();

		std::cout << "CV Sequence Time: " << elapsed_cv_seq.count() / 1.0E9f << "s\n";

		for (int i = 0; i < threads; i ++) {
			std::cout << c_CV_seq[i][0] << "\n";
		}

	// END OF CV SEQUENCE 


		for (int i = 0; i < threads; i ++) {
			free(a_thr[i]);
			free(b_thr[i]);
			free(c_NPU_par[i]);
			free(c_NPU_seq[i]);
			free(c_CV_par[i]);
			free(c_CV_seq[i]);

		}
		a_thr.clear();
		b_thr.clear();
		c_NPU_par.clear();
		c_NPU_seq.clear();
		c_CV_par.clear();
		c_CV_seq.clear();
		thr.clear();
	}

	avg_n_par = avg_n_par/double(trials);
	avg_n_seq = avg_n_seq/double(trials);
	avg_cv_par = avg_cv_par/double(trials);
	avg_cv_seq = avg_cv_seq/double(trials);

	std::cout << "Avg NPU Parallel Time: " << avg_n_par / 1.0E9f << "s\n";
	std::cout << "Avg NPU Sequence Time: " << avg_n_seq / 1.0E9f << "s\n";
	std::cout << "Avg CV Parallel Time: " << avg_cv_par / 1.0E9f << "s\n";
	std::cout << "Avg CV Sequence Time: " << avg_cv_seq / 1.0E9f << "s\n";



    MatNpu A_CV(m, k, CV_8S, a);
    MatNpu B_CV(k, n, CV_8S, b);


    auto start_cv = std::chrono::high_resolution_clock::now();

    MatNpu C_CV = A_CV.matmul(B_CV, CV_32S, AC_native, B_native);

    auto end_cv = std::chrono::high_resolution_clock::now();
    auto elapsed_cv = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cv - start_cv);

    std::cout << "Time took: " << elapsed_cv.count() / 1.0E9f << "s\n";

    std::cout << "The first item of the matrix: ";
    std::cout << C_CV.at<int32_t>(0, 0) << "\n";
    */
    
}
