/*
 * ExperimentManager.h
 *
 *  Created on: 23/06/2019
 *      Author: pablomoreira
 */
#ifndef EXPERIMENTMANAGER_H_
#define EXPERIMENTMANAGER_H_

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <list>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusolverDn.h>

using namespace std;



class ExperimentManager {
public:
	ExperimentManager(int inputSize,int batches, bool concurrent, string func1, string func2);
	virtual ~ExperimentManager();

	static const array<string,4> DISPONIBLE_FUNCTIONS;
	static const int STANDARD_SIZE;
	static const int STANDARD_ATTEMPTS;

	void runExperiments();

	thrust::device_vector<double> m_input1;
	thrust::device_vector<double> m_input2;

	vector<thrust::device_vector<double>> m_func1OutputVectors;
	vector<thrust::device_vector<double>> m_func2OutputVectors;

	vector<cublasHandle_t> func1Handles;
	vector<cudaStream_t> func1Streams;

	vector<cublasHandle_t> func2Handles;
	vector<cudaStream_t> func2Streams;

	bool func1cuSolver;
	bool func2cuSolver;

	//Handles used when one of the functions belongs to cuSolver insted of cuBLAS
	vector<cusolverDnHandle_t> func1cuSolverHandles;
	vector<cusolverDnHandle_t> func2cuSolverHandles;

	int m_func1cusolverWorkspaceSize;
	int m_func2cusolverWorkspaceSize;

	static void GPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A);


private:
	int m_func1;
	int m_func2;
	int m_batchesNumber;
	bool m_concurrent;
	int m_inputSize;

	void gpu_blas_mmul(cublasHandle_t* handle, const double *A, const double *B, double *C, const int m, const int k, const int n);
	void gpu_blas_dsyrk(cublasHandle_t* handle,const double *A, double *C, const int n, const int k);
	void gpu_solv_dpotrf(cusolverDnHandle_t* handle,double *A, double* W,const int n, const int k, int* devInfo);
	void gpu_blas_dtrsm(cublasHandle_t* handle, double *A, double *B, int a_b_rows, int a_b_cols);

};

#endif /* EXPERIMENTMANAGER_H_ */
