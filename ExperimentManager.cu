/*
 * ExperimentManager.cpp
 *
 *  Created on: 23/06/2019
 *      Author: pablomoreira
 */

#include "ExperimentManager.h"
#include <chrono>
#include <thread>
#include <string>
#include <cuda_profiler_api.h>

const array<string,4> ExperimentManager::DISPONIBLE_FUNCTIONS = {"dtrsm","dgemm","dsyrk","dpotrf"};
const int ExperimentManager::STANDARD_SIZE = 512;
const int ExperimentManager::STANDARD_ATTEMPTS = 10;


ExperimentManager::ExperimentManager(int inputSize,int batches, bool concurrent, string func1, string func2)
{

	m_concurrent = concurrent;
	m_batchesNumber = batches;
	m_inputSize = inputSize;


	m_input1 = thrust::device_vector<double> (m_inputSize * m_inputSize);
	m_input2 = thrust::device_vector<double> (m_inputSize * m_inputSize);

	m_func1OutputVectors = vector<thrust::device_vector<double>>(m_batchesNumber);
	m_func2OutputVectors = vector<thrust::device_vector<double>>(m_batchesNumber);

	GPU_fill_rand(thrust::raw_pointer_cast(&m_input1[0]), m_inputSize, m_inputSize);
	GPU_fill_rand(thrust::raw_pointer_cast(&m_input2[0]), m_inputSize, m_inputSize);

	func1cuSolver = func2cuSolver = false;
	for(int i = 0; i < DISPONIBLE_FUNCTIONS.size(); ++i)
	{
		if(func1 == DISPONIBLE_FUNCTIONS[i])
		{
			m_func1 = i;
			if(i == 3)
			{
				func1cuSolver = true;

				cusolverDnHandle_t handlePotrf;
				cusolverDnCreate(&handlePotrf);
				cusolverDnDpotrf_bufferSize(handlePotrf,CUBLAS_FILL_MODE_UPPER,m_inputSize,thrust::raw_pointer_cast(&m_input1[0]),m_inputSize,&m_func1cusolverWorkspaceSize );
			}
		}
		if(func2 == DISPONIBLE_FUNCTIONS[i])
		{
			m_func2 = i;
			if(i == 3)
			{
				func2cuSolver = true;

				cusolverDnHandle_t handlePotrf;
				cusolverDnCreate(&handlePotrf);
				cusolverDnDpotrf_bufferSize(handlePotrf,CUBLAS_FILL_MODE_UPPER,m_inputSize,thrust::raw_pointer_cast(&m_input2[0]),m_inputSize,&m_func2cusolverWorkspaceSize );
			}
		}

	}

	if(!func1cuSolver)
		func1Handles = vector<cublasHandle_t>(m_batchesNumber);
	else
		func1cuSolverHandles = vector<cusolverDnHandle_t>(m_batchesNumber);

	if(!func2cuSolver)
		func2Handles = vector<cublasHandle_t>(m_batchesNumber);
	else
		func2cuSolverHandles = vector<cusolverDnHandle_t>(m_batchesNumber);

	if(m_concurrent)
	{
		func1Streams = vector<cudaStream_t>(m_batchesNumber);
		func2Streams = vector<cudaStream_t>(m_batchesNumber);
	}

	for(int i=0; i<m_batchesNumber; i++)
	{

		if(!func1cuSolver)
		{
			cublasCreate(&func1Handles[i]);
			m_func1OutputVectors[i] = thrust::device_vector<double>(m_inputSize * m_inputSize);
		}
		else
		{
			cusolverDnCreate(&func1cuSolverHandles[i]);
			m_func1OutputVectors[i] = thrust::device_vector<double>(m_func1cusolverWorkspaceSize);
		}

		if(!func2cuSolver)
		{
			cublasCreate(&func2Handles[i]);
			m_func2OutputVectors[i] = thrust::device_vector<double>(m_inputSize * m_inputSize);
		}
		else
		{
			cusolverDnCreate(&func2cuSolverHandles[i]);
			m_func2OutputVectors[i] = thrust::device_vector<double>(m_func2cusolverWorkspaceSize);
		}




	}

//	if(!func2cuSolver)
//	{
//		m_func2OutputVectors = thrust::device_vector<double>(m_inputSize * m_inputSize * m_batchesNumber);
//	}
//	else
//	{
//		m_func2OutputVectors = thrust::device_vector<double>(m_func2cusolverWorkspaceSize * m_batchesNumber);
//	}

}

ExperimentManager::~ExperimentManager() {
	// TODO Auto-generated destructor stub



}

void ExperimentManager::GPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
}

void ExperimentManager::runExperiments(){


	int *potrfstates1;
	int *potrfstates2;

	if(m_func1 ==3)
		potrfstates1 = new int[m_batchesNumber];
	if(m_func2 ==3)
		potrfstates2 = new int[m_batchesNumber];

	for(int i=0; i<m_batchesNumber; i++)
	{

		if(m_concurrent)
		{
			cudaStreamCreate(&func1Streams[i]);
			cudaStreamCreate(&func2Streams[i]);
			if(!func1cuSolver)
				cublasSetStream(func1Handles[i], func1Streams[i]);
			else
				cusolverDnSetStream(func1cuSolverHandles[i], func1Streams[i]);
			if(!func2cuSolver)
				cublasSetStream(func2Handles[i], func2Streams[i]);
			else
				cusolverDnSetStream(func2cuSolverHandles[i], func2Streams[i]);
		}

		string func1 = DISPONIBLE_FUNCTIONS[m_func1];
		string func2 = DISPONIBLE_FUNCTIONS[m_func2];

		switch(m_func1)
		{
			case 0:
				gpu_blas_dtrsm(&func1Handles[i],thrust::raw_pointer_cast(&m_input1[0]), thrust::raw_pointer_cast(&m_func1OutputVectors[i][0]), m_inputSize, m_inputSize);
				break;
			case 1:
				gpu_blas_mmul(&func1Handles[i],thrust::raw_pointer_cast(&m_input1[0]), thrust::raw_pointer_cast(&m_input2[0]), thrust::raw_pointer_cast(&m_func1OutputVectors[i][0]), m_inputSize, m_inputSize, m_inputSize);
				break;
			case 2:
				gpu_blas_dsyrk(&func1Handles[i],thrust::raw_pointer_cast(&m_input1[0]),  thrust::raw_pointer_cast(&m_func1OutputVectors[i][0]), m_inputSize, m_inputSize);
				break;
			case 3:
				gpu_solv_dpotrf(&func1cuSolverHandles[i],thrust::raw_pointer_cast(&m_input1[0]),  thrust::raw_pointer_cast(&m_func1OutputVectors[i][0]), m_inputSize, m_func1cusolverWorkspaceSize, &potrfstates1[i]);
				break;

		}

		switch(m_func2)
		{
			case 0:
				gpu_blas_dtrsm(&func2Handles[i],thrust::raw_pointer_cast(&m_input1[0]), thrust::raw_pointer_cast(&m_func2OutputVectors[i][0]), m_inputSize, m_inputSize);
				break;
			case 1:
				gpu_blas_mmul(&func2Handles[i],thrust::raw_pointer_cast(&m_input1[0]), thrust::raw_pointer_cast(&m_input2[0]), thrust::raw_pointer_cast(&m_func2OutputVectors[i][0]), m_inputSize, m_inputSize, m_inputSize);
				break;
			case 2:
				gpu_blas_dsyrk(&func2Handles[i],thrust::raw_pointer_cast(&m_input1[0]),  thrust::raw_pointer_cast(&m_func2OutputVectors[i][0]), m_inputSize, m_inputSize);
				break;
			case 3:
				gpu_solv_dpotrf(&func2cuSolverHandles[i],thrust::raw_pointer_cast(&m_input1[0]),  thrust::raw_pointer_cast(&m_func2OutputVectors[i][0]), m_inputSize, m_func2cusolverWorkspaceSize, &potrfstates2[i]);
				break;

		}







	}

	cudaDeviceSynchronize();

//	std::this_thread::sleep_for (std::chrono::seconds(10));




	cout << "finished" <<endl;
}

void ExperimentManager::gpu_blas_mmul(cublasHandle_t* handle, const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);


}

void ExperimentManager::gpu_blas_dsyrk(cublasHandle_t* handle,const double *A, double *C, const int n, const int k) {
	int lda=k,ldc=k;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	cublasDsyrk(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, alpha, A,lda, beta, C, ldc);

}

void ExperimentManager::gpu_solv_dpotrf(cusolverDnHandle_t* handle,double *A, double* W,const int n, const int k, int* devInfo) {

	int lda=n;

	cusolverDnDpotrf(*handle, CUBLAS_FILL_MODE_UPPER, n, A, lda, W, k, devInfo);

}

void ExperimentManager::gpu_blas_dtrsm(cublasHandle_t* handle, double *A, double *B, int a_b_rows, int a_b_cols){

	int lda=a_b_rows,ldb=a_b_cols;
	const double alf = 1;
	const double *alpha = &alf;


	cublasDtrsm(*handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, a_b_rows, a_b_cols, alpha, A, lda, B, ldb);




}
