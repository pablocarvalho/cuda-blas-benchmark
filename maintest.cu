// High level matrix multiplication on GPU using CUDA with Thrust, CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <array>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusolverDn.h>


// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t* handle, const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	// cublasHandle_t handle;
	// cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	// cublasDestroy(*handle);
}

void gpu_blas_dsyrk(cublasHandle_t* handle,const double *A, double *C, const int n, const int k) {
	int lda=k,ldc=k;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	// cublasHandle_t handle;
	// cublasCreate(&handle);

	// Do the actual multiplication
	cublasDsyrk(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, alpha, A,lda, beta, C, ldc);

	// Destroy the handle
	// cublasDestroy(*handle);
}

void gpu_solv_dpotrf(cusolverDnHandle_t* handle,double *A, double* W,const int n, const int k, int* devInfo) {
	int lda=n;

	// Create a handle for CUBLAS
	// cublasHandle_t handle;
	// cublasCreate(&handle);

	// Do the actual multiplication

	cusolverDnDpotrf(*handle, CUBLAS_FILL_MODE_UPPER, n, A, lda, W, k, devInfo);


	// Destroy the handle
	// cusolverDnDestroy(*handle);
}

void gpu_blas_dtrsm(cublasHandle_t* handle, double *A, double *B, int a_b_rows, int a_b_cols){

	int lda=a_b_rows,ldb=a_b_cols;
	const double alf = 1;
	const double *alpha = &alf;


	cublasDtrsm(*handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, a_b_rows, a_b_cols, alpha, A, lda, B, ldb);

	// Destroy the handle
	cublasDestroy(*handle);


}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const thrust::device_vector<double> &A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

//int main() {
//
//    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
//	// for simplicity we are going to use square arrays
//	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 512;
//
//	thrust::device_vector<double> d_A(nr_rows_A * nr_cols_A), d_B(nr_rows_B * nr_cols_B), d_C(nr_rows_C * nr_cols_C);
//
//	// Fill the Input arrays
//	GPU_fill_rand(thrust::raw_pointer_cast(&d_A[0]), nr_rows_A, nr_cols_A);
//    GPU_fill_rand(thrust::raw_pointer_cast(&d_B[0]), nr_rows_B, nr_cols_B);
//
//    // bool potrfMode = false;
//
//    int batch_count = 10;
//
//    std::array<thrust::device_vector<double>,10> answerVectors;
//	std::array<thrust::device_vector<double>,10> answerVectorsB;
//
//	std::array<cublasHandle_t, 10> handles;
//	std::array<cudaStream_t, 10> streams;
//
//	std::array<cublasHandle_t, 10> handlesB;
//    std::array<cudaStream_t, 10> streamsB;
//
//    // std::array<cusolverDnHandle_t, 10> handlesPotrf;
//    // std::array<cudaStream_t, 10> streamsPotrf;
//    // std::array<int, 10> potrfAns;
//    // cudaStream_t *streams = (cudaStream_t *) malloc(batch_count*sizeof(cudaStream_t));
//
//    // cusolverDnHandle_t handlePotrf;
//    // cusolverDnCreate(&handlePotrf);
//    // int workspaceSize = -1;
//    // cusolverDnDpotrf_bufferSize(handlePotrf,CUBLAS_FILL_MODE_UPPER,nr_rows_A,thrust::raw_pointer_cast(&d_A[0]),nr_rows_A,&workspaceSize );
//    // nr_rows_C = nr_cols_C = workspaceSize;
//	for(int i=0; i<batch_count; i++)
//	{
//		std::cout << "batch " << i << " initializing"<<std::endl;
//
//		cudaStreamCreate(&streams[i]);
//		cublasCreate(&handles[i]);
//
//		cudaStreamCreate(&streamsB[i]);
//		cublasCreate(&handlesB[i]);
//
//		cublasSetStream(handles[i], streams[i]);
//
//        cublasSetStream(handlesB[i], streamsB[i]);
//
//        // cudaStreamCreate(&streamsPotrf[i]);
//        // cusolverDnCreate(&handlesPotrf[i]);
//        // cusolverDnSetStream(handlesPotrf[i], streamsPotrf[i]);
//
//        thrust::device_vector<double> d_C(nr_rows_C * nr_cols_C);
//        thrust::device_vector<double> d_C2(nr_rows_C * nr_cols_C);
//        // thrust::device_vector<double> d_C2(workspaceSize);
//
//		answerVectors[i] = d_C;
//		answerVectorsB[i] = d_C2;
//	}
//
//
//	for(int i=0; i<batch_count; i++){
//		// Set CUDA stream
//
//		std::cout << "passou aqui" << std::endl;
//
//		// DGEMM: C = alpha*A*B + beta*C
//		gpu_blas_mmul(&handles[i],thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_B[0]), thrust::raw_pointer_cast(&answerVectors[i][0]), nr_rows_A, nr_cols_A, nr_cols_B);
//
//        // gpu_blas_dsyrk(&handlesB[i],thrust::raw_pointer_cast(&d_A[0]),  thrust::raw_pointer_cast(&answerVectorsB[i][0]), nr_rows_A, nr_cols_C);
//
//		// gpu_solv_dpotrf(&handlesPotrf[i],thrust::raw_pointer_cast(&d_A[0]),  thrust::raw_pointer_cast(&answerVectorsB[i][0]), nr_rows_A, workspaceSize, &potrfAns[i]);
//
//		gpu_blas_dtrsm(&handlesB[i],thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_B[0]), nr_rows_A, nr_cols_B);
//	}
//
//
//
//
//	std::cout << "finished" <<std::endl;
//
//	return 0;
//}
