
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 *
 * This is the ALP implementation of a subset of the NIST Sparse BLAS standard.
 * While the API is standardised, this header makes some implementation-specific
 * extensions.
 */

#ifndef _H_ALP_SPARSEBLAS_NIST
#define _H_ALP_SPARSEBLAS_NIST

#include "blas_sparse_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The possible transposition types.
 *
 * See the SparseBLAS paper for the full specification.
 *
 * This implementation at present does not support the #blas_conj_trans type.
 */
enum blas_trans_type {
	blas_no_trans = 0,
	blas_trans,
	blas_conj_trans
};

/**
 * The supported dense storages.
 *
 * See the SparseBLAS paper for the full specification.
 */
enum blas_order_type {
	blas_rowmajor,
	blas_colmajor
};

/**
 * A sparse matrix
 *
 * See the SparseBLAS paper for the full specification.
 *
 * \internal This implementation does not expose the type used internally to
 * represent a sparse matrix, and instead stores it as a generic pointer to this
 * internal representation.
 */
typedef void * blas_sparse_matrix;

/**
 * Creates a handle to a new / empty sparse matrix.
 *
 * A call to this function must always be paired with one to
 *  - #BLAS_duscr_end
 *
 * See the SparseBLAS paper for the full specification.
 */
blas_sparse_matrix BLAS_duscr_begin( const int m, const int n );

/**
 * Inserts a single nonzero entry into \a A
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_duscr_insert_entry(
	blas_sparse_matrix A,
	const double val,
	const int row, const int col
);

/**
 * Inserts a block of entries into \a A
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_duscr_insert_entries(
	blas_sparse_matrix A,
	const int nnz,
	const double * vals, const int * rows, const int * cols
);

/**
 * Inserts a column into \a A
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_duscr_insert_col(
	blas_sparse_matrix A,
	const int j, const int nnz,
	const double * vals, const int * rows
);

/**
 * Inserts a row into \a A
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_duscr_insert_row(
	blas_sparse_matrix A,
	const int i, const int nnz,
	const double * vals, const int * cols
);

/**
 * Signals that the matrix \a A can now be finalised -- all contents have been
 * added.
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_duscr_end( blas_sparse_matrix A );

/**
 * Frees a given matrix.
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_usds( blas_sparse_matrix A );

/**
 * Sparse matrix--dense vector multiplication.
 *
 * This function computes one of
 *  - \f$ y \to \alpha A x + y \f$
 *  - \f$ y \to \alpha A^T x + y \f$
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_dusmv(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const double * const x, int incx,
	double * const y, const int incy
);

/**
 * Sparse matrix--dense matrix multiplication.
 *
 * This function computes one of
 * - \f$ C \to \alpha AB + C \f$
 * - \f$ C \to \alpha A^TB + C \f$
 *
 * See the SparseBLAS paper for the full specification.
 */
int BLAS_dusmm(
	const enum blas_order_type order,
	const enum blas_trans_type transa,
	const int nrhs,
	const double alpha, const blas_sparse_matrix A,
	const double * B, const int ldb,
	const double * C, const int ldc
);

/**
 * Performs sparse matrix--sparse vector multiplication.
 *
 * This function is an implementation-specific extension of SparseBLAS that
 * performs one of
 *  - \f$ y \to \alpha A x + y \f$, or
 *  - \f$ y \to \alpha A^T x + y \f$.
 *
 * @param[in] transa The requested transposition of \f$ A \f$.
 * @param[in] alpha  The scalar with which to element-wise multiply the result
 *                   of the matrix--vector multiplication (prior to addition
 *                   to \f$ y \f$).
 * @param[in] A      The matrix \f$ A \f$ with which to multiply \a x.
 * @param[in] x      The vector \f$ x \f$ with which to multiply \a A.
 * @param[in,out] y  The output vector \f$ y \f$ into which the result of the
 *                   matrix--vector multiplication is added.
 *
 * @returns 0 If the requested operation completed successfully.
 * @returns Any other integer in case of error. If returned, all arguments to
 *          the call to this function shall remain unmodified.
 */
int EXTBLAS_dusmsv(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const extblas_sparse_vector x,
	extblas_sparse_vector y
);

/**
 * Performs sparse matrix--sparse matrix multiplication.
 *
 * This function is an implementation-specific extension of SparseBLAS that
 * performs one of
 *  - \f$ C \to \alpha A   B   + C \f$,
 *  - \f$ C \to \alpha A^T B   + C \f$,
 *  - \f$ C \to \alpha A   B^T + C \f$, or
 *  - \f$ C \to \alpha A^T B^T + C \f$.
 *
 * @param[in] transa The requested transposition of \a A.
 * @param[in] alpha  The scalar with which to element-wise multiply the result
 *                   of \f$ AB \f$.
 * @param[in] A      The left-hand input matrix \f$ A \f$.
 * @param[in] transb The requested transposition of \a B.
 * @param[in] B      The right-hand input matrix \f$ B \f$.
 * @param[in,out] C  The output matrix \f$ C \f$ into which the result of the
 *                   matrix--matrix multiplication is added.
 *
 * @returns 0 If the multiplication has completed successfully.
 * @returns Any other integer on error, in which case the contents of all
 *          arguments to this function shall remain unmodified.
 */
int EXTBLAS_dusmsm(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const enum blas_trans_type transb, const blas_sparse_matrix B,
	blas_sparse_matrix C
);

/**
 * Retrieves the number of nonzeroes in a given, finalised, sparse matrix.
 *
 * @param[in]  A  The matrix to return the number of nonzeroes of.
 * @param[out] nz Where to store the number of nonzeroes.
 *
 * @returns 0 If the function call is successful.
 * @returns Any other value on error, in which case \a nz will remain
 *          untouched.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusm_nz( const blas_sparse_matrix A, int * nz );

/**
 * Opens a given sparse matrix for read-out.
 *
 * @param[in] A The matrix to read out.
 *
 * @returns 0 If the call was successful.
 * @returns Any other value if it was not, in which case the state of \a A
 *          shall remain unchanged.
 *
 * After a successful call to this function, \a A moves into a read-out state.
 * This means \a A shall only be a valid argument for calls to #EXTBLAS_dusm_get
 * and #EXTBLAS_dusm_close.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusm_open( const blas_sparse_matrix A );

/**
 * Retrieves a sparse matrix entry.
 *
 * Each call to this function will retrieve a new entry. The order in which
 * entries are returned is unspecified.
 *
 * @param[in] A The matrix to retrieve an entry of.
 *
 * The given matrix must be opened for read-out, and must not have been closed
 * in the mean time.
 *
 * @param[out] val The value of the retrieved nonzero.
 * @param[out] row The row coordinate of the retrieved nonzero.
 * @param[out] col The column coordinate of the retrieved nonzero.
 *
 * @returns 0 If a nonzero was successfully returned and a next value is not
 *            available; i.e., the read-out has completed. When this is
 *            returned, \a A will no longer be a legal argument for a call to
 *            this function.
 * @returns 1 If a nonzero was successfully returned and a next nonzero is
 *            available.
 * @returns Any other integer in case of error.
 *
 * In case of error, the output memory areas pointed to by \a value, \a row, and
 * \a col will remain untouched. Furthermore, \a A will no longer be a legal
 * argument for a call to this function.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusm_get(
	const blas_sparse_matrix A,
	double * value, int * row, int * col
);

/**
 * Closes a sparse matrix read-out.
 *
 * @param[in] A The matrix which is in a read-out state.
 *
 * @returns 0 If \a A is successfully returned to a finalised state.
 * @returns Any other integer in case of error, which brings \a A to an
 *          undefined state.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusm_close( const blas_sparse_matrix A );

/**
 * Removes all entries from a finalised sparse matrix.
 *
 * @param[in,out] A The matrix to clear.
 *
 * @returns 0 If \a A was successfully cleared.
 * @returns Any other integer in case of error, which brings \a A into an
 *          undefined state.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusm_clear( blas_sparse_matrix A );

/**
 * This function is an implementation-specific extension of SparseBLAS that
 * clears any buffer memory that preceding SparseBLAS operations may have
 * created and used.
 *
 * @returns 0 On success.
 * @returns Any other integer on failure, in which case the ALP/SparseBLAS
 *          implementation enters an undefined state.
 */
int EXTBLAS_free();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end _H_ALP_SPARSEBLAS_NIST

