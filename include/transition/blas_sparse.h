
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
 * The supported transposition types.
 */
enum blas_trans_type {
	blas_no_trans = 0,
	blas_trans,
	blas_conj_trans
};

/**
 * The supported dense storages.
 */
enum blas_order_type {
	blas_rowmajor,
	blas_colmajor
};

/** A sparse matrix */
typedef void * blas_sparse_matrix;

/**
 * Creates a handle to a new / empty sparse matrix.
 *
 * A call to this function must always be paired with one to
 *  - #BLAS_duscr_end
 */
blas_sparse_matrix BLAS_duscr_begin( const int m, const int n );

/**
 * Inserts a single nonzero entry into \a A
 */
int BLAS_duscr_insert_entry(
	blas_sparse_matrix A,
	const double val,
	const int row, const int col
);

/**
 * Inserts a block of entries into \a A
 */
int BLAS_duscr_insert_entries(
	blas_sparse_matrix A,
	const int nnz,
	const double * vals, const int * rows, const int * cols
);

/**
 * Inserts a column into \a A
 */
int BLAS_duscr_insert_col(
	blas_sparse_matrix A,
	const int j, const int nnz,
	const double * vals, const int * rows
);

/**
 * Inserts a row into \a A
 */
int BLAS_duscr_insert_row(
	blas_sparse_matrix A,
	const int i, const int nnz,
	const double * vals, const int * cols
);

/**
 * Signals that the matrix \a A can now be finalised -- all contents have been
 * added.
 */
int BLAS_duscr_end( blas_sparse_matrix A );

/**
 * Removes all entries from a finalised sparse matrix.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_duscr_clear( blas_sparse_matrix x );

/**
 * Frees a given matrix.
 */
int BLAS_usds( blas_sparse_matrix A );

/**
 * This function computes one of
 *  - \f$ y \to \alpha A x + y \f$
 *  - \f$ y \to \alpha A^T x + y \f$
 */
int BLAS_dusmv(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const double * const x, int incx,
	double * const y, const int incy
);

/**
 * This function computes one of
 * - \f$ C \to \alpha AB + C \f$
 * - \f$ C \to \alpha A^TB + C \f$
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
 * This function is an implementation-specific extension of SparseBLAS that
 * performs sparse matrix sparse vector multiplication; i.e., one of
 *  - \f$ y \to \alpha A x + y \f$, or
 *  - \f$ y \to \alpha A^T x + y \f$.
 */
int EXTBLAS_dusmsv(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const extblas_sparse_vector x,
	extblas_sparse_vector y
);

/**
 * This function is an implementation-specific extension of SparseBLAS that
 * performs sparse matrix sparse matrix multiplication, i.e., one of
 *  - \f$ C \to \alpha A   B   + C \f$,
 *  - \f$ C \to \alpha A^T B   + C \f$,
 *  - \f$ C \to \alpha A   B^T + C \f$, or
 *  - \f$ C \to \alpha A^T B^T + C \f$.
 */
int EXTBLAS_dusmsm(
	const enum blas_trans_type transa,
	const double alpha, const blas_sparse_matrix A,
	const enum blas_trans_type transb, const blas_sparse_matrix B,
	blas_sparse_matrix C
);

/**
 * This function is an implementation-specific extension of SparseBLAS that
 * clears any buffer memory that preceding SparseBLAS operations may have
 * created and used.
 */
int EXTBLAS_free();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end _H_ALP_SPARSEBLAS_NIST

