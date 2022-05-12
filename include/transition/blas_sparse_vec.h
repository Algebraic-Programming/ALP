
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
 * This is an ALP-specific extension to the NIST Sparse BLAS standard, which
 * the ALP libsparseblas transition path also introduces to the de-facto spblas
 * standard.
 */

#ifndef _H_ALP_SPARSEBLAS_EXT_VEC
#define _H_ALP_SPARSEBLAS_EXT_VEC

#ifdef __cplusplus
extern "C" {
#endif

/** A sparse vector. This is an implementation-specific extension. */
typedef void * extblas_sparse_vector;

/**
 * Creates a handle to a new sparse vector that holds no entries.
 *
 * This is an implementation-specific extension.
 *
 * @param[in] n The returned vector size.
 *
 * @returns An #extblas_sparse_vector that is under construction.
 */
extblas_sparse_vector EXTBLAS_dusv_begin( const int n );

/**
 * Inserts a new nonzero entry into a sparse vector that is under construction.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusv_insert_entry(
	extblas_sparse_vector x,
	const double val,
	const int index
);

/**
 * Signals the end of sparse vector construction, making the given vector ready
 * for use.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusv_end( extblas_sparse_vector x );

/**
 * Destroys the given sparse vector.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusvds( extblas_sparse_vector x );

/**
 * @returns The number of nonzeroes in a given sparse vector.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusvnz( const extblas_sparse_vector x );

/**
 * Opens a sparse vector for read-out.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusv_open( const extblas_sparse_vector x );

/**
 * Retrieves a sparse vector entry. Each call to this function will retrieve a
 * new entry. The order in which entries are returned is unspecified.
 *
 * The given vector must be opened for read-out, and must not have been closed
 * in the mean time.
 *
 * This is an implementation-specific extension.
 *
 * @returns 0  If a value was successfully returned but a next value is not
 *             available (i.e., the read-out has completed).
 * @returns 1  If a value was successfully returned and a next value is
 *             available.
 * @returns 10 In case of error.
 */
int EXTBLAS_dusv_get(
	const extblas_sparse_vector x,
	double * const val, int * const ind
);

/**
 * Closes a sparse vector read-out.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusv_close( const extblas_sparse_vector x );

/**
 * Removes all entries from a finalised sparse vector.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_dusv_clear( extblas_sparse_vector x );

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end `_H_ALP_SPARSEBLAS_EXT_VEC'

