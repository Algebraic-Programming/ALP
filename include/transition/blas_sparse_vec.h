
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

#define __SPBLAS_CONC( _a, _b ) _a ## _b
#define __SPBLAS_CONCAT( _a, _b ) __SPBLAS_CONC( _a, _b )
#define SPCONCAT( _a, _b ) __SPBLAS_CONCAT( _a, _b )

#ifdef __cplusplus
extern "C" {
#endif

#define EXTBLAS_FUN( name ) SPCONCAT( EXTBLAS_, name )

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
extblas_sparse_vector EXTBLAS_FUN( dusv_begin )( const int n );

/**
 * Inserts a new nonzero entry into a sparse vector that is under construction.
 *
 * @param[in,out] x   The sparse vector to which to add a nonzero.
 * @param[in]     val The nonzero to add to \a x.
 * @param[in]   index The nonzero coordinate.
 *
 * The value \a index must be smaller than the size of the vector \a x as given
 * during the call to #EXTBLAS_dusv_begin that returned \a x.
 *
 * @returns 0 If \a x has successfully ingested the given nonzero.
 * @returns Any other integer on error, in which case the state of \a x shall
 *          become undefined.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_insert_entry )(
	extblas_sparse_vector x,
	const double val,
	const int index
);

/**
 * Signals the end of sparse vector construction, making the given vector ready
 * for use.
 *
 * @param[in,out] x The sparse vector that is under construction.
 *
 * @returns 0 If \a x has successfully been moved to a finalised state.
 * @returns Any other integer if the call was unsuccessful, in which case the
 *          state of \a x becomes undefined.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_end )( extblas_sparse_vector x );

/**
 * Destroys the given sparse vector.
 *
 * @param[in] x The finalised sparse vector to destroy.
 *
 * @returns 0 If the call was successful, after which \a x should no longer be
 *            used unless it is overwritten by a call to #EXTBLAS_dusv_begin.
 * @returns Any other integer if the call was unsuccessful, in which case the
 *          state of \a x becomes undefined.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusvds )( extblas_sparse_vector x );

/**
 * Retrieves the number of nonzeroes in a given finalised sparse vector.
 *
 * @param[in]  x  The vector of which to return the number of nonzeroes.
 * @param[out] nz Where to store the number of nonzeroes in a given sparse
 *                vector.
 *
 * @returns 0 If the call was successful and \a nz was set.
 * @returns Any other integer if the call was unsuccessful, in which case \a nz
 *          shall remain untouched.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_nz )( const extblas_sparse_vector x, int * nz );

/**
 * Opens a sparse vector for read-out.
 *
 * @param[in] x The vector to read out.
 *
 * @returns 0 If the call was successful.
 * @returns Any other integer indicating an error, in which case the state of
 *          \a x shall remain unchanged.
 *
 * After a successful call to this function, \a x moves into a read-out state.
 * This means \a x shall only be a valid argument for calls to #EXTBLAS_dusv_get
 * and #EXTBLAS_dusv_close.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_open )( const extblas_sparse_vector x );

/**
 * Retrieves a sparse vector entry.
 *
 * Each call to this function will retrieve a new entry. The order in which
 * entries are returned is unspecified.
 *
 * @param[in] x The vector to retrieve an entry of.
 *
 * The given vector must be opened for read-out, and must not have been closed
 * in the mean time.
 *
 * @param[out] val The value of the retrieved nonzero.
 * @param[out] ind The index of the retrieved nonzero value.
 *
 * @returns 0 If a nonzero was successfully returned but a next value is not
 *            available; i.e., the read-out has completed. When this is
 *            returned, \a x will no longer be a legal argument for a call to
 *            this function.
 * @returns 1 If a value was successfully returned and a next nonzero is
 *            available.
 * @returns Any other integer in case of error.
 *
 * In case of error, the output memory areas pointed to by \a val and \a ind
 * shall remain untouched. Furthermore, \a x will no longer be a valid argument
 * for a call to this function.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_get )(
	const extblas_sparse_vector x,
	double * const val, int * const ind
);

/**
 * Closes a sparse vector read-out.
 *
 * @param[in] x The vector which is in a read-out state.
 *
 * @returns 0 If \a x is successfully returned to a finalised state.
 * @returns Any other integer in case of error, which brings \a A to an
 *          undefined state.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_close )( const extblas_sparse_vector x );

/**
 * Removes all entries from a finalised sparse vector.
 *
 * @param[in] x The vector to clear.
 *
 * @returns 0 If \a x was successfully cleared.
 * @returns Any other integer in case of error, which brings \a x into an
 *          undefined state.
 *
 * This is an implementation-specific extension.
 */
int EXTBLAS_FUN( dusv_clear )( extblas_sparse_vector x );

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end `_H_ALP_SPARSEBLAS_EXT_VEC'

