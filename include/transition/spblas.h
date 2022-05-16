
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
 * This is the ALP implementation of a subset of the de-facto *_spblas.h Sparse
 * BLAS standard. This implementation uses the spblas_ prefix; e.g.,
 * #spblas_dcsrgemv. All functions defined have <tt>void</tt> return types --
 * i.e., if breaking the contract defined in the APIs, undefined behaviour will
 * occur.
 */

#ifndef _H_ALP_SPBLAS
#define _H_ALP_SPBLAS

#include "blas_sparse_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Computes either
 *  - \f$ y \to Ax \f$, or
 *  - \f$ y \to A^Tx \f$.
 *
 * The matrix \f$ A \f$ is \f$ m \times n \f$ and holds \f$ k \f$ nonzeroes,
 * and is assumed to be stored in Compressed Row Storage (CRS).
 *
 * @param[in] transa Either 'N' or 'T' for transposed ('T') or not ('N').
 * @param[in] m      The row size of \f$ A \f$.
 * @param[in] a      The nonzero value array of \f$ A \f$ of size \f$ k \f$.
 * @param[in] ia     The row offset array of \f$ A \f$ of size \f$ m+1 \f$.
 * @param[in] ja     The column indices of nonzeroes of \f$ A \f$. Must be of
 *                   size \f$ k \f$.
 * @param[in] x      The dense input vector \f$ x \f$ of length \f$ n \f$.
 * @param[out] y     The dense output vector \f$ y \f$ of length \f$ m \f$.
 *
 * All memory regions must be pre-allocated and initialised.
 */
void spblas_dcsrgemv(
	const char * transa,
	const int * m,
	const double * a, const int * ia, const int * ja,
	const double * x,
	double * y
);

/**
 * Computes a variant of \f$ C \to \alpha AB+\beta C \f$.
 *
 * The matrix \f$ A \f$ is sparse and employs the Compressed Row Storage (CRS).
 * The matrices \f$ B, C \f$ are dense. \f$ A \f$ has size \f$ m \times k \f$,
 * \f$ B \f$ is \f$ k \times n \f$ and \f$ C \f$ is \f$ m \times n \f$.
 *
 *
 * @param[in] transa    Either 'N' or 'T'.
 * @param[in] m, n, k   Pointers to integers that equal \f$ m, n, k \f$, resp.
 * @param[in] alpha     Pointer to the scalar \f$ \alpha \f$.
 * @param[in] matdescra Has several entries. Going from first to last:
 *                      Either 'G', 'S', 'H', 'T', 'A', or 'D' (similar to MatrixMarket)
 *                      Either 'L' or 'U', in the case of 'T' (triangular)
 *                      Either 'N' or 'U' for the diagonal type
 *                      Either 'F' or 'C' (one or zero based indexing)
 * @param[in] indx      The column index of the matrix \f$ A \f$.
 * @param[in] pntrb     The Compressed Row Storage (CRS) row start array.
 * @param[in] pntre     The array \a pntrb shifted by one.
 * @param[in] b         Pointer to the values of \f$ B \f$.
 * @param[in] ldb       Leading dimension of \a b. If in row-major format, this
 *                      should be \f$ n \f$. If in column-major format, this
 *                      should be \f$ k \f$.
 * @param[in] beta      Pointer to the scalar \f$ \beta \f$.
 * @param[in] c         Pointer to the values of \f$ C \f$.
 * @param[in] ldc       Leading dimension of \a c. If in row-major format, this
 *                      should be \f$ n \f$. If in column-major format, this
 *                      should be \f$ m \f$.
 */
void spblas_dcsrmm(
	const char * transa,
	const int * m, const int * n, const int * k,
	const double * alpha,
	const char * matdescra, const double * val, const int * indx,
	const int * pntrb, const int * pntre,
	const double * b, const int * ldb,
	const double * beta,
	double * c, const int * ldc
);

/**
 * Computes \f$ C \to AB \f$ or \f$ C \to A^TB \f$, where all matrices are
 * sparse and employ the Compressed Row Storage (CRS).
 *
 * The matrix \f$ C \f$ is \f$ m \times n \f$, the matrix \f$ A \f$ is
 * \f$ m \times k \f$, and the matrix \f$ B \f$ is \f$ k \times n \f$.
 *
 * @param[in] trans Either 'N' or 'T', indicating whether A is to be transposed.
 *                  The Hermitian operator on \a A is currently not supported;
 *                  if required, please submit a ticket.
 * @param[in] request A pointer to an integer that reads either 0, 1, or 2.
 *                    0: the output memory area has been pre-allocated and is
 *                       guaranteed sufficient for storing the output
 *                    1: a symbolic phase will be executed that only modifies
 *                       the row offset array \a ic. This array must have been
 *                       pre-allocated and of sufficient size (\f$ m+1 \f$).
 *                    2: assumes 1 has executed prior to this call and that the
 *                       contents of the row offset arrays have not been
 *                       modified. It also assumes that the column index and
 *                       value arrays are (now) of sufficient size to hold the
 *                       output.
 * @param[in] sort A pointer to an integer value of 7. All other values are not
 *                 supported by this interface. If you require it, please submit
 *                 a ticket.
 * @param[in] m,n,k Pointers to the integer sizes of \a A, \a B, and \a C.
 * @param[in] a     The value array of nonzeroes in \a A.
 * @param[in] ja    The column index array of nonzeroes in \a A.
 * @param[in] ia    The row offset array of nonzeroes in \a A.
 * @param[in] b, ib, jb  Similar for the nonzeroes in \a B.
 * @param[out] c, ic, jc Similar for the nonzeroes in \a C. For these parameters
 *                       depending on \a request there are various assumptions
 *                       on capacity and, for \a ic, contents.
 * @param[in] nzmax A pointer to an integer that holds the capacity of \a c and
 *                  \a jc.
 * @param[out] info The integer pointed to will be set to 0 if the call was
 *                  successful, -1 if the routine only computed the required
 *                  size of \a c and \a jc (stored in \a ic), and any positive
 *                  integer when computation has proceeded successfully until
 *                  (but not including) the returned integer.
 */
void spblas_dcsrmultcsr(
	const char * trans, const int * request, const int * sort,
	const int * m, const int * n, const int * k,
	double * a, int * ja, int * ia,
	double * b, int * jb, int * ib,
	double * c, int * jc, int * ic,
	const int * nzmax, int * info
);

/**
 * An extension that provides sparse matrix times sparse vector multiplication;
 * i.e., either of
 *  -# \f$ y \to y + \alpha A x \f$, or
 *  -# \f$ y \to y + \alpha A^T x \f$.
 * Here, \f$ A \f$ is assumed in Compressed Row Storage (CRS), while \f$ x \f$
 * and \f$ y \f$ are assumed to be using the #extblas_sparse_vector extension.
 *
 * \warning This is an ALP implementation-specific extension.
 *
 * This API follows loosely that of #spblas_dcsrmultcsr.
 * @param[in] trans Either 'N' or 'T', indicating whether A is to be transposed.
 *                  The Hermitian operator on \a A is currently not supported;
 *                  if required, please submit a ticket.
 * @param[in] request A pointer to an integer that reads either 0, 1, or 2.
 *                    0: the output vector is guaranteed to have sufficient
 *                       capacity to hold the output of the computation.
 *                    1: a symbolic phase will be executed that only modifies
 *                       the capacity of the output vector so that it is
 *                       guaranteed to be able to hold the output of the
 *                       requested computation.
 * @param[in] m, n Pointers to integers equal to \f$ m, n \f$.
 * @param[in] a  The value array of the nonzeroes in \f$ A \f$.
 * @param[in] ja The column indices of the nonzeroes in \f$ A \f$.
 * @param[in] ia The row offset arrays of the nonzeroes in \f$ A \f$.
 * @param[in]  x The sparse input vector.
 * @param[out] y The sparse output vector.
 */
void extspblas_dcsrmultsv(
	const char * trans, const int * request,
	const int * m, const int * n,
	const double * a, const int * ja, const int * ia,
	const extblas_sparse_vector x,
	extblas_sparse_vector y
);

/**
 * An extension that frees any buffers the ALP/GraphBLAS-generated SparseBLAS
 * library may have allocated.
 */
void extspblas_free();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end _H_ALP_SPBLAS

