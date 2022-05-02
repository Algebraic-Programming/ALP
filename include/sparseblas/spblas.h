
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
 * BLAS standard.
 */

/**
 * Computes either
 *  - \f$ y \to Ax \f$, or
 *  - \f$ y \to A^Tx \f$.
 *
 * @param[in] transa Either 'N' or 'T' for transposed ('T') or not ('N').
 */
void spblas_dcsrgemv(
	const char * const transa,
	const int * const m,
	const double * const a, const int * const ia, const int * const ja,
	const double * const x,
	double * const y
);

/**
 * Computes a variant of \f$ C \to \alpha AB+\beta C \f$.
 *
 * @param[in] transa Either 'N' or 'T'.
 * @param[in] matdescra Has several entries. Going from first to last:
 *                      Either 'G', 'S', 'H', 'T', 'A', or 'D' (similar to MatrixMarket)
 *                      Either 'L' or 'U', in the case of 'T' (triangular)
 *                      Either 'N' or 'U' for the diagonal type
 *                      Either 'F' or 'C' (one or zero based indexing)
 * @param[in] indx  The column index of the matrix \f$ A \f$.
 * @param[in] pntrb The Compressed Row Storage (CRS) row start array.
 * @param[in] pntre The array \a pntre, shifted by one.
 */
void spblas_dcsrmm(
	const char * const transa,
	const int * const m, const int * const n, const int * const k,
	const double * const alpha,
	const char * const matdescra, const double * const val, const int * const indx, const int * const pntrb, const int * const pntre,
	const double * const b, const int * const ldb,
	const double * const beta,
	double * const c, const int * const ldc
);

