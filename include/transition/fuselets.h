
/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
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
 * \ingroup TRANS
 *
 * Provides a set of fused level-1 and level-2 ALP kernels.
 *
 * The fused kernels are designed to be easily callable from existing code
 * bases, using standard data structures such as raw pointers to vectors and the
 * Compressed Row Storage (CRS) for sparse matrices.
 *
 * @author A. N. Yzelman
 * @date 27/09/2024
 */

/**
 * \defgroup TRANS_FUSELETS Fuselets
 * \ingroup TRANS
 * @{
 *
 * \todo add documentation
 */

#ifndef _H_ALP_FUSELETS
#define _H_ALP_FUSELETS

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

	/**
	 * Initialisation routine that should be called before calling any fuselets.
	 *
	 * A user application shall
	 *  1. call this function after application start and before calling any
	 *     fuselets, as well as
	 *  2. call this function after a call to #finalize_fuselets and before any
	 *     subsequent calls to fuselets.
	 * This function shall not be called in any other case.
	 *
	 * \note For example, it is not legal to call this function twice without an
	 *       call to #finalize_fuselets in between.
	 *
	 * To ensure proper clean-up before application termination, all calls to this
	 * function should be matched with a call to #finalize_fuselets.
	 *
	 * @returns Zero, if the initialisation has proceeded successfully.
	 * @returns Any other value, if initialisation has failed. In this case,
	 *          it shall be as though this call had never occurred. In particular,
	 *          any subsequent calls to fuselets shall (thus) induce undefined
	 *          behaviour.
	 *
	 * The recommendation is to call this function once and as soon as possible
	 * after the application <tt>main</tt> function has started.
	 */
	int initialize_fuselets();

	/**
	 * Cleans up fuselet resources.
	 *
	 * It may only be called once after every call to #initialize_fuselets. Cannot
	 * follow another call to #finalize_fuselets without a call to
	 * #initialize_fuselets in between.
	 *
	 * The recommendation is to call this function once and just before the
	 * application <tt>main</tt> function terminates.
	 */
	int finalize_fuselets();

	/**
	 * Computes \f$ p, u, \alpha \f$ from:
	 *
	 *  - \f$ p = z + \beta p \f$,
	 *  - \f$ u = Ap \f$,
	 *  - \f$ \alpha = (u,p) \f$.
	 *
	 * @param[in,out] p  The input and output vector \f$ p \f$
	 * @param[out]    u  The output vector \f$ u \f$
	 * @param[out] alpha The output scalar \f$ \alpha \f$
	 *
	 * The pointers \a p and \a u should be pointers to arrays, while \a alpha
	 * should be a pointer to a scalar. The contents of \a u need \em not be zeroed
	 * out(!)-- this fuselet will reset the vector. Similarly, the initial value of
	 * \a alpha will be ignored.
	 *
	 * @param[in]   z  The input vector \f$ z \f$
	 * @param[in] beta The input scalar \f$ \beta \f$
	 *
	 * The pointer \a z should point to an array while \a beta is a scalar.
	 *
	 * @param[in] ia The CRS row offset array of \f$ A \f$
	 * @param[in] ij The CRS column index array of \f$ A \f$
	 * @param[in] iv The CRS value array of \f$ A \f$
	 *
	 * The pointers \a ia, \a ij, and \a iv correspond to a CRS of \f$ A \f$.
	 *
	 * @param[in] n The row-wise \em and column-wise dimension of \a A
	 *
	 * The size of the arrays \a p, \a u, and \a z is \f$ n \f$. The size of the
	 * array \f$ ia \f$ is \f$ n + 1 \f$. The size of the arrays \a ij and * \a iv
	 * is <tt>ia[n]</tt>.
	 *
	 * @returns Zero if and only if the call executed successfully.
	 * @returns A nonzero error code otherwise.
	 */
	int update_spmv_dot(
		double * const p, double * const u, double * const alpha, // outputs
		const double * const z, const double beta,                // input 1
		const size_t * const ia, const unsigned int * const ij,
		const double * const iv,                                  // input 2
		const size_t n                                            // size
	);

	/**
	 * Computes \f$ x, r, \mathit{norm} \f$ from:
	 *
	 *  - \f$ x = \alpha p + x \f$,
	 *  - \f$ r = \beta u + r \f$,
	 *  - \f$ \mathit{norm} = ||r||_2^2 \f$.
	 *
	 * @param[in,out] x     The input and output vector \f$ x \f$
	 * @param[in,out] r     The input and output vector \f$ r \f$
	 * @param[out]    norm2 The 2-norm-squared of \f$ r \f$
	 *
	 * The pointers \a x and \a r should be pointers to arrays, while \a norm2
	 * should be a pointer to a scalar. The initial value of \a norm2 will be
	 * ignored.
	 *
	 * @param[in] alpha The input scalar \f$ \alpha \f$
	 * @param[in] p     The input vector \f$ p \f$
	 *
	 * Here, \a alpha is a scalar while \a p is a pointer to an array.
	 *
	 * @param[in] beta The input scalar \f$ \beta \f$
	 * @param[in] u    The input vector \f$ u \f$
	 *
	 * Here, \a beta is a scalar while \a u is a pointer to an array.
	 *
	 * @param[in] n The size of the vectors \a x, \a r, \a p, and \a u.
	 */
	int update_update_norm2(
		double * const x, double * const r, double * const norm2, // outputs
		const double alpha, const double * const p,               // input 1
		const double beta, const double * const u,                // input 2
		const size_t n                                            // size
	);

	/**
	 * Computes \f$ p \f$ from:
	 *
	 *  - \f$ p = \alpha r + \beta v + \gamma p \f$
	 *
	 * @param[in,out] p     The input and output vector \f$ p \f$
	 *
	 * The pointer \a p should be a poitner to an array.
	 *
	 * @param[in] alpha The input scalar \f$ \alpha \f$
	 * @param[in] r     The input vector \f$ r \f$
	 * @param[in] beta  The input scalar \f$ \beta \f$
	 * @param[in] v     The input vector \f$ v \f$
	 * @param[in] gamma The input scalar \f$ \gamma \f$
	 *
	 * Here, \a alpha, \a beta, \a gamma are scalars while \a p, \a r, and \a v are
	 * pointers to arrays.
	 *
	 * @param[in] n The size of the vectors \a p, \a r, and \a v.
	 */
	int double_update(
		double * const p,                           // output
		const double alpha, const double * const r, // input 1
		const double beta, const double * const v,  // input 2
		const double gamma,                         // input 3
		const size_t n                              // size
	);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end ifdef _H_ALP_FUSELETS

/** @} */ // ends doxygen page for fuselets

