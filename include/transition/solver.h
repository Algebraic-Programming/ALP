
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * This exposes a transition path API to the linear system solvers implemented
 * in ALP. The primary benefit compared to simply using a SpBLAS or SparseBLAS
 * interface, is that solvers herein defined can be compiled using the
 * nonblocking backend, thus automatically optimising across (Sparse)BLAS
 * primitives.
 *
 * @author A. N. Yzelman
 * @date 5th of October, 2023
 */

#ifndef _H_ALP_SPARSE_LINSOLVERS
#define _H_ALP_SPARSE_LINSOLVERS

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {

	/** The call has completed successfully. */
	NO_ERROR,

	/**
	 * Illegal NULL pointer provided as argument.
	 */
	NULL_ARGUMENT,

	/**
	 * Illegal argument provided.
	 */
	ILLEGAL_ARGUMENT,

	/**
	 * Out of memory error detected during call.
	 */
	OUT_OF_MEMORY,

	/*
	* The algorithm has failed achieving its intendeed result. For example, an
	* iterative solver did not converge.
	*/
	FAILED,

	/**
	 * An unknown error has been encountered. The state of the underlying solver
	 * library has become undefined.
	 */
	UNKNOWN

} sparse_err_t;

typedef void * sparse_cg_handle_t;

sparse_err_t sparse_cg_init_sii(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const int * const ja, const int * const ia
);

sparse_err_t sparse_cg_init_dii(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const int * const ja, const int * const ia
);

sparse_err_t sparse_cg_init_szi(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const size_t * const ja, const int * const ia
);

sparse_err_t sparse_cg_init_dzi(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const size_t * const ja, const int * const ia
);

sparse_err_t sparse_cg_init_szz(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const size_t * const ja, const size_t * const ia
);

sparse_err_t sparse_cg_init_dzz(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const size_t * const ja, const size_t * const ia
);

// Note that siz and diz are skipped on purpose. Such variants would not seem
// sensible, though could easily be provided if they do turn out to be needed

sparse_err_t sparse_cg_get_tolerance_sii(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_tolerance_szi(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_tolerance_szz(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_tolerance_dii(
	const sparse_cg_handle_t handle, double * const tol );

sparse_err_t sparse_cg_get_tolerance_dzi(
	const sparse_cg_handle_t handle, double * const tol );

sparse_err_t sparse_cg_get_tolerance_dzz(
	const sparse_cg_handle_t handle, double * const tol );

sparse_err_t sparse_cg_set_tolerance_sii(
	sparse_cg_handle_t handle, const float tol );

sparse_err_t sparse_cg_set_tolerance_szi(
	sparse_cg_handle_t handle, const float tol );

sparse_err_t sparse_cg_set_tolerance_szz(
	sparse_cg_handle_t handle, const float tol );

sparse_err_t sparse_cg_set_tolerance_dii(
	sparse_cg_handle_t handle, const double tol );

sparse_err_t sparse_cg_set_tolerance_dzi(
	sparse_cg_handle_t handle, const double tol );

sparse_err_t sparse_cg_set_tolerance_dzz(
	sparse_cg_handle_t handle, const double tol );

sparse_err_t sparse_cg_get_residual_sii(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_residual_szi(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_residual_szz(
	const sparse_cg_handle_t handle, float * const tol );

sparse_err_t sparse_cg_get_residual_dii(
	const sparse_cg_handle_t handle, double * const tol );

sparse_err_t sparse_cg_get_residual_dzi(
	const sparse_cg_handle_t handle, double * const tol );

sparse_err_t sparse_cg_get_residual_dzz(
	const sparse_cg_handle_t handle, double * const tol );

// another variant of sparse_cg_get_iter_count could provide output as an int,
// uint, etc.

sparse_err_t sparse_cg_get_iter_count_sii(
	const sparse_cg_handle_t handle, size_t * const iters );

sparse_err_t sparse_cg_get_iter_count_szi(
	const sparse_cg_handle_t handle, size_t * const iters );

sparse_err_t sparse_cg_get_iter_count_szz(
	const sparse_cg_handle_t handle, size_t * const iters );

sparse_err_t sparse_cg_get_iter_count_dii(
	const sparse_cg_handle_t handle, size_t * const iters );

sparse_err_t sparse_cg_get_iter_count_dzi(
	const sparse_cg_handle_t handle, size_t * const iters );

sparse_err_t sparse_cg_get_iter_count_dzz(
	const sparse_cg_handle_t handle, size_t * const iters );

// another variant of sparse_cg_set_max_iter_count could take int, uint, etc.
// inputs

sparse_err_t sparse_cg_set_max_iter_count_sii(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_set_max_iter_count_szi(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_set_max_iter_count_szz(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_set_max_iter_count_dii(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_set_max_iter_count_dzi(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_set_max_iter_count_dzz(
	sparse_cg_handle_t handle, const size_t max_iters );

sparse_err_t sparse_cg_solve_sii(
	sparse_cg_handle_t handle, float * const x, const float * const b );

sparse_err_t sparse_cg_solve_szi(
	sparse_cg_handle_t handle, float * const x, const float * const b );

sparse_err_t sparse_cg_solve_szz(
	sparse_cg_handle_t handle, float * const x, const float * const b );

sparse_err_t sparse_cg_solve_dii(
	sparse_cg_handle_t handle, double * const x, const double * const b );

sparse_err_t sparse_cg_solve_dzi(
	sparse_cg_handle_t handle, double * const x, const double * const b );

sparse_err_t sparse_cg_solve_dzz(
	sparse_cg_handle_t handle, double * const x, const double * const b );

sparse_err_t sparse_cg_destroy_sii( sparse_cg_handle_t handle );

sparse_err_t sparse_cg_destroy_szi( sparse_cg_handle_t handle );

sparse_err_t sparse_cg_destroy_szz( sparse_cg_handle_t handle );

sparse_err_t sparse_cg_destroy_dii( sparse_cg_handle_t handle );

sparse_err_t sparse_cg_destroy_dzi( sparse_cg_handle_t handle );

sparse_err_t sparse_cg_destroy_dzz( sparse_cg_handle_t handle );

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end ifdef _H_ALP_LINSOLVERS

