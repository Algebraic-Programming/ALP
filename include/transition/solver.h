
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
 * Defines the solver transition path API.
 *
 * @see blas_sparse.h for the SparseBLAS transition path API
 * @see spblas_impl.h for the SpBLAS transition path API
 *
 * @author A. N. Yzelman
 * @date 5th of October, 2023
 */

/**
 * \defgroup TRANS_SOLVERS Solvers
 * \ingroup TRANS
 * @{
 *
 * This exposes a transition path API to the linear system solvers implemented
 * in ALP. The primary benefit compared to simply using a SpBLAS or SparseBLAS
 * interface, is that solvers herein defined can be compiled using the
 * nonblocking backend, thus automatically optimising across (Sparse)BLAS
 * primitives. This is an experimental feature designed to evaluate exactly
 * this advantage.
 *
 * The C-style interface here defined expects the industry-standard Compressed
 * Row Storage (CRS) for matrices, also known as CSR. It employs a systematic
 * postfix to the functions it defines. For example, the basic functions of the
 * Conjugate Gradient solver are:
 *  - <tt>sparse_cg_init_xyy</tt>,
 *  - <tt>sparse_cg_solve_xyy</tt>, and
 *  - <tt>sparse_cg_destroy_xyy</tt>.
 *
 * \parblock
 * \par The postfix system and the sparse matrix storage format
 *
 * First explaining the postfix system, each of the above <tt>x</tt> characters
 * may be <tt>d</tt> or <tt>s</tt>, indicating the precision used during the
 * solve: double- or single-precision, respectively.
 *
 * Each of the <tt>y</tt> characters may be <tt>z</tt> or <tt>i</tt>, indicating
 * the integer type used during the solve: a <tt>size_t</tt> or a regular
 * <tt>int</tt>, respectively.
 *
 * The first <tt>y</tt> character indicates the integer type of the CRS column
 * index array, which maintains for each nonzero entry on which column it
 * resides. For really large dimension matrices, a <tt>size_t</tt> integer
 * type, which usually defaults to 64-bit unsigned integers, may be required.
 *
 * The second <tt>y</tt> character indicates the integer type of the row offset
 * array of the CRS. The \f$ i \f$-th contiguous pair \f$ (a, b] \f$ of this
 * array indicates where in the value and column arrays the \f$ i \f$-th row of
 * the matrix is encoded-- the start position \f$ a \f$ is inclusive whereas the
 * end position \f$ b \f$ is exclusive. The row offset array is of size
 * \f$ m + 1 \f$, where \f$ m \f$ is the number of rows in the matrix. Hence the
 * last entry of the offset array is the total number of nonzeroes the matrix
 * contains. The data type of the offset array must be <tt>size_t</tt> when
 * matrices contain many nonzeroes.
 *
 * \note Usually, if the offset array must be of type <tt>size_t</tt>, then the
 *       column index array must also be of type <tt>size_t</tt>. Certainly for
 *       matrices used with (linear) solvers, after all, the number of
 *       nonzeroes is a multiple of the number of matrix rows.
 * \endparblock
 *
 * \parblock
 * \par Implemented solvers
 *
 * Currently, the following ALP solvers are exposed:
 *  -# the sparse Conjugate Gradient (CG) solver, implemented at
 *     #grb::algorithms::conjugate_gradient.
 *
 * If you require any other solver, please feel free to submit a feature request
 * or to contact the maintainer.
 * \endparblock
 *
 * \warning The solvers here defined, and the transition path functionalities as a
 *          whole, are currently in an experimental prototype stage.
 */

#ifndef _H_ALP_SPARSE_LINSOLVERS
#define _H_ALP_SPARSE_LINSOLVERS

#include <stddef.h> // for size_t


#ifdef __cplusplus
extern "C" {
#endif

/**
 * The various error codes sparse solver library functions may return.
 */
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

	/**
	 * The algorithm has failed achieving its intended result. For example, an
	 * iterative solver did not converge.
	 */
	FAILED,

	/**
	 * An unknown error has been encountered. The state of the underlying solver
	 * library has become undefined.
	 */
	UNKNOWN

} sparse_err_t;

/**
 * A solver handle for the Conjugate Gradient algorithm.
 */
typedef void * sparse_cg_handle_t;

/**
 * A user-defined preconditioner function type for CG solver that employ
 * single-precision floating-point nonzero values.
 *
 * I.e. and more precisely, a preconditioner function type for CG solver handles
 * of types * <tt>sii</tt>, <tt>siz</tt>, and <tt>szz</tt>.
 *
 * A preconditioner is assumed to be a plain C function pointer, where
 *  -# the function returns an <tt>int</tt> error code (where zero will be
 *     interpreted as success);
 *  -# the first argument is where the result of applying the preconditioner
 *     will be stored. It is a raw vector pointer, i.e., <tt>float *</tt>;
 *  -# the second argument contains the data on which the preconditioner
 *     action should be computed. It is a raw const vector pointer, i.e.,
 *     <tt>const float*</tt>;
 *  -# the third argument contains a pointer to any preconditioner data it
 *     may require. It is a raw void pointer, meaning, although usually not
 *     necessary nor recommended, the preconditioner data may be stateful.
 *
 * The function signature must match exactly this specification.
 */
typedef int (*sparse_cg_preconditioner_sxx_t) (
	float * const,
	const float * const,
	void * const
);

/**
 * A user-defined preconditioner function type for CG solver that employ
 * double-precision floating-point nonzero values.
 *
 * I.e. and more precisely, a preconditioner function type for CG solver handles
 * of types * <tt>dii</tt>, <tt>diz</tt>, and <tt>dzz</tt>.
 *
 * A preconditioner is assumed to be a plain C function pointer, where
 *  -# the function returns an <tt>int</tt> error code (where zero will be
 *     interpreted as success);
 *  -# the first argument is where the result of applying the preconditioner
 *     will be stored. It is a raw vector pointer, i.e., <tt>double *</tt>;
 *  -# the second argument contains the data on which the preconditioner
 *     action should be computed. It is a raw const vector pointer, i.e.,
 *     <tt>const double *</tt>;
 *  -# the third argument contains a pointer to any preconditioner data it
 *     may require. It is a raw void pointer, meaning, although usually not
 *     necessary nor recommended, the preconditioner data may be stateful.
 *
 * The function signature must match exactly this specification.
 */
typedef int (*sparse_cg_preconditioner_dxx_t) (
	double * const,
	const double * const,
	void * const
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * @param[out] handle An uninitialised handler to a CG solver.
 *
 * Initialisation proceeds according to a given system matrix given in
 * Compressed Row Storage (CRS), also commonly known as the Compressed Sparse
 * Rows (CSR) format.
 *
 * @param[in]  n      The size of the system matrix.
 * @param[in]  a      The nonzero values of the system matrix.
 * @param[in]  ja     The column indices of the nonzeroes of the system matrix.
 * @param[in]  ia     The row offset array of the system matrix.
 *
 * This variant is for single-precision floating point nonzeroes and integer
 * \a ja and \a ia arrays, as also indicated by the <tt>sii</tt> postfix.
 *
 * @returns #ILLEGAL_ARGUMENT If \a n equals zero.
 * @returns #NULL_ARGUMENT    If \a handle is <tt>NULL</tt>.
 * @returns #NULL_ARGUMENT    If any of \a a, \a ja, or \a ia are <tt>NULL</tt>.
 * @returns #OUT_OF_MEMORY    In case of encountering out-of-memory conditions.
 *
 * On returning any of the above errors, the call to this function shall not have
 * any other effects (than having returned the error code).
 *
 * \note This implies in particular that any initially given invalid and
 *       non-<tt>NULL</tt> \a handle may be reused for future, potentially
 *       successful, calls to this (or any other) initialisation function.
 *
 * @returns #NO_ERROR         If initialisation of the handle proceeded
 *                            successfully. Only in this case shall \a handle
 *                            henceforth be a \em valid handle.
 */
sparse_err_t sparse_cg_init_sii(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const int * const ja, const int * const ia
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * This variant is for double-precision floating point nonzeroes and integer
 * \a ja and \a ia arrays, as also indicated by the <tt>dii</tt> postfix.
 *
 * @see #sparse_cg_init_sii for full documentation.
 */
sparse_err_t sparse_cg_init_dii(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const int * const ja, const int * const ia
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * This variant is for single-precision floating point nonzeroes,
 * <tt>size_t</tt>-valued \a ja, and integer-valued \a ia, as also indicated by
 * the <tt>siz</tt> postfix.
 *
 * @see #sparse_cg_init_sii for full documentation.
 */
sparse_err_t sparse_cg_init_siz(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const int * const ja, const size_t * const ia
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * This variant is for double-precision floating point nonzeroes,
 * <tt>size_t</tt>-valued \a ja, and integer-valued \a ia, as also indicated by
 * the <tt>diz</tt> postfix.
 *
 * @see #sparse_cg_init_sii for full documentation.
 */
sparse_err_t sparse_cg_init_diz(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const int * const ja, const size_t * const ia
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * This variant is for single-precision floating point nonzeroes and
 * <tt>size_t</tt>-valued \a ja and \a ia, as also indicated by the <tt>szz</tt>
 * postfix.
 *
 * @see #sparse_cg_init_sii for full documentation.
 */
sparse_err_t sparse_cg_init_szz(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const size_t * const ja, const size_t * const ia
);

/**
 * Initialises a #sparse_cg_handle_t object.
 *
 * This variant is for double-precision floating point nonzeroes and
 * <tt>size_t</tt>-valued \a ja and \a ia, as also indicated by the <tt>dzz</tt>
 * postfix.
 *
 * @see #sparse_cg_init_sii for full documentation.
 */
sparse_err_t sparse_cg_init_dzz(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const size_t * const ja, const size_t * const ia
);

// Note that szi and dzi are skipped on purpose. Such variants would not seem
// sensible, though could easily be provided if they do turn out to be needed


/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * @param[in]  handle A handle to a valid CG solver object.
 * @param[out] tol    Where to store the currently effective tolerance.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. If this error is
 *                         returned, the call to this function shall not have
 *                         any other effects.
 * @returns #NULL_ARGUMENT If \a tol is <tt>NULL</tt>. If this error code is
 *                         returned, the call to this function shall have no
 *                         other effects.
 * @returns #NO_ERROR      Otherwise.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance, the effect
 *          of calling this function is undefined(!).
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_sii(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_get_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_siz(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_get_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_szz(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_get_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_dii(
	const sparse_cg_handle_t handle, double * const tol );

/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_get_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_diz(
	const sparse_cg_handle_t handle, double * const tol );

/**
 * Gets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_get_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_get_tolerance_dzz(
	const sparse_cg_handle_t handle, double * const tol );


/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * @param[in,out] handle A handle to a valid CG solver object.
 * @param[in]     tol    The given tolerance.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. If this error is
 *                         returned, the call to this function shall not have
 *                         any other effects.
 * @returns #NO_ERROR      Otherwise.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance, the effect
 *          of calling this function is undefined(!).
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_sii(
	sparse_cg_handle_t handle, const float tol );

/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_siz(
	sparse_cg_handle_t handle, const float tol );

/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_szz(
	sparse_cg_handle_t handle, const float tol );

/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_dii(
	sparse_cg_handle_t handle, const double tol );

/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_diz(
	sparse_cg_handle_t handle, const double tol );

/**
 * Sets the current accepted relative tolerance for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_tolerance_dzz(
	sparse_cg_handle_t handle, const double tol );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * @param[in,out] handle A handle to a valid CG solver object.
 * @param[in]  max_iters The given maximum number of iterations.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. If this error is
 *                         returned, the call to this function shall not have
 *                         any other effects.
 * @returns #NO_ERROR      Otherwise.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance, the effect
 *          of calling this function is undefined(!).
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_sii(
	sparse_cg_handle_t handle, const size_t max_iters );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_siz(
	sparse_cg_handle_t handle, const size_t max_iters );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_szz(
	sparse_cg_handle_t handle, const size_t max_iters );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_dii(
	sparse_cg_handle_t handle, const size_t max_iters );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_diz(
	sparse_cg_handle_t handle, const size_t max_iters );

/**
 * Sets the current maximum number of iterations for the given CG solver.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_set_tolerance_sii for full documentation.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_set_max_iter_count_dzz(
	sparse_cg_handle_t handle, const size_t max_iters );

// other variants of sparse_cg_set_max_iter_count could take int, uint, etc.
// inputs


/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * @param[in]  handle A handle to a valid CG solver object.
 * @param[out] tol    Where to store the requested residual.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. If this error is
 *                         returned, the call to this function shall not have
 *                         any other effects.
 * @returns #NULL_ARGUMENT If \a tol is <tt>NULL</tt>. If this error code is
 *                         returned, the call to this function shall have no
 *                         other effects.
 * @returns #NO_ERROR      Otherwise.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance, the effect
 *          of calling this function is undefined(!).
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_sii On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_sii(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_get_residual_sii for full documentation.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_siz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_siz(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_get_residual_sii for full documentation.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_szz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_szz(
	const sparse_cg_handle_t handle, float * const tol );

/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_get_residual_sii for full documentation.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_dii On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_dii(
	const sparse_cg_handle_t handle, double * const tol );

/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_get_residual_sii for full documentation.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_diz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_diz(
	const sparse_cg_handle_t handle, double * const tol );

/**
 * Retrieves the residual the given CG solve has achieved.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_get_residual_sii for full documentation.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_dzz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write infinity into \a tol.
 */
sparse_err_t sparse_cg_get_residual_dzz(
	const sparse_cg_handle_t handle, double * const tol );


/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * @param[in]  handle A handle to a valid CG solver object.
 * @param[out] iters  Where to store the requested number of iterations.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. If this error is
 *                         returned, the call to this function shall not have
 *                         any other effects.
 * @returns #NULL_ARGUMENT If \a iters is <tt>NULL</tt>. If this error code is
 *                         returned, the call to this function shall have no
 *                         other effects.
 * @returns #NO_ERROR      Otherwise.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance, the effect
 *          of calling this function is undefined(!).
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_sii On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_sii(
	const sparse_cg_handle_t handle, size_t * const iters );

/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_get_iter_count_sii for full documentation.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_siz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_siz(
	const sparse_cg_handle_t handle, size_t * const iters );

/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_get_iter_count_sii for full documentation.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_szz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_szz(
	const sparse_cg_handle_t handle, size_t * const iters );

/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_get_iter_count_sii for full documentation.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_dii On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_dii(
	const sparse_cg_handle_t handle, size_t * const iters );

/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_get_iter_count_sii for full documentation.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_diz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_diz(
	const sparse_cg_handle_t handle, size_t * const iters );

/**
 * Retrieves the number of iterations the given CG solver has employed.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_get_iter_count_sii for full documentation.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 *
 * @see #sparse_cg_solve_dzz On how to execute a CG solve on a valid handle.
 *
 * \note Only after successful execution of a solver instance will a call to
 *       this function be useful; a valid instance that was freshly constructed
 *       will otherwise always write zero into \a iters.
 */
sparse_err_t sparse_cg_get_iter_count_dzz(
	const sparse_cg_handle_t handle, size_t * const iters );

// another variant of sparse_cg_get_iter_count could provide output as an int,
// uint, etc.

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_sxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_sii  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_sii On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_sii(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_sxx_t preconditioner,
	void * const data
);

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_dxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>dii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_dii  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_dii On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_dii(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_dxx_t preconditioner,
	void * const data
);

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_sxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_siz  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_siz On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_siz(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_sxx_t preconditioner,
	void * const data
);

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_dxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>dii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_diz  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_diz On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_diz(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_dxx_t preconditioner,
	void * const data
);

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_sxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>sii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_szz  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_szz On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_szz(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_sxx_t preconditioner,
	void * const data
);

/**
 * Sets a new preconditioner to apply during a next solve call.
 *
 * @param[in,out] handle         A handle to a valid CG solver object.
 * @param[in]     preconditioner The preconditioner as a C function pointer.
 *
 * @see #sparse_cg_preconditioner_dxx_t On the required signature for
 *                                      \a preconditioner.
 *
 * @param[in]     data           Pointer to any data the preconditioner may
 *                               require.
 *
 * This variant is for CG solver handles of type <tt>dii</tt>.
 *
 * \warning If \a handle did not refer to a valid CG solver instance of a
 *          matching type, the effect of calling this function is undefined(!)
 *
 * @see #sparse_cg_init_dzz  On how to obtain a valid CG solver instance for use
 *                           with this function.
 * @see #sparse_cg_solve_dzz On how to call the resulting preconditioned solver.
 */
sparse_err_t sparse_cg_set_preconditioner_dzz(
	const sparse_cg_handle_t handle,
	const sparse_cg_preconditioner_dxx_t preconditioner,
	void * const data
);

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * @param[in]     handle A handle to a valid CG solver object, which embeds the
 *                       linear system matrix on which the solve is executed.
 * @param[in,out] x      On input: an initial guess to the solution. On output:
 *                       the last-obtained iterative refinement of the initial
 *                       guess.
 * @param[in]     b      The right-hand side of the linear system to solve.
 *
 * The solve continues until convergence, until the maximum number of iterations
 * has been achieved, or until an error is encountered.
 *
 * Calling this function with an invalid \a handle will incur undefined
 * behaviour.
 *
 * @returns #NULL_ARGUMENT If one or more of \a handle, \a x, or \a b is
 *                         <tt>NULL</tt>. In this case, the call to this
 *                         function shall not have any other effects.
 * @returns #FAILED        If the solver did not converge to the given relative
 *                         tolerance within the given number of maximum
 *                         iterations. In this case, \a x contains the last
 *                         iteratively refined guess to the solution.
 * @returns #NO_ERROR      If an acceptable solution has been found.
 *
 * @see #sparse_cg_get_residual_sii to retrieve the residual of \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * @returns #UNKNOWN If the solver failed due to any other error. In this case,
 *                   the state of the solver library and the contents of \a x
 *                   shall be undefined and the user is urged to gracefully exit
 *                   the application at the next available opportunity, and in
 *                   any case to not make any further calls into the solver
 *                   library.
 *
 * @see #sparse_cg_init_sii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_sii(
	sparse_cg_handle_t handle, float * const x, const float * const b );

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_solve_sii for full documentation.
 *
 * @see #sparse_cg_get_residual_siz to retrieve the last-known residual of
 *                                  \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * \note The contents of \a x are only modified by this solve call if it returns
 *       #NO_ERROR or #FAILED. The contents of \a x will be guaranteed
 *       unmodified from the initial guess if the function returns
 *       #NULL_ARGUMENT. The contents of \a x will be undefined if the function
 *       returns #UNKNOWN.
 *
 * @see #sparse_cg_init_siz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_siz(
	sparse_cg_handle_t handle, float * const x, const float * const b );

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_solve_sii for full documentation.
 *
 * @see #sparse_cg_get_residual_szz to retrieve the last-known residual of
 *                                  \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * \note The contents of \a x are only modified by this solve call if it returns
 *       #NO_ERROR or #FAILED. The contents of \a x will be guaranteed
 *       unmodified from the initial guess if the function returns
 *       #NULL_ARGUMENT. The contents of \a x will be undefined if the function
 *       returns #UNKNOWN.
 *
 * @see #sparse_cg_init_szz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_szz(
	sparse_cg_handle_t handle, float * const x, const float * const b );

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_solve_sii for full documentation.
 *
 * @see #sparse_cg_get_residual_dii to retrieve the last-known residual of
 *                                  \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * \note The contents of \a x are only modified by this solve call if it returns
 *       #NO_ERROR or #FAILED. The contents of \a x will be guaranteed
 *       unmodified from the initial guess if the function returns
 *       #NULL_ARGUMENT. The contents of \a x will be undefined if the function
 *       returns #UNKNOWN.
 *
 * @see #sparse_cg_init_dii On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_dii(
	sparse_cg_handle_t handle, double * const x, const double * const b );

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_solve_sii for full documentation.
 *
 * @see #sparse_cg_get_residual_diz to retrieve the last-known residual of
 *                                  \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * \note The contents of \a x are only modified by this solve call if it returns
 *       #NO_ERROR or #FAILED. The contents of \a x will be guaranteed
 *       unmodified from the initial guess if the function returns
 *       #NULL_ARGUMENT. The contents of \a x will be undefined if the function
 *       returns #UNKNOWN.
 *
 * @see #sparse_cg_init_diz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_diz(
	sparse_cg_handle_t handle, double * const x, const double * const b );

/**
 * Executes a solve using a given CG solver handle, a given right-hand side
 * \a b, and an initial guess \a x.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_solve_sii for full documentation.
 *
 * @see #sparse_cg_get_residual_dzz to retrieve the last-known residual of
 *                                  \a x.
 *
 * \note Retrieving the residual may be of interest both on convergence
 *       (#NO_ERROR) and when convergence was not obtained (#FAILED).
 *
 * \note The contents of \a x are only modified by this solve call if it returns
 *       #NO_ERROR or #FAILED. The contents of \a x will be guaranteed
 *       unmodified from the initial guess if the function returns
 *       #NULL_ARGUMENT. The contents of \a x will be undefined if the function
 *       returns #UNKNOWN.
 *
 * @see #sparse_cg_init_dzz On how to obtain a valid CG solver instance for use
 *                          with this function.
 */
sparse_err_t sparse_cg_solve_dzz(
	sparse_cg_handle_t handle, double * const x, const double * const b );


/**
 * Destroys a valid CG solver handle.
 *
 * @param[in,out] handle A handle to a valid CG solver object.
 *
 * @returns #NULL_ARGUMENT If \a handle is <tt>NULL</tt>. When returning this
 *                         error code, the call to this function shall have no
 *                         other effects.
 * @returns #NO_ERROR      If the given \a handle was successfully destroyed.
 *                         The handle shall henceforth be invalid.
 *
 * This variant is for CG solver instances of type <tt>sii</tt>.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>sii</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_sii( sparse_cg_handle_t handle );

/**
 * Destroys a valid CG solver handle.
 *
 * This variant is for CG solver instances of type <tt>siz</tt>.
 *
 * @see #sparse_cg_destroy_sii for full documentation.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>siz</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_siz( sparse_cg_handle_t handle );

/**
 * Destroys a valid CG solver handle.
 *
 * This variant is for CG solver instances of type <tt>szz</tt>.
 *
 * @see #sparse_cg_destroy_sii for full documentation.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>szz</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_szz( sparse_cg_handle_t handle );

/**
 * Destroys a valid CG solver handle.
 *
 * This variant is for CG solver instances of type <tt>dii</tt>.
 *
 * @see #sparse_cg_destroy_sii for full documentation.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>dii</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_dii( sparse_cg_handle_t handle );

/**
 * Destroys a valid CG solver handle.
 *
 * This variant is for CG solver instances of type <tt>diz</tt>.
 *
 * @see #sparse_cg_destroy_sii for full documentation.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>diz</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_diz( sparse_cg_handle_t handle );

/**
 * Destroys a valid CG solver handle.
 *
 * This variant is for CG solver instances of type <tt>dzz</tt>.
 *
 * @see #sparse_cg_destroy_sii for full documentation.
 *
 * \note After a call to this function, the newly-invalid \a handle may be
 *       re-initialised, even for initialising non-<tt>dzz</tt> solvers.
 */
sparse_err_t sparse_cg_destroy_dzz( sparse_cg_handle_t handle );

/**@}*/ // ends doxygen page for the solver library

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end ifdef _H_ALP_LINSOLVERS

