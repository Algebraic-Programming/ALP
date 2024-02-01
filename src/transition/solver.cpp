
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


#include <assert.h>
#include <array>

#include <graphblas.hpp>
#include <graphblas/algorithms/conjugate_gradient.hpp>

#include "solver.h"

/**
 * @file
 * This implements a transition path API to the linear system solvers.
 *
 * @author Alberto Scolari
 * @date 26/01/2024
 */

/**
 * @tparam T   The nonzero value type.
 * @tparam NZI The nonzero index type.
 * @tparam RSI The row and column index type.
 */
template< typename T, typename NZI, typename RSI >
class CG_Data {

	private:

		typedef grb::Matrix< T, grb::config::default_backend, RSI, RSI, NZI > Matrix;

		// input args
		size_t size;
		T tolerance;
		size_t max_iter;
		Matrix matrix;

		// outputs
		T residual;
		size_t iters;

		// workspace
		std::array< grb::Vector< T >, 3 > workspace;

	public:

		CG_Data() = delete;

		CG_Data(
			const size_t n,
			const T * const a, const RSI * const ja, const NZI * const ia
		) :
			size( n ), tolerance( 1e-5 ), max_iter( 1000 ), matrix( 0, 0 ),
			residual( std::numeric_limits< T >::infinity() ), iters( 0 ),
			workspace( {
				grb::Vector< T >( n ), grb::Vector< T >( n ), grb::Vector< T >( n )
			} )
		{
			assert( n > 0 );
			assert( a != nullptr );
			assert( ja != nullptr );
			assert( ia != nullptr );
			Matrix A = grb::internal::wrapCRSMatrix( a, ja, ia, n, n );
			std::swap( A, matrix );
		}

		size_t getSize() const noexcept { return size; }

		T getTolerance() const noexcept { return tolerance; }

		T getResidual() const noexcept { return residual; }

		size_t getIters() const noexcept { return iters; }

		void setMaxIters( const size_t &in ) noexcept { max_iter = in; }

		void setTolerance( const T &in ) noexcept { tolerance = in; }

		grb::RC solve( grb::Vector< T > &x, const grb::Vector< T > &b ) {
			constexpr grb::Descriptor descr =
				grb::descriptors::dense | grb::descriptors::force_row_major;
			return grb::algorithms::conjugate_gradient< descr >(
				x, matrix, b,
				max_iter, tolerance,
				iters, residual,
				workspace[ 0 ], workspace[ 1 ], workspace[ 2 ]
			);
		}

};


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_init_impl(
	sparse_cg_handle_t * const handle, const size_t n,
	const T * const a, const RSI * const ja, const NZI * const ia
) {
	if( n == 0 ) { return ILLEGAL_ARGUMENT; }
	if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
		return NULL_ARGUMENT;
	}
	try {
		*handle = static_cast< void * >(
			new CG_Data< T, NZI, RSI >( n, a, ja, ia ) );
	} catch( std::exception &e ) {
		// the grb::Matrix constructor may only throw on out of memory errors
		std::cerr << "Error: " << e.what() << "\n";
		*handle = nullptr;
		return OUT_OF_MEMORY;
	}
	return NO_ERROR;
}

sparse_err_t sparse_cg_init_sii(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const int * const ja, const int * const ia
) {
	return sparse_cg_init_impl< float, int, int >( handle, n, a, ja, ia );
}

sparse_err_t sparse_cg_init_dii(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const int * const ja, const int * const ia
) {
	return sparse_cg_init_impl< double, int, int >( handle, n, a, ja, ia );
}

sparse_err_t sparse_cg_init_siz(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const int * const ja, const size_t * const ia
) {
	return sparse_cg_init_impl< float, size_t, int >( handle, n, a, ja, ia );
}

sparse_err_t sparse_cg_init_diz(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const int * const ja, const size_t * const ia
) {
	return sparse_cg_init_impl< double, size_t, int >( handle, n, a, ja, ia );
}

sparse_err_t sparse_cg_init_szz(
	sparse_cg_handle_t * const handle, const size_t n,
	const float * const a, const size_t * const ja, const size_t * const ia
) {
	return sparse_cg_init_impl< float, size_t, size_t >( handle, n, a, ja, ia );
}

sparse_err_t sparse_cg_init_dzz(
	sparse_cg_handle_t * const handle, const size_t n,
	const double * const a, const size_t * const ja, const size_t * const ia
) {
	return sparse_cg_init_impl< double, size_t, size_t >( handle, n, a, ja, ia );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_get_tolerance_impl(
	const sparse_cg_handle_t handle, T * const tol
) {
	if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
	*tol = static_cast< CG_Data< T, NZI, RSI > * >( handle )->getTolerance();
	return NO_ERROR;
}

sparse_err_t sparse_cg_get_tolerance_sii(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_tolerance_impl< float, int, int >( handle, tol );
}

sparse_err_t sparse_cg_get_tolerance_siz(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_tolerance_impl< float, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_get_tolerance_szz(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_tolerance_impl< float, size_t, size_t >( handle, tol );
}

sparse_err_t sparse_cg_get_tolerance_dii(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_tolerance_impl< double, int, int >( handle, tol );
}

sparse_err_t sparse_cg_get_tolerance_diz(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_tolerance_impl< double, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_get_tolerance_dzz(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_tolerance_impl< double, size_t, size_t >( handle, tol );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_set_tolerance_impl(
	sparse_cg_handle_t handle, const T tol
) {
	if( handle == nullptr ) { return NULL_ARGUMENT; }
	static_cast< CG_Data< T, NZI, RSI > * >( handle )->setTolerance( tol );
	return NO_ERROR;
}

sparse_err_t sparse_cg_set_tolerance_sii(
	sparse_cg_handle_t handle, const float tol
) {
	return sparse_cg_set_tolerance_impl< float, int, int >( handle, tol );
}

sparse_err_t sparse_cg_set_tolerance_siz(
	sparse_cg_handle_t handle, const float tol
) {
	return sparse_cg_set_tolerance_impl< float, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_set_tolerance_szz(
	sparse_cg_handle_t handle, const float tol
) {
	return sparse_cg_set_tolerance_impl< float, size_t, size_t >( handle, tol );
}

sparse_err_t sparse_cg_set_tolerance_dii(
	sparse_cg_handle_t handle, const double tol
) {
	return sparse_cg_set_tolerance_impl< double, int, int >( handle, tol );
}

sparse_err_t sparse_cg_set_tolerance_diz(
	sparse_cg_handle_t handle, const double tol
) {
	return sparse_cg_set_tolerance_impl< double, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_set_tolerance_dzz(
	sparse_cg_handle_t handle, const double tol
) {
	return sparse_cg_set_tolerance_impl< double, size_t, size_t >( handle, tol );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_get_residual_impl(
	const sparse_cg_handle_t handle, T * const tol
) {
	if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
	*tol = static_cast< CG_Data< T, NZI, RSI > * >( handle )->getResidual();
	return NO_ERROR;
}

sparse_err_t sparse_cg_get_residual_sii(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_residual_impl< float, int, int >( handle, tol );
}

sparse_err_t sparse_cg_get_residual_siz(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_residual_impl< float, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_get_residual_szz(
	const sparse_cg_handle_t handle, float * const tol
) {
	return sparse_cg_get_residual_impl< float, size_t, size_t >( handle, tol );
}

sparse_err_t sparse_cg_get_residual_dii(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_residual_impl< double, int, int >( handle, tol );
}

sparse_err_t sparse_cg_get_residual_diz(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_residual_impl< double, size_t, int >( handle, tol );
}

sparse_err_t sparse_cg_get_residual_dzz(
	const sparse_cg_handle_t handle, double * const tol
) {
	return sparse_cg_get_residual_impl< double, size_t, size_t >( handle, tol );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_get_iter_count_impl(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
	*iters = static_cast< CG_Data< T, NZI, RSI > * >( handle )->getIters();
	return NO_ERROR;
}

sparse_err_t sparse_cg_get_iter_count_sii(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< float, int, int >( handle, iters );
}

sparse_err_t sparse_cg_get_iter_count_siz(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< float, size_t, int >( handle, iters );
}

sparse_err_t sparse_cg_get_iter_count_szz(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< float, size_t, size_t >( handle, iters );
}

sparse_err_t sparse_cg_get_iter_count_dii(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< double, int, int >( handle, iters );
}

sparse_err_t sparse_cg_get_iter_count_diz(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< double, size_t, int >( handle, iters );
}

sparse_err_t sparse_cg_get_iter_count_dzz(
	const sparse_cg_handle_t handle, size_t * const iters
) {
	return sparse_cg_get_iter_count_impl< double, size_t, size_t >( handle, iters );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_set_max_iter_count_impl(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	if( handle == nullptr ) { return NULL_ARGUMENT; }
	static_cast< CG_Data< T, NZI, RSI > * >( handle )->
		setMaxIters( max_iters );
	return NO_ERROR;
}

sparse_err_t sparse_cg_set_max_iter_count_sii(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< float, int, int >( handle,
		max_iters );
}

sparse_err_t sparse_cg_set_max_iter_count_siz(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< float, size_t, int >( handle,
		max_iters );
}

sparse_err_t sparse_cg_set_max_iter_count_szz(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< float, size_t, size_t >( handle,
		max_iters );
}

sparse_err_t sparse_cg_set_max_iter_count_dii(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< double, int, int >( handle,
		max_iters );
}

sparse_err_t sparse_cg_set_max_iter_count_diz(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< double, size_t, int >( handle,
		max_iters );
}

sparse_err_t sparse_cg_set_max_iter_count_dzz(
	sparse_cg_handle_t handle, const size_t max_iters
) {
	return sparse_cg_set_max_iter_count_impl< double, size_t, size_t >( handle,
		max_iters );
}


template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_solve_impl(
	sparse_cg_handle_t handle, T * const x, const T * const b
) {
	if( handle == nullptr || x == nullptr || b == nullptr ) {
		return NULL_ARGUMENT;
	}
	auto &data = * static_cast< CG_Data< T, NZI, RSI > * >( handle );
	grb::Vector< T > alp_x =
		grb::internal::template wrapRawVector< T >( data.getSize(), x );
	const grb::Vector< T > alp_b =
		grb::internal::template wrapRawVector< T >( data.getSize(), b );
	const grb::RC rc = data.solve( alp_x, alp_b );
	// ALP spec should not allow going out of memory
	assert( rc != grb::OUTOFMEM );
	// should we have a return code for failed convergence?
	if( rc == grb::FAILED ) {
		return FAILED;
	} else if( rc == grb::PANIC ) {
		return UNKNOWN;
	} else if( rc != grb::SUCCESS ) {
		std::cerr << "Warning: ALP should not have returned the following error\n"
			<< "\t" << grb::toString( rc ) << "\n"
			<< "Please submit a bug report.\n";
		return UNKNOWN;
	}
	return NO_ERROR;
}

sparse_err_t sparse_cg_solve_sii(
	sparse_cg_handle_t handle, float * const x, const float * const b
) {
	return sparse_cg_solve_impl< float, int, int >( handle, x, b );
}

sparse_err_t sparse_cg_solve_siz(
	sparse_cg_handle_t handle, float * const x, const float * const b
) {
	return sparse_cg_solve_impl< float, size_t, int >( handle, x, b );
}

sparse_err_t sparse_cg_solve_szz(
	sparse_cg_handle_t handle, float * const x, const float * const b
) {
	return sparse_cg_solve_impl< float, size_t, size_t >( handle, x, b );
}

sparse_err_t sparse_cg_solve_dii(
	sparse_cg_handle_t handle, double * const x, const double * const b
) {
	return sparse_cg_solve_impl< double, int, int >( handle, x, b );
}

sparse_err_t sparse_cg_solve_diz(
	sparse_cg_handle_t handle, double * const x, const double * const b
) {
	return sparse_cg_solve_impl< double, size_t, int >( handle, x, b );
}

sparse_err_t sparse_cg_solve_dzz(
	sparse_cg_handle_t handle, double * const x, const double * const b
) {
	return sparse_cg_solve_impl< double, size_t, size_t >( handle, x, b );
}

template< typename T, typename NZI, typename RSI >
static sparse_err_t sparse_cg_destroy_impl( sparse_cg_handle_t handle ) {
	if( handle == nullptr ) { return NULL_ARGUMENT; }
	delete static_cast< CG_Data< T, NZI, RSI > * >( handle );
	return NO_ERROR;
}

sparse_err_t sparse_cg_destroy_sii( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< float, int, int >( handle );
}

sparse_err_t sparse_cg_destroy_siz( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< float, size_t, int >( handle );
}

sparse_err_t sparse_cg_destroy_szz( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< float, size_t, size_t >( handle );
}

sparse_err_t sparse_cg_destroy_dii( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< double, int, int >( handle );
}

sparse_err_t sparse_cg_destroy_diz( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< double, size_t, int >( handle );
}

sparse_err_t sparse_cg_destroy_dzz( sparse_cg_handle_t handle ) {
	return sparse_cg_destroy_impl< double, size_t, size_t >( handle );
}

