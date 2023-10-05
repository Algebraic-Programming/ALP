
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

#include "solvers.h"

#include <assert.h>

#include <array>

#include <graphblas.hpp>


template< typename T, typename NZI, typename RSI >
class CG_Data {

	private:

		typedef grb::Matrix< T, grb::config::default_backend, RSI, RSI, NZI > Matrix;

		// input args
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
			const T * const a, const NZI * const ja, const RSI * const ia
		) :
			tolerance( 1e-5 ), max_iter( 1000 ),
			residual( std::numeric_limits< T >::infinity() ), iters( 0 ),
			workspace( {
				grb::Vector< T >( n ), grb::Vector< T >( n ), grb::Vector< T >( n )
			} )
		{
			assert( n > 0 );
			assert( a != nullptr );
			assert( ja != nullptr );
			assert( ia != nullptr );
			const size_t nz = static_cast< size_t >( ia[ n ] );
			matrix = grb::internal::wrapCRSMatrix( a, ja, ia, n, n );
		}

		T getTolerance() const noexcept { return tolerance; }

		T getResidual() const noexcept { return residual; }

		size_t getIters() const noexcept { return iters; }

		void setMaxIters( const size_t &in ) const noexcept { max_iter = in; }

		void setTolerance( const T &in ) const noexcept { tolerance = in; }

		grb::RC solve( grb::Vector< T > &x, const grb::Vector< T > &b ) {
			return grb::algorithms::conjugate_gradient(
				x, matrix, b,
				max_iter, tolerance,
				iters, residual,
				workspace[ 0 ], workspace[ 1 ], workspace[ 2 ]
			);
		}

};

extern "C" {

	sparse_err_t sparse_cg_init_sii(
		sparse_cg_handle_t * const handle, const size_t n,
		const float * const a, const int * const ja, const int * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * >(
				new CG_Data< float, int, int >( n, a, ja, ia ) );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_init_dii(
		sparse_cg_handle_t * const handle, const size_t n,
		const double * const a, const int * const ja, const int * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * >(
				new CG_Data< double, int, int >( n, a, ja, ia ) );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_init_szi(
		sparse_cg_handle_t * const handle, const size_t n,
		const float * const a, const size_t * const ja, const int * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * > (
				new CG_Data< float, size_t, int >( n, a, ja, ia ) );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_init_dzi(
		sparse_cg_handle_t * const handle, const size_t n,
		const double * const a, const size_t * const ja, const int * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * >(
				new CG_Data< double, size_t, int >( n, a, ja, ia );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_init_szz(
		sparse_cg_handle_t * const handle, const size_t n,
		const float * const a, const size_t * const ja, const size_t * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * >(
				new CG_Data< float, size_t, size_t >( n, a, ja, ia ) );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_init_dzz(
		sparse_cg_handle_t * const handle, const size_t n,
		const double * const a, const size_t * const ja, const size_t * const ia
	) {
		if( n == 0 ) { return ILLEGAL_ARGUMENT };
		if( handle == nullptr || a == nullptr || ja == nullptr || ia == nullptr ) {
			return NULL_ARGUMENT;
		}
		try {
			*handle = static_cast< void * >(
				new CG_Data< double, size_t, size_t >( n, a, ja, ia ) );
		} catch( std::exception &e ) {
			// the grb::Matrix constructor may only throw on out of memory errors
			std::cerr << "Error: " << e.what() << "\n";
			*handle = nullptr;
			return OUT_OF_MEMORY;
		}
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_sii(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol = static_cast< CG_data< float, int, int > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_szi(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< float, size_t, int > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_szz(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< float, size_t, size_t > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_dii(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol = static_cast< CG_data< double, int, int > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_dzi(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< double, size_t, int > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_tolerance_dzz(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< double, size_t, size_t > * >( handle ).getTolerance();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_sii(
		sparse_cg_handle_t handle, const float tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, int, int > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_szi(
		sparse_cg_handle_t handle, const float tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, size_t, int > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_szz(
		sparse_cg_handle_t handle, const float tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, size_t, size_t > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_dii(
		sparse_cg_handle_t handle, const double tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, int, int > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_dzi(
		sparse_cg_handle_t handle, const double tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, size_t, int > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_tolerance_dzz(
		sparse_cg_handle_t handle, const double tol
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, size_t, int > * >( handle ).
			setTolerance( tol );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_sii(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol = static_cast< CG_data< float, int, int > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_szi(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< float, size_t, int > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_szz(
		const sparse_cg_handle_t handle, float * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< float, size_t, size_t > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_dii(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol = static_cast< CG_data< double, int, int > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_dzi(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< double, size_t, int > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_residual_dzz(
		const sparse_cg_handle_t handle, double * const tol
	) {
		if( handle == nullptr || tol == nullptr ) { return NULL_ARGUMENT; }
		*tol =
			static_cast< CG_data< double, size_t, size_t > * >( handle ).getResidual();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_sii(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters = static_cast< CG_data< float, int, int > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_szi(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters = static_cast< CG_data< float, size_t, int > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_szz(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters =
			static_cast< CG_data< float, size_t, size_t > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_dii(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters = static_cast< CG_data< double, int, int > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_dzi(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters = static_cast< CG_data< double, size_t, int > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_get_iter_count_dzz(
		const sparse_cg_handle_t handle, size_t * const iters
	) {
		if( handle == nullptr || iters == nullptr ) { return NULL_ARGUMENT; }
		*iters =
			static_cast< CG_data< double, size_t, size_t > * >( handle ).getIters();
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_sii(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, int, int > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_szi(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, size_t, int > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_szz(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< float, size_t, size_t > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_dii(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, int, int > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_dzi(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, size_t, int > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_set_max_iter_count_dzz(
		sparse_cg_handle_t handle, const size_t max_iters
	) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		static_cast< CG_data< double, size_t, size_t > * >( handle ).
			setMaxIters( max_iters );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_solve_sii(
		sparse_cg_handle_t handle, float * const x, const float * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< float, int, int > * >( handle );
		grb::Vector< float > alp_x =
			grb::internal::template wrapRawVector< float >( data.n, x );
		const grb::Vector< float > alp_b =
			grb::internal::template wrapRawVector< float >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_solve_szi(
		sparse_cg_handle_t handle, float * const x, const float * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< float, size_t, int > * >( handle );
		grb::Vector< float > alp_x =
			grb::internal::template wrapRawVector< float >( data.n, x );
		const grb::Vector< float > alp_b =
			grb::internal::template wrapRawVector< float >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_solve_szz(
		sparse_cg_handle_t handle, float * const x, const float * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< float, size_t, size_t > * >( handle );
		grb::Vector< float > alp_x =
			grb::internal::template wrapRawVector< float >( data.n, x );
		const grb::Vector< float > alp_b =
			grb::internal::template wrapRawVector< float >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_solve_dii(
		sparse_cg_handle_t handle, double * const x, const double * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< double, int, int > * >( handle );
		grb::Vector< double > alp_x =
			grb::internal::template wrapRawVector< double >( data.n, x );
		const grb::Vector< double > alp_b =
			grb::internal::template wrapRawVector< double >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_solve_dzi(
		sparse_cg_handle_t handle, double * const x, const double * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< double, size_t, int > * >( handle );
		grb::Vector< double > alp_x =
			grb::internal::template wrapRawVector< double >( data.n, x );
		const grb::Vector< double > alp_b =
			grb::internal::template wrapRawVector< double >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_solve_dzz(
		sparse_cg_handle_t handle, double * const x, const double * const b
	) {
		if( handle == nullptr || x == nullptr || b == nullptr ) {
			return NULL_ARGUMENT;
		}
		auto &data = * static_cast< CG_data< double, size_t, size_t > * >( handle );
		grb::Vector< double > alp_x =
			grb::internal::template wrapRawVector< double >( data.n, x );
		const grb::Vector< double > alp_b =
			grb::internal::template wrapRawVector< double >( data.n, b );
		const grb::RC rc = data.solve( x, b );
		// ALP spec should not allow going out of memory
		assert( rc != grb::OUTOFMEM );
		// should we have a return code for failed convergence?
		if( rc == grb::FAILED ) {
			std::cerr << "Warning: call to sparse_cg_solve_??? did not converge\n";
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

	sparse_err_t sparse_cg_destroy_sii( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< float, int, int > * >( handle );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_destroy_szi( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< float, size_t, int > * >( handle );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_destroy_szz( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< float, size_t, size_t > * >( handle );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_destroy_dii( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< double, int, int > * >( handle );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_destroy_dzi( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< double, size_t, int > * >( handle );
		return NO_ERROR;
	}

	sparse_err_t sparse_cg_destroy_dzz( sparse_cg_handle_t handle ) {
		if( handle == nullptr ) { return NULL_ARGUMENT; }
		delete static_cast< CG_data< double, size_t, size_t > * >( handle );
		return NO_ERROR;
	}

} // end extern "C"

