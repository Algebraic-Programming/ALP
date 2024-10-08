
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
 * \ingroup TRANS_FUSELETS
 *
 * Implements a set of fused level-1 and level-2 ALP kernels.
 *
 * The fused kernels are designed to be easily callable from existing code
 * bases, using standard data structures such as raw pointers to vectors and the
 * Compressed Row Storage for sparse matrices.
 *
 * As a secondary goal, this standard set of fuselets demos the so-called ALP
 * <em>native interface</em>, show-casing the ease by which additional fuselets
 * that satisfy arbitrary needs can be added.
 *
 * @author A. N. Yzelman
 * @date 27/09/2024
 */

#include <graphblas.hpp>

#include <exception>
#include <iostream>

#include <assert.h>

#include "fuselets.h"


template< typename T >
using MySemiring = grb::semirings::PLUS_TIMES< T >;

template< typename T >
using MyAddMonoid = grb::monoids::PLUS< T >;

template< typename T >
using MyTimesMonoid = grb::monoids::TIMES< T >;

typedef MySemiring< double > DBL_SEMIRING;
typedef MyAddMonoid< double > DBL_PLUS_MONOID;
typedef MyTimesMonoid< double > DBL_TIMES_MONOID;

static DBL_SEMIRING dblSemiring;
static DBL_PLUS_MONOID dblPlusMonoid;
static DBL_TIMES_MONOID dblTimesMonoid;

int initialize_fuselets() {
	const grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		return 255;
	} else {
		return 0;
	}
}

int finalize_fuselets() {
	const grb::RC rc = grb::finalize();
	if( rc != grb::SUCCESS ) {
		return 255;
	} else {
		return 0;
	}
}

int update_spmv_dot(
	double * const p, double * const u, double * const alpha,
	const double * const z, const double beta,
	const size_t * const ia, const unsigned int * const ij,
	const double * const iv,
	const size_t n
) {
	// typedef our matrix type, which depends on the above argument types
	typedef grb::Matrix<
			double,                       // the matrix value type
			grb::config::default_backend, // use the compile-time selected backend (nonblocking)
			unsigned int, unsigned int,   // the types of the row- and column-indices
			size_t                        // the type of the ia array
		> MyMatrixType;

	// catch trivial op
	if( n == 0 ) {
		return 0;
	}

	// simple dynamic sanity checks
	assert( p != nullptr );
	assert( u != nullptr );
	assert( alpha != nullptr );
	assert( ia != nullptr );
	assert( ij != nullptr );
	assert( iv != nullptr );
	assert( ij[ n ] / n <= n );
#ifndef NDEBUG
	// we employ defensive programming and perform expensive input checks when
	// compiled in debug mode:
	for( size_t i = 0; i < n; ++i ) {
		assert( ia[ i + 1 ] >= ia[ i ] );
		for( size_t k = ia[ i ]; k < ia[ i + 1 ]; ++k ) {
			assert( ij[ k ] < n );
		}
	}
#endif

	// get ALP versions of (input/)output containers
	grb::Vector< double > alp_p =
		grb::internal::template wrapRawVector< double >( n, p );
	grb::Vector< double > alp_u =
		grb::internal::template wrapRawVector< double >( n, u );
	double &alp_alpha = *alpha;

	// do first op
	const grb::Vector< double > alp_z =
		grb::internal::template wrapRawVector< double >( n, z );
	grb::RC ret = grb::foldr< grb::descriptors::dense >(
		beta, alp_p,
		dblTimesMonoid
	);
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		alp_z, alp_p,
		dblPlusMonoid
	);
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 1: "
			<< grb::toString( ret ) << "\n";
		return 1;
	}

	// do second op
	const MyMatrixType alp_A = grb::internal::wrapCRSMatrix( iv, ij, ia, n, n );
	ret = grb::set< grb::descriptors::dense >( alp_u, 0.0 );
	ret = ret ? ret : grb::mxv<
		grb::descriptors::dense | grb::descriptors::force_row_major
	>( alp_u, alp_A, alp_p, dblSemiring );
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 2: "
			<< grb::toString( ret ) << "\n";
		return 2;
	}

	// do third op
	alp_alpha = 0.0;
	ret = grb::dot< grb::descriptors::dense >(
		alp_alpha, alp_u, alp_p, dblSemiring );
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 3: "
			<< grb::toString( ret ) << "\n";
		return 3;
	}

	// done
	ret = grb::wait();
	if( ret == grb::SUCCESS ) {
		return 0;
	} else {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error: "
			<< grb::toString( ret ) << "\n";
		return 255;
	}
}

int update_update_norm2(
	double * const x, double * const r, double * const norm2, // outputs
	const double alpha, const double * const p,               // input 1
	const double beta, const double * const u,                // input 2
	const size_t n                                            // size
) {
	// catch trivial op
	if( n == 0 ) {
		return 0;
	}

	// simple dynamic sanity checks
	assert( x != nullptr );
	assert( r != nullptr );
	assert( norm2 != nullptr );
	assert( p != nullptr );
	assert( u != nullptr );

	// get (input/)output containers
	grb::Vector< double > alp_x =
		grb::internal::template wrapRawVector< double >( n, x );
	grb::Vector< double > alp_r =
		grb::internal::template wrapRawVector< double >( n, r );
	double &alp_norm2 = *norm2;

	// perform operation 1
	const grb::Vector< double > alp_p =
		grb::internal::template wrapRawVector< double >( n, p );
	grb::RC ret = grb::foldr< grb::descriptors::dense >(
		static_cast< double >(1) / alpha, alp_x, dblTimesMonoid ); // 1/alpha x
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		alp_p, alp_x, dblPlusMonoid );                             // p + 1/alpha x
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		alpha, alp_x, dblTimesMonoid );                            // alpha p + x
	// this sequence costs n more divisions and affects numerical error
	// propagation-- the alternative is to pre-compute alpha p in a buffer, but
	// then we should be given such a buffer
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 1: "
			<< grb::toString( ret ) << "\n";
		return 1;
	}

	// perform operation 2
	const grb::Vector< double > alp_u =
		grb::internal::template wrapRawVector< double >( n, u );
	ret = grb::foldr< grb::descriptors::dense >(
		static_cast< double >(1) / beta, alp_r, dblTimesMonoid ); // 1/beta r
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		alp_u, alp_r, dblPlusMonoid );                            // u + 1/beta r
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		beta, alp_r, dblTimesMonoid );                            // beta u + r
	// same remark as above applies
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 2: "
			<< grb::toString( ret ) << "\n";
		return 2;
	}

	// perform operation 3
	alp_norm2 = 0.0;
	ret = grb::dot< grb::descriptors::dense >(
		alp_norm2, alp_r, alp_r, dblSemiring );
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error at operation 3: "
			<< grb::toString( ret ) << "\n";
		return 3;
	}

	// done
	ret = grb::wait();
	if( ret != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error: "
			<< grb::toString( ret ) << "\n";
		return 255;
	} else {
		return 0;
	}
}

int double_update(
	double * const p,
	const double alpha, const double * const r,
	const double beta, const double * const v,
	const double gamma,
	const size_t n
) {
	// check trivial op
	if( n == 0 ) {
		return 0;
	}

	// dynamic checks
	assert( p != nullptr );
	assert( r != nullptr );
	assert( v != nullptr );
	grb::Vector< double > alp_p =
		grb::internal::template wrapRawVector< double >( n, p );
	const grb::Vector< double > alp_r =
		grb::internal::template wrapRawVector< double >( n, r );
	const grb::Vector< double > alp_v =
		grb::internal::template wrapRawVector< double >( n, v );

	grb::RC rc = grb::SUCCESS;
	// p = gamma * p
	if( gamma != 1.0 ) {
		rc = grb::foldr( gamma, alp_p, dblTimesMonoid );
	} else if( gamma == 0.0 || gamma == -0.0 ) {
		rc = grb::set( alp_p, 0 );
	}

	if( beta != 0.0 && beta != -0.0 ) {
		if( beta != 1.0 ) {
			// p = (gamma .* p) / beta
			rc = rc ? rc :
				grb::foldr( static_cast< double >(1.0) / beta, alp_p, dblTimesMonoid );
		}
		// p = v + (gamma .* p) / beta
		rc = rc ? rc : grb::foldr( alp_v, alp_p, dblPlusMonoid );
		if( beta != 1.0 ) {
			// p = beta .* v + gamma .* p
			rc = rc ? rc : grb::foldr( beta, alp_p, dblTimesMonoid );
		}
	}

	if( alpha != 0.0 && beta != -0.0 ) {
		if( alpha != 1.0 ) {
			// p = (beta .* v + gamma .* p) / alpha
			rc = rc ? rc :
				grb::foldr( static_cast< double >(1.0) / alpha, alp_p, dblTimesMonoid );
		}
		// p = r + (beta .* v + gamma .* p) / alpha
		rc = rc ? rc : grb::foldr( alp_r, alp_p, dblPlusMonoid );
		if( alpha != 1.0 ) {
			// p = alpha .* r + beta .* v + gamma .* p
			rc = rc ? rc : grb::foldr( alpha, alp_p, dblTimesMonoid );
		}
	}

	// done
	rc = rc ? rc : grb::wait();
	if( rc != grb::SUCCESS ) {
		std::cerr << "ALP/Fuselets double_update encountered error: "
			<< grb::toString( rc ) << "\n";
		return 255;
	} else {
		return 0;
	}
}

