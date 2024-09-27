
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
	const double * const z, const double * const beta,
	const size_t * const ia, const unsigned int * const ij,
	const double * const iv,
	const double * const q,
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
	assert( q != nullptr );
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

	// get ALP versions of input and output containers
	grb::Vector< double > alp_p =
		grb::internal::template wrapRawVector< double >( n, p );
	grb::Vector< double > alp_u =
		grb::internal::template wrapRawVector< double >( n, u );
	double &alp_alpha = *alpha;

	// then input vectors and matrix
	const grb::Vector< double > alp_z =
		grb::internal::template wrapRawVector< double >( n, z );
	const grb::Vector< double > alp_q = grb::internal::wrapRawVector( n, q );
	const MyMatrixType alp_A = grb::internal::wrapCRSMatrix( iv, ij, ia, n, n );

	// we use fall-through return-code checking
	grb::RC ret = grb::SUCCESS;

	// do first op
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		*beta, alp_p,
		dblTimesMonoid
	);
	ret = ret ? ret : grb::foldr< grb::descriptors::dense >(
		alp_z, alp_p,
		dblPlusMonoid
	);

	// do second op
	ret = ret ? ret : grb::mxv( alp_u, alp_A, alp_p, dblSemiring );

	// do third op
	ret = ret ? ret : grb::dot( alp_alpha, alp_u, alp_q, dblSemiring );

	// done
	if( ret == grb::SUCCESS ) {
		return 0;
	} else {
		std::cerr << "ALP/Fuselets update_spmv_dot encountered error: "
			<< grb::toString( ret ) << "\n";
		return 255;
	}
}

