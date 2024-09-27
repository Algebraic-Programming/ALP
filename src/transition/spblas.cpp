
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
 * Implements the SpBLAS interface using ALP/GraphBLAS.
 *
 * This file was split from the preceding <tt>sparseblas.cpp</tt>, which
 * previously implemented both this SpBLAS interface as well as the SparseBLAS
 * one.
 *
 * @author A. N. Yzelman
 * @date 1/2/2024
 */

#include "spblas.h"
#include "native.hpp"
#include "sparse_vector_impl.hpp"

#include <assert.h>

#include <graphblas.hpp>


/** \internal Namespace for helper functions of the SpBLAS implementation. */
namespace spblas {

	/**
	 * \internal Utility function that converts a #extblas_sparse_vector to a
	 *           sparseblas::SparseVector. This is for vectors of doubles.
	 */
	static native::SparseVector< double > * getDoubleVector(
		EXTBLAS_TYPE( sparse_vector ) x
	) {
		return static_cast< native::SparseVector< double >* >( x );
	}

}

SPBLAS_RET_T SPBLAS_NAME( dcsrgemv )(
	const char * transa,
	const int * m_p,
	const double * a, const int * ia, const int * ja,
	const double * x,
	double * y
) {
	// declare algebraic structures
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	grb::Monoid<
		grb::operators::max< int >, grb::identities::negative_infinity
	> maxMonoid;

	// declare minimum necessary descriptors
	constexpr grb::Descriptor minDescr = grb::descriptors::dense |
		grb::descriptors::force_row_major;

	// determine matrix size
	const int m = *m_p;
	const grb::Vector< int > columnIndices =
		grb::internal::template wrapRawVector< int >( ia[ m ], ja );
	int n = 0;
	grb::RC rc = foldl( n, columnIndices, maxMonoid );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Could not determine matrix column size\n";
		assert( false );
		return;
	}

	// retrieve necessary ALP/GraphBLAS container wrappers
	const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
		grb::internal::wrapCRSMatrix( a, ja, ia, m, n );
	const grb::Vector< double > input = grb::internal::template
		wrapRawVector< double >( n, x );
	grb::Vector< double > output = grb::internal::template
		wrapRawVector< double >( m, y );

	// set output vector to zero
	rc = grb::set< minDescr >( output, ring.template getZero< double >() );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Could not set output vector to zero\n";
		assert( false );
		return;
	}

	// do either y=Ax or y=A^Tx
	if( transa[0] == 'N' ) {
		rc = grb::mxv< minDescr >(
			output, A, input, ring
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "ALP/GraphBLAS returns error during SpMV: "
				<< grb::toString( rc ) << ".\n";
			assert( false );
			return;
		}
	} else {
		// Hermitian is not supported
		assert( transa[0] == 'T' );
		rc = grb::mxv<
			minDescr |
			grb::descriptors::transpose_matrix
		>(
			output, A, input, ring
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "ALP/GraphBLAS returns error during transposed SpMV: "
				<< grb::toString( rc ) << ".\n";
			assert( false );
			return;
		}
	}

	rc = rc ? rc : grb::wait( output );
	if( rc != grb::SUCCESS ) {
		std::cerr << "ALP/GraphBLAS returns error during call to grb::wait: "
			<< grb::toString( rc ) << ".\n";
		assert( false );
		return;
	}
	// done
}

SPBLAS_RET_T SPBLAS_NAME( dcsrmm )(
	const char * const transa,
	const int * m, const int * n, const int * k,
	const double * alpha,
	const char * matdescra, const double * val, const int * indx,
	const int * pntrb, const int * pntre,
	const double * b, const int * ldb,
	const double * beta,
	double * c, const int * ldc
) {
	assert( transa[0] == 'N' || transa[0] == 'T' );
	assert( m != nullptr );
	assert( n != nullptr );
	assert( k != nullptr );
	assert( alpha != nullptr );
	// not sure yet what constraints if any on matdescra
	if( *m > 0 && *k > 0 ) {
		assert( pntrb != nullptr );
		assert( pntre != nullptr );
	}
	// val and indx could potentially be NULL if there are no nonzeroes
	assert( b != nullptr );
	assert( ldb != nullptr );
	assert( beta != nullptr );
	assert( c != nullptr );
	assert( ldc != nullptr );
	(void) transa;
	(void) m; (void) n; (void) k;
	(void) alpha;
	(void) matdescra; (void) val; (void) indx; (void) pntrb; (void) pntre;
	(void) b; (void) ldb;
	(void) beta;
	(void) c; (void) ldc;
	// requires dense ALP and mixed sparse/dense operations
	assert( false );
}

SPBLAS_RET_T EXT_SPBLAS_NAME( dcsrmultsv )(
	const char * trans, const int * request,
	const int * m, const int * n,
	const double * a, const int * ja, const int * ia,
	const EXTBLAS_TYPE( sparse_vector ) x,
	EXTBLAS_TYPE( sparse_vector ) y
) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
		grb::internal::wrapCRSMatrix( a, ja, ia, *m, *n );
	auto input  = spblas::getDoubleVector( x );
	auto output = spblas::getDoubleVector( y );
	if( !(input->finalized) ) {
		throw std::runtime_error( "Uninitialised input vector during SpMSpV\n" );
	}
	if( !(output->finalized) ) {
		throw std::runtime_error( "Uninitialised output vector during SpMSpV\n" );
	}
	if( request[ 0 ] != 0 && request[ 1 ] != 1 ) {
		throw std::runtime_error( "Illegal request during call to dcsrmultsv\n" );
	}
	grb::Phase phase = grb::EXECUTE;
	if( request[ 0 ] == 1 ) {
		phase = grb::RESIZE;
	}
	grb::RC rc;
	if( trans[0] == 'N' ) {
		rc = grb::mxv< grb::descriptors::force_row_major >( *(output->vector), A,
			*(input->vector), ring, phase );
	} else {
		if( trans[1] != 'T' ) {
			throw std::runtime_error( "Illegal trans argument to dcsrmultsv\n" );
		}
		rc = grb::mxv<
			grb::descriptors::force_row_major |
			grb::descriptors::transpose_matrix
		>( *(output->vector), A, *(input->vector), ring, phase );
	}
	if( rc != grb::SUCCESS ) {
		throw std::runtime_error( "ALP/GraphBLAS returns error during call to "
			"SpMSpV: " + grb::toString( rc ) );
	}
}

SPBLAS_RET_T SPBLAS_NAME( dcsrmultcsr )(
	const char * trans, const int * request, const int * sort,
	const int * m_p, const int * n_p, const int * k_p,
	double * a, int * ja, int * ia,
	double * b, int * jb, int * ib,
	double * c, int * jc, int * ic,
	const int * nzmax, int * info
) {
	assert( trans[0] == 'N' );
	assert( sort != nullptr && sort[0] == 7 );
	assert( m_p != nullptr );
	assert( n_p != nullptr );
	assert( k_p != nullptr );
	assert( a != nullptr ); assert( ja != nullptr ); assert( ia != nullptr );
	assert( b != nullptr ); assert( jb != nullptr ); assert( ib != nullptr );
	assert( c != nullptr ); assert( jc != nullptr ); assert( ic != nullptr );
	assert( nzmax != nullptr );
	assert( info != nullptr );

	// declare algebraic structures
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// check support
	if( trans[ 0 ] != 'N' ) {
		std::cerr << "ALP/SparseBLAS, error: illegal trans argument to dcsrmultcsr\n";
		*info = 4;
	}
	if( sort[ 0 ] != 7 ) {
		std::cerr << "ALP/SparseBLAS, error: illegal sort argument to dcsrmultcsr\n";
		*info = 5;
		return;
	}

	// declare minimum necessary descriptors
	constexpr const grb::Descriptor minDescr = grb::descriptors::dense |
		grb::descriptors::force_row_major | grb::descriptors::non_owning_view;

	// determine matrix size
	const int m = *m_p;
	const int n = *n_p;
	const int k = *k_p;

	// retrieve buffers (only when A needs to be output also)
	char * bitmask = nullptr;
	char * stack = nullptr;
	double * valbuf = nullptr;
	if( grb::native::template getSPA< double >(
			bitmask, stack, valbuf, n
		) == false
	) {
		std::cerr << "ALP/SpBLAS, error: could not allocate buffer for computations "
			<< "on an output matrix (dcsrmultcsr)\n";
		*info = 10;
		return;
	}

	// retrieve necessary ALP/GraphBLAS container wrappers
	const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
		grb::internal::wrapCRSMatrix( a, ja, ia, m, k );
	const grb::Matrix< double, grb::config::default_backend, int, int, int > B =
		grb::internal::wrapCRSMatrix( b, jb, ib, k, n );
	grb::Matrix< double, grb::config::default_backend, int, int, int > C =
		grb::internal::wrapCRSMatrix(
			c, jc, ic,
			m, n, *nzmax,
			bitmask, stack, valbuf
		);

	// set output vector to zero
	grb::RC rc = grb::clear( C );
	if( rc != grb::SUCCESS ) {
		std::cerr << "ALP/SparseBLAS, error: Could not clear output matrix\n";
		assert( false );
		*info = 20;
		return;
	}

	// do either C=AB or C=A^TB
	if( trans[0] == 'N' ) {
		if( *request == 1 ) {
			rc = grb::mxm< minDescr >( C, A, B, ring, grb::RESIZE );
		} else {
			assert( *request == 0 || *request == 2 );
			rc = grb::mxm< minDescr >( C, A, B, ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "ALP/SparseBLAS, error during call to SpMSpM: "
				<< grb::toString( rc ) << ".\n";
			assert( false );
			*info = 30;
			return;
		}
	} else {
		// this case is not supported
		assert( false );
	}

	// done
	if( *request == 1 ) {
		*info = -1;
	} else {
		*info = 0;
	}
}

SPBLAS_RET_T EXT_SPBLAS_NAME( free )() {
	grb::native::destroyGlobalBuffer();
	const grb::RC rc = grb::finalize();
	if( rc != grb::SUCCESS ) {
		std::cerr << "Error during call to EXT_SPBLAS_free\n";
		assert( false );
	}
}

