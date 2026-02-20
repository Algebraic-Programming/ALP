
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

#include <sstream>
#include <iostream>

#include <graphblas.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7, 6 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 7, 5, 1 };

/*
 * The above two arrays describe the following sparse structure:
 *
 * x
 *   x
 *     x
 *       x
 *         x
 *           x
 *   x         x
 *           x   x
 *               x x
 * x         x       x
 *
 */

/** Generic implementation of masked tests */
template<
	Descriptor descr = descriptors::no_operation,
	typename Tout, typename Tmask, typename Tin
>
RC masked_tests_generic_impl(
	RC &rc, grb::Matrix< Tout > &output,
	const grb::Matrix< Tmask > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	const bool emptyMask = grb::nrows( mask ) == 0 || grb::ncols( mask ) == 0;

	std::cout << "\t\t with structural descriptor\n";
	rc = grb::set< descr | descriptors::structural >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set structural (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( grb::nnz( output ) != 2 * n - 1 ) {
		std::cerr << "\t unexpected number of output elements ( "
			<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
		rc = FAILED;
	}
	for( const auto &triplet : output ) {
		if(
			triplet.first.first != triplet.first.second &&
			triplet.first.first != triplet.first.second - 1
		) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		}
		if( triplet.first.first != static_cast< size_t >(triplet.second) ) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return rc; }

	std::cout << "\t\t without descriptor\n";
	rc = grb::set< descr >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( emptyMask ) {
		if( grb::nnz( output ) != 2 * n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
			rc = FAILED;
		}
	} else {
		if( grb::nnz( output ) != n ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << n <<".\n";
			rc = FAILED;
		}
	}
	for( const auto &triplet : output ) {
		if( emptyMask ) {
			if( triplet.first.first != triplet.first.second &&
				triplet.first.first != triplet.first.second - 1
			) {
				std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
					<< triplet.first.second << " ), value " << triplet.second << ".\n";
				rc = FAILED;
			}
		} else {
			if( triplet.first.first != triplet.first.second ) {
				std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
					<< triplet.first.second << " ), value " << triplet.second << ".\n";
				rc = FAILED;
			}
		}
		if( triplet.first.first != static_cast< size_t >(triplet.second) ) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return rc; }

	// done
	return rc;
}

/** Specialisation for void output */
template<
	Descriptor descr = descriptors::no_operation,
	typename Tmask, typename Tin
>
RC masked_tests_generic_impl(
	RC &rc, grb::Matrix< void > &output,
	const grb::Matrix< Tmask > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	const bool emptyMask = grb::nrows( mask ) == 0 || grb::ncols( mask ) == 0;

	std::cout << "\t\t with structural descriptor\n";
	rc = grb::set< descr | descriptors::structural >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set structural (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( grb::nnz( output ) != 2 * n - 1 ) {
		std::cerr << "\t unexpected number of output elements ( "
			<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
		rc = FAILED;
	}
	for( const auto &triplet : output ) {
		if(
			triplet.first != triplet.second &&
			triplet.first != triplet.second - 1
		) {
			std::cerr << "\t unexpected entry at ( " << triplet.first << ", "
				<< triplet.second << " ), no value (pattern matrix).\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return rc; }

	std::cout << "\t\t without descriptor\n";
	rc = grb::set< descr >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( emptyMask ) {
		if( grb::nnz( output ) != 2 * n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
			rc = FAILED;
		}
	} else {
		if( grb::nnz( output ) != n ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << n <<".\n";
			rc = FAILED;
		}
	}
	for( const auto &triplet : output ) {
		if( emptyMask ) {
			if( triplet.first != triplet.second &&
				triplet.first != triplet.second - 1
			) {
				std::cerr << "\t unexpected entry at ( " << triplet.first << ", "
					<< triplet.second << " ), no value (pattern matrix).\n";
				rc = FAILED;
			}
		} else {
			if( triplet.first != triplet.second ) {
				std::cerr << "\t unexpected entry at ( " << triplet.first << ", "
					<< triplet.second << " ), no value (pattern matrix).\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) { return rc; }

	// done
	return rc;
}

/** Implementation of masked tests for non-void masks (nvm). */
template<
	Descriptor descr = descriptors::no_operation,
	typename Tout, typename Tmask, typename Tin
>
RC masked_tests_nvm_impl(
	RC &rc, grb::Matrix< Tout > &output,
	const grb::Matrix< Tmask > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	const bool emptyMask = grb::nrows( mask ) == 0 || grb::ncols( mask ) == 0;

	std::cout << "\t\t with invert_mask descriptor\n";
	rc = grb::set< descr | descriptors::invert_mask >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set invert mask (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( emptyMask ) {
		if( grb::nnz( output ) != 2 * n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
			rc = FAILED;
		}
	} else {
		if( grb::nnz( output ) != n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << n - 1 <<".\n";
			rc = FAILED;
		}
	}
	for( const auto &triplet : output ) {
		if( emptyMask ) {
			if( triplet.first.first != triplet.first.second &&
				triplet.first.first != triplet.first.second - 1
			) {
				std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
					<< triplet.first.second << " ), value " << triplet.second << ".\n";
				rc = FAILED;
			}
		} else {
			if( triplet.first.first != triplet.first.second - 1 ) {
				std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
					<< triplet.first.second << " ), value " << triplet.second << ".\n";
				rc = FAILED;
			}
		}
		if( triplet.first.first != static_cast< size_t >(triplet.second) ) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return rc; }

	// done
	return rc;
}

/** Specialisation for void output */
template<
	Descriptor descr = descriptors::no_operation,
	typename Tmask, typename Tin
>
RC masked_tests_nvm_impl(
	RC &rc, grb::Matrix< void > &output,
	const grb::Matrix< Tmask > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	const bool emptyMask = grb::nrows( mask ) == 0 || grb::ncols( mask ) == 0;

	std::cout << "\t\t with invert_mask descriptor\n";
	rc = grb::set< descr | descriptors::invert_mask >( output, mask, input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set invert mask (matrix to matrix masked) FAILED\n";
		return rc;
	}
	if( emptyMask ) {
		if( grb::nnz( output ) != 2 * n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << 2 * n - 1 <<".\n";
			rc = FAILED;
		}
	} else {
		if( grb::nnz( output ) != n - 1 ) {
			std::cerr << "\t unexpected number of output elements ( "
				<< grb::nnz( output ) << " ), expected " << n - 1 <<".\n";
			rc = FAILED;
		}
	}
	for( const auto &triplet : output ) {
		if( emptyMask ) {
			if( triplet.first != triplet.second &&
				triplet.first != triplet.second - 1
			) {
				std::cerr << "\t unexpected entry at ( " << triplet.first << ", "
					<< triplet.second << " ), no value (pattern matrix).\n";
				rc = FAILED;
			}
		} else {
			if( triplet.first != triplet.second - 1 ) {
				std::cerr << "\t unexpected entry at ( " << triplet.first << ", "
					<< triplet.second << " ), no value (pattern matrix).\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) { return rc; }

	// done
	return rc;
}

/** Dispatch for generic masked tests */
template< typename Tout, typename Tmask, typename Tin >
RC masked_tests(
	RC &rc, grb::Matrix< Tout > &output,
	const grb::Matrix< Tmask > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	if( masked_tests_generic_impl( rc, output, mask, input, n ) != SUCCESS ) {
		return rc;
	}
	if( masked_tests_nvm_impl( rc, output, mask, input, n ) != SUCCESS ) {
		return rc;
	}
	std::cout << "\t re-running previous tests with force_row_major descriptor\n";
	if( masked_tests_generic_impl< descriptors::force_row_major >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	if( masked_tests_nvm_impl< descriptors::force_row_major >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	std::cout << "\t\t no_casting descriptor SKIPPED\n";
	return rc;
}

/** Specialised dispatch for masked tests with void masks */
template< typename Tout, typename Tin >
RC masked_tests(
	RC &rc, grb::Matrix< Tout > &output,
	const grb::Matrix< void > &mask, const grb::Matrix< Tin > &input,
	const size_t n
) {
	if( masked_tests_generic_impl( rc, output, mask, input, n ) != SUCCESS ) {
		return rc;
	}
	std::cout << "\t re-running previous tests with force_row_major descriptor\n";
	if( masked_tests_generic_impl< descriptors::force_row_major >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	std::cout << "\t\t invert_mask descriptor SKIPPED\n";
	std::cout << "\t\t no_casting descriptor SKIPPED\n";
	return rc;
}

/** Specialised dispatch for masked tests with no-cast domains */
template< typename T >
RC masked_tests(
	RC &rc, grb::Matrix< T > &output,
	const grb::Matrix< bool > &mask, const grb::Matrix< T > &input,
	const size_t n
) {
	if( masked_tests_generic_impl( rc, output, mask, input, n ) != SUCCESS ) {
		return rc;
	}
	if( masked_tests_nvm_impl( rc, output, mask, input, n ) != SUCCESS ) {
		return rc;
	}
	std::cout << "\t re-running previous tests with force_row_major descriptor\n";
	if( masked_tests_generic_impl< descriptors::force_row_major >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	if( masked_tests_nvm_impl< descriptors::force_row_major >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	std::cout << "\t re-running previous tests with no_casting descriptor\n";
	if( masked_tests_generic_impl< descriptors::no_casting >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	if( masked_tests_nvm_impl< descriptors::no_casting >(
		rc, output, mask, input, n ) != SUCCESS
	) {
		return rc;
	}
	std::cout << "\t re-running previous tests with no_casting *and* "
		<< "force_row_major descriptors\n";
	if(
		masked_tests_generic_impl<
			descriptors::no_casting | descriptors::force_row_major
		>(
			rc, output, mask, input, n
		) != SUCCESS
	) {
		return rc;
	}
	if(
		masked_tests_nvm_impl<
			descriptors::no_casting | descriptors::force_row_major
		>(
			rc, output, mask, input, n
		) != SUCCESS
	) {
		return rc;
	}
	return rc;
}

void grb_program( const size_t &n, grb::RC &rc ) {
	// initialize non-masked test containers
	grb::Matrix< double > A( n, n );
	grb::Matrix< double > B( n, n );
	grb::Matrix< void > C( n, n );
	grb::Matrix< void > D( n, n );
	grb::Matrix< unsigned int > E( n, n );

	// initalize masked test containers
	grb::Matrix< int > mask( n, n );
	grb::Matrix< bool > maskBool( n, n );
	grb::Matrix< void > maskVoid( n, n );
	grb::Matrix< double > maskEmpty( 0, 0 );
	grb::Matrix< int > input( n, n );
	grb::Matrix< void > inputVoid( n, n );
	grb::Matrix< float > inputFloat( n, n );
	grb::Matrix< int > output( n, n );
	grb::Matrix< void > outputVoid( n, n );
	
	// initialise non-masked test data
	int chk[ 10 ][ 10 ];
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			chk[ i ][ j ] = 0;
		}
	}
	for( size_t k = 0; k < 15; ++k ) {
		chk[ I[ k ] ][ J[ k ] ] = data1[ k ];
	}

	rc = grb::resize( A, 15 );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( A, I, J, data1, 15, SEQUENTIAL );
		for( const auto &triplet : A ) {
			if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
				std::cerr << "\t unexpected entry at A( " << triplet.first.first << ", "
					<< triplet.first.second << " ).\n";
				rc = FAILED;
			} else if( chk[ triplet.first.first ][ triplet.first.second ] != triplet.second ) {
				std::cerr << "\t unexpected entry at A( " << triplet.first.first << ", "
					<< triplet.first.second << " ) with value " << triplet.second;
				if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
					std::cerr << ", expected no entry here.\n";
				} else {
					std::cerr << ", expected value "
						<< chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
				}
				rc = FAILED;
			}
		}
	}
	rc = rc ? rc : grb::resize( B, 15 );
	rc = rc ? rc : grb::resize( C, 15 );
	rc = rc ? rc : grb::resize( D, 15 );
	rc = rc ? rc : grb::resize( E, 15 );
	rc = rc ? rc : grb::resize( output, 15 );
	if( rc != SUCCESS || grb::nnz( A ) != 15 ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// initialise data for masked-set tests
	//  - mask will be an n by n matrix with its diagonal set to 1 and its
	//    superdiagonal set to 0.
	//  - input will be an n by n matrix with each element on its diagonal and
	//    superdiagonal set to its row index (meaning the entries on row 0 have
	//    value 0).
	size_t * const I_mask = new size_t[ 2 * n - 1 ];
	size_t * const J_mask = new size_t[ 2 * n - 1 ];
	int * const mask_vals = new int[ 2 * n - 1 ];
	int * const input_vals = new int[ 2 * n - 1 ];
	if( I_mask == nullptr || J_mask == nullptr || mask_vals == nullptr ||
		input_vals == nullptr
	) {
		std::cerr << "\t initialisation FAILED\n";
		return;
	}
	for( size_t k = 0; k < n; ++k ) {
		I_mask[ k ] = J_mask[ k ] = k;
		mask_vals[ k ] = 1;
		input_vals[ k ] = static_cast< int >( k );
		if( k < n - 1 ) {
			I_mask[ n + k ] = k;
			J_mask[ n + k ] = k + 1;
			mask_vals [ n + k ] = 0;
			input_vals[ n + k ] = k;
		}
	}
	rc = grb::buildMatrixUnique( mask, I_mask, J_mask, mask_vals, 2 * n - 1,
		SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of mask matrix FAILED\n";
		return;
	}
	rc = grb::buildMatrixUnique( maskBool, I_mask, J_mask, mask_vals, 2 * n - 1,
		SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of maskBool matrix FAILED\n";
		return;
	}
	try {
		maskVoid = grb::algorithms::matrices< void >::identity( n );
	} catch( ... ) {
		std::cerr << "\t constructing maskVoid FAILED\n";
		rc = FAILED;
		return;
	}
	rc = grb::buildMatrixUnique( input, I_mask, J_mask, input_vals, 2 * n - 1,
		SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of input matrix FAILED\n";
		return;
	}
	rc = grb::buildMatrixUnique( inputFloat, I_mask, J_mask, input_vals, 2 * n - 1,
		SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of inputFloat matrix FAILED\n";
		return;
	}
	rc = grb::resize( inputVoid, 2 * n - 1 );
	rc = rc ? rc : grb::resize( outputVoid, 2 * n - 1 );
	if( rc != SUCCESS ) {
		std::cerr << "\t error resizing matrices for masked tests\n";
		return;
	}
	// postpone materialisation of inputVoid since it relies on unmasked grb::set
	// (which is itself unit-tested later)

	delete [] I_mask;
	delete [] J_mask;
	delete [] mask_vals;
	delete [] input_vals;
	std::cout << "\t test initialisation complete\n";

	// check grb::set for non-voids
	std::cout << "\t testing set( matrix, matrix ), non-void, no-cast\n";
	rc = grb::set( B, A );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set FAILED\n";
		return;
	}
	if( grb::nnz( B ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in B ( "
			<< grb::nnz( B ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto &triplet : B ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\t unexpected entry at B( " << triplet.first.first << ", "
				<< triplet.first.second << " ).\n";
			rc = FAILED;
		} else if( chk[ triplet.first.first ][ triplet.first.second ] != triplet.second ) {
			std::cerr << "\t unexpected entry at B( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value "
					<< chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check grb::set for non-void to void
	std::cout << "\t testing set( matrix, matrix ), non-void to void\n";
	rc = grb::set( C, B );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (non-void to void) FAILED\n";
		return;
	}
	if( grb::nnz( C ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in C ( "
			<< grb::nnz( C ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto &pair : C ) {
		if( pair.first >= 10 ||
			pair.second >= 10 ||
			chk[ pair.first ][ pair.second ] == 0
		) {
			std::cerr << "\t unexpected entry at C( " << pair.first << ", "
				<< pair.second << " ).\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check grb::set for void-to-void
	std::cout << "\t testing set( matrix, matrix ), void to void\n";
	rc = grb::set( D, C );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (void to void) FAILED\n";
		return;
	}
	if( grb::nnz( D ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in D ( "
			<< grb::nnz( D ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto &pair : D ) {
		if( pair.first >= 10 ||
			pair.second >= 10 ||
			chk[ pair.first ][ pair.second ] == 0
		) {
			std::cerr << "\t unexpected entry at D( " << pair.first << ", "
				<< pair.second << " ).\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check casting grb::set
	std::cout << "\t testing set( matrix, matrix ), non-void, casting\n";
	rc = grb::set( E, A );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (cast from double to int) FAILED\n";
		return;
	}
	if( grb::nnz( E ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in E ( "
			<< grb::nnz( E ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto &triplet : E ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\t unexpected entry at E( " << triplet.first.first << ", "
				<< triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} else if( static_cast< unsigned int >(
				chk[ triplet.first.first ][ triplet.first.second ]
			) != triplet.second
		) {
			std::cerr << "\t unexpected entry at E( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value "
					<< chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	std::cout << "\t testing set( matrix, mask, value ), non-void, casting\n";
	rc = grb::clear( E );
	rc = rc ? rc : grb::set( E, A, 117.175 );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (masked set-to-value-while-casting) FAILED\n";
		return;
	}
	if( grb::nnz( E ) != 15 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( E )
			<< " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto &triplet : E ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} else if( 117 != triplet.second ) {
			std::cerr << "\t unexpected entry at ( " << triplet.first.first << ", "
				<< triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value 117.\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) { return; }

	// check masked matrix set
	// first, finish initialisation
	rc = grb::set( inputVoid, input );
	rc = rc ? rc : grb::resize( output, 2 * n - 1 );
	if( rc != SUCCESS || grb::nnz( inputVoid ) != 2 * n - 1 ) {
		std::cerr << "\t error in inputVoid (an earlier test likely failed)\n";
		if( rc == SUCCESS ) { rc = FAILED; }
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, no-cast, "
		<< "empty mask\n";
	if( masked_tests( rc, output, maskEmpty, input, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, no-cast, "
		<< "non-empty mask\n";
	if( masked_tests( rc, output, mask, input, n ) != grb::SUCCESS ) { return; }

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, no-cast, "
		<< "non-empty Boolean mask\n";
	if( masked_tests( rc, output, maskBool, input, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, casting from "
		<< "float to int, empty mask\n";
	if( masked_tests( rc, output, maskEmpty, inputFloat, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, casting from "
		<< "float to int, non-empty mask\n";
	if( masked_tests( rc, output, mask, inputFloat, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), non-void, casting from "
		<< "float to int, non-empty Boolean mask\n";
	if( masked_tests( rc, output, maskBool, inputFloat, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), void-to-void (no cast), "
		<< "empty mask\n";
	if( masked_tests( rc, outputVoid, maskEmpty, inputVoid, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), void-to-void (no cast), "
		<< "non-empty mask\n";
	if( masked_tests( rc, outputVoid, mask, inputVoid, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), void-to-void (no cast), "
		<< "non-empty Boolean mask\n";
	if( masked_tests( rc, outputVoid, maskBool, inputVoid, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), int-to-void, casting "
		<< "(sort of), empty mask\n";
	if( masked_tests( rc, outputVoid, maskEmpty, input, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), int-to-void, casting "
		<< "(sort of), non-empty mask\n";
	if( masked_tests( rc, outputVoid, mask, input, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), int-to-void, casting "
		<< "(sort of), Boolean mask\n";
	if( masked_tests( rc, outputVoid, maskBool, input, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), float-to-void, casting "
		<< "(sort of), empty mask\n";
	if(
		masked_tests( rc, outputVoid, maskEmpty, inputFloat, n ) !=
			grb::SUCCESS
	) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), float-to-void, casting "
		<< "(sort of), non-empty mask\n";
	if( masked_tests( rc, outputVoid, mask, inputFloat, n ) != grb::SUCCESS ) {
		return;
	}

	std::cout << "\t testing set( matrix, mask, matrix ), float-to-void, casting "
		<< "(sort of), Boolean mask\n";
	if( masked_tests( rc, outputVoid, maskBool, inputFloat, n ) != grb::SUCCESS ) {
		return;
	}

	// done
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( !( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}

