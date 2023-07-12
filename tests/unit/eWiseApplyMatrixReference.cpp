
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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;


// static data corresponding to small matrices

/**
 * A:
 * 1 _ 3 _
 * _ 4 2 _
 * _ _ 6 7
 * 5 _ _ 8
 */
static const std::vector< size_t > I_A { 0, 0, 1, 1, 2, 2, 3, 3 };
static const std::vector< size_t > J_A { 0, 2, 1, 2, 2, 3, 0, 2 };
static const std::vector< int > V_A { 1, 3, 4, 2, 6, 7, 5, 8 };

/**
 * B:
 *  9 __ __ __
 * __ 11 12 __
 * __ 14 __ __
 * __ __ __ 13
 */
static const std::vector< size_t > I_B { 0, 0, 1, 2, 3, 3 };
static const std::vector< size_t > J_B { 0, 3, 1, 1, 2, 3 };
static const std::vector< int > V_B { 9, 10, 11, 12, 14, 13 };

/**
 * C_intersection:
 *   9 ___ ___ ___
 * ___  44 ___ ___
 * ___ ___ ___ ___
 * ___ ___ 112 ___
 */
static const std::vector< size_t > I_C_intersection { 0, 1, 3 };
static const std::vector< size_t > J_C_intersection { 0, 1, 2 };
static const std::vector< int > V_C_intersection { 9, 44, 112 };

/**
 * C_union_A_B:
 *   9 ___   3  10
 * ___  44   2 ___
 * ___  12   6   7
 *   5 ___ 112  13
 */

static const std::vector< size_t > I_C_union { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3 };
static const std::vector< size_t > J_C_union { 0, 2, 3, 1, 2, 1, 2, 3, 0, 2, 3 };
static const std::vector< int > V_C_union_A_B { 9, 3, 10, 44, 2, 12, 6, 7, 5, 112, 13 };

/**
 * C_union_A_B_pattern:
 * 1 _ 3 1
 * _ 4 2 _
 * _ 1 6 7
 * 5 _ 8 1
 */
static const std::vector< size_t > I_C_union_A_B_pattern { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3 };
static const std::vector< size_t > J_C_union_A_B_pattern { 0, 2, 3, 1, 2, 1, 2, 3, 0, 2, 3 };
static const std::vector< int > V_C_union_A_B_pattern { 1, 3, 1, 4, 2, 1, 6, 7, 5, 8, 1 };

/**
 * C_union_A_pattern_B:
 *  9 __  1 10
 * __ 11  1 __
 * __ 12  1  1
 *  1 __ 14 13
 */
static const std::vector< size_t > I_C_union_A_pattern_B { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3 };
static const std::vector< size_t > J_C_union_A_pattern_B { 0, 2, 3, 1, 2, 1, 2, 3, 0, 2, 3 };
static const std::vector< int > V_C_union_A_pattern_B { 9, 1, 10, 11, 1, 12, 1, 1, 1, 14, 13 };

/**
 * C_union_A_pattern_B_pattern:
 *  1 _ 1 1
 * _ 1 1 _
 * _ 1 1 1
 * 1 _ 1 1
 */
static const std::vector< size_t > I_C_union_A_pattern_B_pattern { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3 };
static const std::vector< size_t > J_C_union_A_pattern_B_pattern { 0, 2, 3, 1, 2, 1, 2, 3, 0, 2, 3 };
static const std::vector< int > V_C_union_A_pattern_B_pattern { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

// helper function to check internal data structures
// of the reference backend
template< typename T >
void checkCRSandCCS( 
	const Matrix< T > & obtained, 
	const Matrix< T > & expected, 
	RC & rc 
) {
	if( nnz( obtained ) != nnz( expected ) ) {
		std::cerr << "Error: unexpected number of non-zero entries; "
				  << "expected " << nnz( expected ) << ", "
				  << "obtained " << nnz( obtained ) << ".\n";
		rc = FAILED;
	}

	{ // check CRS output
		const auto & crsObtained = internal::getCRS( obtained );
		const auto & crsExpected = internal::getCRS( expected );
		for( size_t i = 0; i < nrows( obtained ); ++i ) {
			for( size_t k = crsObtained.col_start[ i ]; k < crsObtained.col_start[ i + 1 ]; ++k ) {
				if( crsObtained.row_index[ k ] != crsExpected.row_index[ k ] ) {
					std::cerr << "Error: unexpected entry at ( " << i << ", " << crsObtained.row_index[ k ] << " ), "
							  << "expected one at ( " << i << ", " << crsExpected.row_index[ k ] << " ) "
							  << "instead (CRS).\n";
					rc = FAILED;
				}
				if( crsObtained.values[ k ] != crsExpected.values[ k ] ) {
					std::cerr << "Error: unexpected value " << crsObtained.values[ k ] << "; "
							  << "expected " << crsExpected.values[ k ] << " (CRS).\n";
					rc = FAILED;
				}
			}
		}
	}

	{ // check CCS output
		const auto & ccsObtained = internal::getCCS( obtained );
		const auto & ccsExpected = internal::getCCS( expected );
		for( size_t j = 0; j < ncols( obtained ); ++j ) {
			for( size_t k = ccsExpected.col_start[ j ]; k < ccsExpected.col_start[ j + 1 ]; ++k ) {
				if( ccsObtained.row_index[ k ] != ccsExpected.row_index[ k ] ) {
					std::cerr << "Error: unexpected entry at "
							  << "( " << ccsObtained.row_index[ k ] << ", " << j << " ), "
							  << "expected one at ( " << ccsExpected.row_index[ k ] << ", " << j << " ) "
							  << "instead (CCS).\n";
					rc = FAILED;
				}
				if( ccsObtained.values[ k ] != ccsExpected.values[ k ] ) {
					std::cerr << "Error: unexpected value " << ccsObtained.values[ k ] << "; "
							  << "expected " << ccsExpected.values[ k ] << " (CCS).\n";
					rc = FAILED;
				}
			}
		}
	}
}

void grbProgram( const void *, const size_t, RC & rc ) {

	// initialize test
	const grb::Monoid< 	grb::operators::mul< int >, 
						grb::identities::one > mulmono;

	const size_t n = 4;
	const size_t nelts_A = 8;
	const size_t nelts_B = 6;

	Matrix< int > A( n, n );
	Matrix< int > B( n, n );
	Matrix< void > A_pattern( n, n );
	Matrix< void > B_pattern( n, n );
	Matrix< int > C( n, n );

	assert( SUCCESS == resize( A, nelts_A ) );
	assert( SUCCESS ==
		buildMatrixUnique( A, I_A.data(), J_A.data(), V_A.data(), nelts_A, SEQUENTIAL )
	);
	assert( SUCCESS == resize( B, nelts_B ) );
	assert( SUCCESS ==
		buildMatrixUnique( B, I_B.data(), J_B.data(), V_B.data(), nelts_B, SEQUENTIAL )
	);
	assert( SUCCESS == resize( A_pattern, nelts_A ) );
	assert( SUCCESS ==
		buildMatrixUnique( A_pattern, I_A.data(), J_A.data(), nelts_A, SEQUENTIAL )
	);
	assert( SUCCESS == resize( B_pattern, nelts_B ) );
	assert( SUCCESS ==
		buildMatrixUnique( B_pattern, I_B.data(), J_B.data(), nelts_B, SEQUENTIAL )
	);	

	{ // test 1: compute with the monoid mxm_elementwise
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A and B value matrices\n";
		clear( C );
		rc = grb::eWiseApply( C, A, B, mulmono, RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		Matrix< int > union_A_B( n, n );
		assert( SUCCESS ==
			buildMatrixUnique( 
			union_A_B, 
			I_C_union.data(), 
			J_C_union.data(), 
			V_C_union_A_B.data(), 
			I_C_union.size(), 
			SEQUENTIAL )
		);
		checkCRSandCCS( C, union_A_B, rc );

		if( rc != SUCCESS ) {
			return;
		}
	}

	{ // test 2: compute with the monoid mxm_elementwise, A value matrix, B pattern matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A value matrix, B pattern matrix\n";
		clear( C );
		rc = grb::eWiseApply( C, A, B_pattern, mulmono, RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B_pattern, mulmono );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		Matrix< int > union_A_B_pattern( n, n );
		assert( SUCCESS ==
			buildMatrixUnique( 
			union_A_B_pattern, 
			I_C_union_A_B_pattern.data(), 
			J_C_union_A_B_pattern.data(), 
			V_C_union_A_B_pattern.data(), 
			I_C_union_A_B_pattern.size(), 
			SEQUENTIAL )
		);
		checkCRSandCCS( C, union_A_B_pattern, rc );

		if( rc != SUCCESS ) {
			return;
		}
	}

	{ // test 3: compute with the monoid mxm_elementwise, A pattern matrix, B value matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A pattern matrix, B value matrix\n";
		clear( C );
		rc = grb::eWiseApply( C, A_pattern, B, mulmono, RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A_pattern, B, mulmono );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		Matrix< int > union_A_pattern_B( n, n );
		assert( SUCCESS ==
			buildMatrixUnique( 
			union_A_pattern_B, 
			I_C_union_A_pattern_B.data(), 
			J_C_union_A_pattern_B.data(), 
			V_C_union_A_pattern_B.data(), 
			I_C_union_A_pattern_B.size(), 
			SEQUENTIAL )
		);
		checkCRSandCCS( C, union_A_pattern_B, rc );

		if( rc != SUCCESS ) {
			return;
		}
	}

	{ // test 4: compute with the monoid mxm_elementwise, A pattern matrix, B pattern matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A pattern matrix, B pattern matrix\n";
		clear( C );
		rc = grb::eWiseApply( C, A_pattern, B_pattern, mulmono, RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A_pattern, B_pattern, mulmono );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		Matrix< int > union_A_pattern_B_pattern( n, n );
		assert( SUCCESS ==
			buildMatrixUnique( 
				union_A_pattern_B_pattern, 
				I_C_union_A_pattern_B_pattern.data(), 
				J_C_union_A_pattern_B_pattern.data(), 
				V_C_union_A_pattern_B_pattern.data(),
				I_C_union_A_pattern_B_pattern.size(), 
				SEQUENTIAL )
		);
		checkCRSandCCS( C, union_A_pattern_B_pattern, rc );

		if( rc != SUCCESS ) {
			return;
		}
	}

	{ // test 5: compute with the operator mxm_elementwise (pattern matrices not allowed) \n";
		std::cout << "\t Verifying the operator version of mxm_elementwise "
				  << "(only value matrices)\n";
		clear( C );
		rc = grb::eWiseApply( C, A, B, mulmono.getOperator(), RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono.getOperator() );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		Matrix< int > intersection_A_B( n, n );
		assert( SUCCESS ==
			buildMatrixUnique( 
				intersection_A_B, 
				I_C_intersection.data(), 
				J_C_intersection.data(), 
				V_C_intersection.data(), 
				I_C_intersection.size(), 
				SEQUENTIAL )
		);
		checkCRSandCCS( C, intersection_A_B, rc );

		if( rc != SUCCESS ) {
			return;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	RC rc;
	grb::Launcher< grb::AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, rc ) != SUCCESS ) {
		std::cerr << "Test failed to launch\n";
		rc = FAILED;
	}
	if( rc == SUCCESS ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED.\n" << std::endl;
	}

	// done
	return 0;
}
