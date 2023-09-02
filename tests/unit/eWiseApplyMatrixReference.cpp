
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

#include <limits>
#include <sstream>
#include <iostream>

#include <graphblas.hpp>

#ifdef _DEBUG
 #include <utils/print_vec_mat.hpp>
#endif


// static data corresponding to small matrices

static const size_t I_A[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
static const size_t J_A[] = { 0, 2, 1, 2, 2, 3, 0, 2 };
static const double V_A[] = { 1, 3, 4, 2, 6, 7, 5, 8 };

static const size_t I_B[] = { 0, 0, 1, 2, 3, 3 };
static const size_t J_B[] = { 0, 3, 1, 1, 2, 3 };
static const double V_B[] = { 9, 10, 11, 12, 14, 13 };

static const size_t I_C[] = { 0, 1, 3 };
static const size_t J_C[] = { 0, 1, 2 };
static const double V_C[] = { 9, 44, 112 };

static const size_t rowlens[] = { 1, 1, 0, 1 };
static const size_t collens[] = { 1, 1, 1, 0 };

static const double expect1_CRS[] = { 9, 44, 112 };
static const double expect1_CCS[] = { 9, 44, 112 };

static const double expect2_CRS[] = { 1, 4, 8 };
static const double expect2_CCS[] = { 1, 4, 8 };

static const double expect3_CRS[] = { 9, 11, 14 };
static const double expect3_CCS[] = { 9, 11, 14 };

static const double expect4_CRS[] = { 1, 1, 1 };
static const double expect4_CCS[] = { 1, 1, 1 };

// helper function to check internal data structures
// of the reference backend
template< typename T >
void checkCRSandCCS( const grb::Matrix< T > &C,
	const size_t n,
	const size_t * rlens,
	const size_t * clens,
	const size_t * I,
	const size_t * J,
	const double * expect_CRS,
	const double * expect_CCS,
	grb::RC &rc
) {
	// check CRS output
	const auto &crs1 = grb::internal::getCRS( C );
	for( size_t i = 0; i < n; ++i ) {
		const size_t entries = crs1.col_start[ i + 1 ] - crs1.col_start[ i ];
		if( entries != rlens[ i ] ) {
			std::cerr << "Error: unexpected number of entries " << entries << ", "
			       << " expected " << rlens[ i ] << " (CRS).\n";
			rc = grb::FAILED;
		}
		for( size_t k = crs1.col_start[ i ]; k < crs1.col_start[ i + 1 ]; ++k ) {
			if( crs1.row_index[ k ] != J[ k ] ) {
				std::cerr << "Error: unexpected entry at ( " << i << ", "
					<< crs1.row_index[ k ] << " ), "
					<< "expected one at ( " << i << ", " << J[ k ] << " ) "
					<< "instead (CRS).\n";
				rc = grb::FAILED;
			}
			if( crs1.values[ k ] != expect_CRS[ k ] ) {
				std::cerr << "Error: unexpected value " << crs1.values[ k ] << "; "
					<< "expected " << expect_CRS[ k ] << " (CRS).\n";
				rc = grb::FAILED;
			}
		}
	}

	// check CCS output
	const auto &ccs1 = grb::internal::getCCS( C );
	for( size_t j = 0; j < n; ++j ) {
		const size_t entries = ccs1.col_start[ j + 1 ] - ccs1.col_start[ j ];
		if( entries != clens[ j ] ) {
			std::cerr << "Error: unexpected number of entries " << entries << ", "
				<< "expected " << clens[ j ] << " (CCS).\n";
			rc = grb::FAILED;
		}
		for( size_t k = ccs1.col_start[ j ]; k < ccs1.col_start[ j + 1 ]; ++k ) {
			if( ccs1.row_index[ k ] != I[ k ] ) {
				std::cerr << "Error: unexpected entry at "
					<< "( " << ccs1.row_index[ k ] << ", " << j << " ), "
					<< "expected one at ( " << I[ k ] << ", " << j << " ) "
					<< "instead (CCS).\n";
				rc = grb::FAILED;
			}
			if( ccs1.values[ k ] != expect_CCS[ k ] ) {
				std::cerr << "Error: unexpected value " << ccs1.values[ k ] << "; "
					<< "expected " << expect_CCS[ k ] << " (CCS).\n";
				rc = grb::FAILED;
			}
		}
	}
}

void grbProgram( const void *, const size_t, grb::RC &rc ) {
#ifdef _DEBUG
	constexpr const size_t smax = std::numeric_limits< size_t >::max();
#endif

	// initialise test
	grb::Monoid< grb::operators::mul< double >, grb::identities::one > mulmono;

	const size_t n = 4;
	const size_t nelts_A = 8;
	const size_t nelts_B = 6;

	grb::Matrix< double > A( n, n );
	grb::Matrix< double > B( n, n );
	grb::Matrix< void > A_pattern( n, n );
	grb::Matrix< void > B_pattern( n, n );
	grb::Matrix< double > C( n, n );

	rc = grb::resize( A, nelts_A );
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( A, I_A, J_A, V_A, nelts_A, grb::SEQUENTIAL );
#ifdef _DEBUG
		print_matrix( A, smax, "A" );
#endif
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( B, nelts_B );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( B, I_B, J_B, V_B, nelts_B, grb::SEQUENTIAL );
#ifdef _DEBUG
		print_matrix( B, smax, "B" );
#endif
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( A_pattern, nelts_A );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( A_pattern, I_A, J_A, nelts_A, grb::SEQUENTIAL );
#ifdef _DEBUG
		print_matrix( A_pattern, smax, "A_pattern" );
#endif
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( B_pattern, nelts_B );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( B_pattern, I_B, J_B, nelts_B, grb::SEQUENTIAL );
#ifdef _DEBUG
		print_matrix( B_pattern, smax, "B_pattern" );
#endif
	}
	if( rc != grb::SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// test 1: compute with the monoid mxm_elementwise
	std::cout << "\t Verifying the monoid version of mxm_elementwise, "
		<< "A and B value matrices\n";
	rc = grb::eWiseApply( C, A, B, mulmono, grb::RESIZE );
	rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::eWiseApply FAILED\n";
		return;
	}

	checkCRSandCCS( C, n, rowlens, collens, I_C, J_C, expect1_CRS, expect1_CCS, rc );

	if( rc != grb::SUCCESS ) {
		return;
	}

	// test 2: compute with the monoid mxm_elementwise, A value matrix, B pattern matrix \n";
	std::cout << "\t Verifying the monoid version of mxm_elementwise, "
		<< "A value matrix, B pattern matrix\n";
	rc = grb::eWiseApply( C, A, B_pattern, mulmono, grb::RESIZE );
	rc = rc ? rc : grb::eWiseApply( C, A, B_pattern, mulmono );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::eWiseApply FAILED\n";
		return;
	}

	checkCRSandCCS( C, n, rowlens, collens, I_C, J_C, expect2_CRS, expect2_CCS, rc );

	if( rc != grb::SUCCESS ) {
		return;
	}

	// test 3: compute with the monoid mxm_elementwise, A pattern matrix, B value matrix \n";
	std::cout << "\t Verifying the monoid version of mxm_elementwise, "
		<< "A pattern matrix, B value matrix\n";
	rc = grb::eWiseApply( C, A_pattern, B, mulmono, grb::RESIZE );
	rc = rc ? rc : grb::eWiseApply( C, A_pattern, B, mulmono );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::eWiseApply FAILED\n";
		return;
	}

	checkCRSandCCS( C, n, rowlens, collens, I_C, J_C, expect3_CRS, expect3_CCS, rc );

	if( rc != grb::SUCCESS ) {
		return;
	}

	// test 4: compute with the monoid mxm_elementwise, A pattern matrix, B pattern matrix \n";
	std::cout << "\t Verifying the monoid version of mxm_elementwise, "
		<< "A pattern matrix, B pattern matrix\n";
	rc = grb::eWiseApply( C, A_pattern, B_pattern, mulmono, grb::RESIZE );
	rc = rc ? rc : grb::eWiseApply( C, A_pattern, B_pattern, mulmono );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::eWiseApply FAILED\n";
		return;
	}

	checkCRSandCCS( C, n, rowlens, collens, I_C, J_C, expect4_CRS, expect4_CCS, rc );

	if( rc != grb::SUCCESS ) {
		return;
	}

	// test 5: compute with the operator mxm_elementwise (pattern matrices not allowed) \n";
	std::cout << "\t Verifying the operator version of mxm_elementwise "
		<< "(only value matrices)\n";
	rc = grb::eWiseApply( C, A, B, mulmono.getOperator(), grb::RESIZE );
	rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono.getOperator() );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::eWiseApply FAILED\n";
		return;
	}

	checkCRSandCCS( C, n, rowlens, collens, I_C, J_C, expect1_CRS, expect1_CCS, rc );
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	grb::RC rc;
	grb::Launcher< grb::AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, rc ) != grb::SUCCESS ) {
		std::cerr << "Test failed to launch\n";
		rc = grb::FAILED;
	}
	if( rc == grb::SUCCESS ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED.\n" << std::endl;
	}

	// done
	return 0;
}

