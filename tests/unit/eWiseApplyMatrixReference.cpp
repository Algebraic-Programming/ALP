
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

template< class Iterator >
void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
	return;
#endif
	std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
	if( rows > 50 || cols > 50 ) {
		os << "   Matrix too large to print" << std::endl;
	} else {
		os.precision( 3 );
		for( size_t y = 0; y < rows; y++ ) {
			os << std::string( 3, ' ' );
			for( size_t x = 0; x < cols; x++ ) {
				auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
					return a.first.first == y && a.first.second == x;
				} );
				if( nnz_val != end )
					os << std::fixed << std::setw( 3 ) << ( *nnz_val ).second;
				else
					os << "___";
				os << " ";
			}
			os << std::endl;
		}
	}
	os << "]" << std::endl;
	std::flush( os );
}

template< typename D >
void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	grb::wait( mat );
	printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

template< class Storage, typename D >
void printCompressedStorage( const Storage& storage, const grb::Matrix< D > & mat, std::ostream & os = std::cout ) {
	os << "  row_index: [ ";
	for( size_t i = 0; i < grb::nrows( mat ); ++i ) {
		os << storage.row_index[ i ] << " ";
	}
	os << "]" << std::endl;
	os << "  col_start: [ ";
	for( size_t i = 0; i <= grb::nrows( mat ); ++i ) {
		os << storage.col_start[ i ] << " ";
	}
	os << "]" << std::endl;
	os << "  values:    [ ";
	for( size_t i = 0; i < grb::nnz( mat ); ++i ) {
		os << storage.values[ i ] << " ";
	}
	os << "]" << std::endl << std::flush;
}

template< typename D >
void printCRS( const grb::Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CRS \"" << label << "\" (" << grb::nrows( mat ) << "x" << grb::ncols( mat ) << "):" << std::endl;
	printCompressedStorage(  grb::internal::getCRS( mat ), mat, os );
}

template< typename D >
void printCCS( const grb::Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CCS \"" << label << "\" (" << grb::nrows( mat ) << "x" << grb::ncols( mat ) << "):" << std::endl;
	printCompressedStorage(  grb::internal::getCCS( mat ), mat, os );
}

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
void checkCRSandCCS( const grb::Matrix< T > & obtained, const grb::Matrix< T > & expected, grb::RC & rc ) {
	printCRS( obtained, "obtained" );
	printCRS( expected, "expected" );

	if( grb::nnz( obtained ) != grb::nnz( expected ) ) {
		std::cerr << "Error: unexpected number of non-zero entries; "
				  << "expected " << grb::nnz( expected ) << ", "
				  << "obtained " << grb::nnz( obtained ) << ".\n";
		rc = grb::FAILED;
	}

	{ // check CRS output
		const auto & crsObtained = grb::internal::getCRS( obtained );
		const auto & crsExpected = grb::internal::getCRS( expected );
		for( size_t i = 0; i < grb::nrows( obtained ); ++i ) {
			for( size_t k = crsObtained.col_start[ i ]; k < crsObtained.col_start[ i + 1 ]; ++k ) {
				if( crsObtained.row_index[ k ] != crsExpected.row_index[ k ] ) {
					std::cerr << "Error: unexpected entry at ( " << i << ", " << crsObtained.row_index[ k ] << " ), "
							  << "expected one at ( " << i << ", " << crsExpected.row_index[ k ] << " ) "
							  << "instead (CRS).\n";
					rc = grb::FAILED;
				}
				if( crsObtained.values[ k ] != crsExpected.values[ k ] ) {
					std::cerr << "Error: unexpected value " << crsObtained.values[ k ] << "; "
							  << "expected " << crsExpected.values[ k ] << " (CRS).\n";
					rc = grb::FAILED;
				}
			}
		}
	}

	printCCS( obtained, "obtained" );
	printCCS( expected, "expected" );

	{ // check CCS output
		const auto & ccsObtained = grb::internal::getCCS( obtained );
		const auto & ccsExpected = grb::internal::getCCS( expected );
		for( size_t j = 0; j < grb::ncols( obtained ); ++j ) {
			for( size_t k = ccsExpected.col_start[ j ]; k < ccsExpected.col_start[ j + 1 ]; ++k ) {
				if( ccsObtained.row_index[ k ] != ccsExpected.row_index[ k ] ) {
					std::cerr << "Error: unexpected entry at "
							  << "( " << ccsObtained.row_index[ k ] << ", " << j << " ), "
							  << "expected one at ( " << ccsExpected.row_index[ k ] << ", " << j << " ) "
							  << "instead (CCS).\n";
					rc = grb::FAILED;
				}
				if( ccsObtained.values[ k ] != ccsExpected.values[ k ] ) {
					std::cerr << "Error: unexpected value " << ccsObtained.values[ k ] << "; "
							  << "expected " << ccsExpected.values[ k ] << " (CCS).\n";
					rc = grb::FAILED;
				}
			}
		}
	}
}

void grbProgram( const void *, const size_t, grb::RC & rc ) {

	// initialize test
	grb::Monoid< grb::operators::mul< int >, grb::identities::one > mulmono;

	const size_t n = 4;
	const size_t nelts_A = 8;
	const size_t nelts_B = 6;

	grb::Matrix< int > A( n, n );
	grb::Matrix< int > B( n, n );
	grb::Matrix< void > A_pattern( n, n );
	grb::Matrix< void > B_pattern( n, n );
	grb::Matrix< int > C( n, n );

	rc = grb::resize( A, nelts_A );
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( A, I_A.data(), J_A.data(), V_A.data(), nelts_A, grb::SEQUENTIAL );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( B, nelts_B );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( B, I_B.data(), J_B.data(), V_B.data(), nelts_B, grb::SEQUENTIAL );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( A_pattern, nelts_A );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( A_pattern, I_A.data(), J_A.data(), nelts_A, grb::SEQUENTIAL );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::resize( B_pattern, nelts_B );
	}
	if( rc == grb::SUCCESS ) {
		rc = grb::buildMatrixUnique( B_pattern, I_B.data(), J_B.data(), nelts_B, grb::SEQUENTIAL );
	}
	if( rc != grb::SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	printSparseMatrix( A, "A" );
	printCRS( A, "A" );
	printCCS( A, "A" );
	printSparseMatrix( B, "B" );
	printCRS( B, "B" );
	printCCS( B, "B" );

	{ // test 1: compute with the monoid mxm_elementwise
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A and B value matrices\n";
		grb::clear( C );
		rc = grb::eWiseApply( C, A, B, mulmono, grb::RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono );
		printSparseMatrix( C, "eWiseApply( C, A, B, mulmono )" );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		grb::Matrix< int > union_A_B( n, n );
		grb::buildMatrixUnique( union_A_B, I_C_union.data(), J_C_union.data(), V_C_union_A_B.data(), I_C_union.size(), grb::SEQUENTIAL );
		checkCRSandCCS( C, union_A_B, rc );

		if( rc != grb::SUCCESS ) {
			return;
		}
	}

	{ // test 2: compute with the monoid mxm_elementwise, A value matrix, B pattern matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A value matrix, B pattern matrix\n";
		grb::clear( C );
		rc = grb::eWiseApply( C, A, B_pattern, mulmono, grb::RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B_pattern, mulmono );
		printSparseMatrix( C, "eWiseApply( C, A, B_pattern, mulmono )" );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		grb::Matrix< int > union_A_B_pattern( n, n );
		grb::buildMatrixUnique( union_A_B_pattern, I_C_union_A_B_pattern.data(), J_C_union_A_B_pattern.data(), V_C_union_A_B_pattern.data(), I_C_union_A_B_pattern.size(), grb::SEQUENTIAL );
		checkCRSandCCS( C, union_A_B_pattern, rc );

		if( rc != grb::SUCCESS ) {
			return;
		}
	}

	{ // test 3: compute with the monoid mxm_elementwise, A pattern matrix, B value matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A pattern matrix, B value matrix\n";
		grb::clear( C );
		rc = grb::eWiseApply( C, A_pattern, B, mulmono, grb::RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A_pattern, B, mulmono );
		printSparseMatrix( C, "eWiseApply( C, A_pattern, B, mulmono )" );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		grb::Matrix< int > union_A_pattern_B( n, n );
		grb::buildMatrixUnique( union_A_pattern_B, I_C_union_A_pattern_B.data(), J_C_union_A_pattern_B.data(), V_C_union_A_pattern_B.data(), I_C_union_A_pattern_B.size(), grb::SEQUENTIAL );
		checkCRSandCCS( C, union_A_pattern_B, rc );

		if( rc != grb::SUCCESS ) {
			return;
		}
	}

	{ // test 4: compute with the monoid mxm_elementwise, A pattern matrix, B pattern matrix \n";
		std::cout << "\t Verifying the monoid version of mxm_elementwise, "
				  << "A pattern matrix, B pattern matrix\n";
		grb::clear( C );
		rc = grb::eWiseApply( C, A_pattern, B_pattern, mulmono, grb::RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A_pattern, B_pattern, mulmono );
		printSparseMatrix( C, "eWiseApply( C, A_pattern, B_pattern, mulmono )" );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		grb::Matrix< int > union_A_pattern_B_pattern( n, n );
		grb::buildMatrixUnique( union_A_pattern_B_pattern, I_C_union_A_pattern_B_pattern.data(), J_C_union_A_pattern_B_pattern.data(), V_C_union_A_pattern_B_pattern.data(),
			I_C_union_A_pattern_B_pattern.size(), grb::SEQUENTIAL );
		checkCRSandCCS( C, union_A_pattern_B_pattern, rc );

		if( rc != grb::SUCCESS ) {
			return;
		}
	}

	{ // test 5: compute with the operator mxm_elementwise (pattern matrices not allowed) \n";
		std::cout << "\t Verifying the operator version of mxm_elementwise "
				  << "(only value matrices)\n";
		grb::clear( C );
		rc = grb::eWiseApply( C, A, B, mulmono.getOperator(), grb::RESIZE );
		rc = rc ? rc : grb::eWiseApply( C, A, B, mulmono.getOperator() );
		printSparseMatrix( C, "eWiseApply( C, A, B, mulmono.getOperator() )" );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Call to grb::eWiseApply FAILED\n";
			return;
		}
		grb::Matrix< int > intersection_A_B( n, n );
		grb::buildMatrixUnique( intersection_A_B, I_C_intersection.data(), J_C_intersection.data(), V_C_intersection.data(), I_C_intersection.size(), grb::SEQUENTIAL );
		checkCRSandCCS( C, intersection_A_B, rc );
		if( rc != grb::SUCCESS ) {
			return;
		}
	}
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
