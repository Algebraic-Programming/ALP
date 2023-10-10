
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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

#define _DEBUG

namespace {
	template< class Iterator >
	void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
		return;
#endif
		std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
		if( rows > 1000 || cols > 1000 ) {
			os << "   Matrix too large to print" << std::endl;
		} else {
			// os.precision( 3 );
			for( size_t y = 0; y < rows; y++ ) {
				os << std::string( 3, ' ' );
				for( size_t x = 0; x < cols; x++ ) {
					auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
						return a.first.first == y && a.first.second == x;
					} );
					if( nnz_val != end )
						os << std::fixed << ( *nnz_val ).second;
					else
						os << '_';
					os << " ";
				}
				os << std::endl;
			}
		}
		os << "]" << std::endl;
	}

	template< typename D >
	void printSparseMatrix( const Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
		wait( mat );
		printSparseMatrixIterator( nrows( mat ), ncols( mat ), mat.cbegin(), mat.cend(), name, os );
	}

} // namespace

template< typename D >
bool tril_predicate( size_t r, size_t c, D val ) {
	(void)val;
	return r >= c;
}

template< typename D, typename Func >
bool matrix_validate_predicate( const Matrix< D > & B, Func predicate ) {
	return std::all_of( B.cbegin(), B.cend(), [ predicate ]( const std::pair< std::pair< size_t, size_t >, D > & e ) {
		return predicate( e.first.first, e.first.second, e.second );
	} );
}

template< typename D, Backend implementation, typename RIT, typename CIT, typename NIT >
void grb_program( const Matrix< D, implementation, RIT, CIT, NIT > & A, RC & rc ) {
	{ // Test 01: Lower triangular matrix select, same matrix types, boolean predicate
		std::cout << "Test 01: Lower triangular matrix select, same matrix types, boolean predicate" << std::endl;
		Matrix< D > B( nrows( A ), ncols( A ) );
		std::cout << "B.initial: nnz=" << nnz( B ) << ", capacity=" << capacity( B ) << std::endl;
		rc = rc ? rc : select( B, A, operators::is_diagonal<D, RIT, CIT>(), Phase::RESIZE );
		std::cout << "B.resized: nnz=" << nnz( B ) << ", capacity=" << capacity( B ) << std::endl;
		rc = rc ? rc : select( B, A, operators::is_diagonal<D, RIT, CIT>(), Phase::EXECUTE );
		std::cout << "B.executed: nnz=" << nnz( B ) << ", capacity=" << capacity( B ) << std::endl;
		printSparseMatrix( B, "tril" );
		matrix_validate_predicate( B, tril_predicate< D > );
	}
}

int main( int argc, char ** argv ) {
	(void) argc;
	(void) argv;

	RC out = SUCCESS;

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	Launcher< EXEC_MODE::AUTOMATIC > launcher;

	{ // Transposed identity matrix
		Matrix< int > A0( 5, 5  );
		std::vector< size_t > A0_rows( nrows( A0 ), 0 ), A0_cols( ncols( A0 ), 0 );
		std::vector< int > A0_vals( A0_rows.size(), 1 );
		for( size_t i = 0; i < A0_rows.size(); i++ ) {
			A0_rows[ i ] = nrows( A0 ) - 1 - i;
			A0_cols[ i ] = i;
		}
		buildMatrixUnique( A0, A0_rows.data(), A0_cols.data(), A0_vals.data(), A0_rows.size(), IOMode::PARALLEL );

		printSparseMatrix( A0, "A0" );
		if( launcher.exec( &grb_program, A0, out, true ) != RC::SUCCESS ) {
			std::cerr << "Launching test FAILED\n";
			return 255;
		}
	}

	if( out != RC::SUCCESS ) {
		std::cout << "Test FAILED (" << toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
