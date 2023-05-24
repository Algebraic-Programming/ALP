
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

/*
 * @author Benjamin Lozes
 * @date 24th of May, 2023
 * 
 * @brief Test for eWiseApply(Matrix, Monoid) 
 *		  and eWiseApply(Matrix, Operator) variants
 * 
 * This test is meant to ensure the behaviour of the eWiseApply(Matrix, Monoid)
 * and eWiseApply(Matrix, Operator) variants is correct. Precisely, we expect
 * the following behaviour:
 * 		- eWiseApply(Matrix, Monoid) should apply the monoid to all elements of
 * 		  the two matrices, INCLUDING the couples (non_zero, zero), using the
 * 		  provided identity value for the zero elements.
 * 		- eWiseApply(Matrix, Operator) should apply the operator to all elements
 * 		  of the two matrices, EXCLUDING the couples (non_zero, zero)
 * 
 */

#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

#define _DEBUG

using nz_type = int;

constexpr size_t M = 10;
constexpr size_t N = 10;
constexpr nz_type A_INITIAL_VALUE = 1;
constexpr nz_type B_INITIAL_VALUE = 3;

namespace utils {
	template< class Iterator >
	void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
		return;
#endif
		std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
		if( rows > 50 || cols > 50 ) {
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
		std::flush( os );
	}

	template< typename D >
	void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
		grb::wait( mat );
		printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
	}

	template< typename D >
	bool equals_matrix( const grb::Matrix< D > & A, const grb::Matrix< D > & B ) {
		if( grb::nrows( A ) != grb::nrows( B ) || grb::ncols( A ) != grb::ncols( B ) )
			return false;
		grb::wait( A );
		grb::wait( B );
		std::vector< std::pair< std::pair< size_t, size_t >, D > > A_vec( A.cbegin(), A.cend() );
		std::vector< std::pair< std::pair< size_t, size_t >, D > > B_vec( B.cbegin(), B.cend() );
		return std::is_permutation( A_vec.cbegin(), A_vec.cend(), B_vec.cbegin() );
	}
} // namespace utils

template< class Monoid >
struct input_t {
	const grb::Matrix< nz_type > & A;
	const grb::Matrix< nz_type > & B;
	const grb::Matrix< nz_type > & C_monoid;
	const grb::Matrix< nz_type > & C_operator;
	const Monoid & monoid;

	input_t( 
		const grb::Matrix< nz_type > & A = {0,0},
		const grb::Matrix< nz_type > & B = {0,0},
		const grb::Matrix< nz_type > & C_monoid = {0,0},
		const grb::Matrix< nz_type > & C_operator = {0,0},
		const Monoid & monoid = Monoid() ) :
		A( A ), B( B ), C_monoid( C_monoid ), C_operator( C_operator ), monoid( monoid ) {}
};	

struct output_t {
	grb::RC rc;
};

template< class Monoid >
void grb_program( const input_t< Monoid > & input, output_t & output ) {
	static_assert( grb::is_monoid< Monoid >::value, "Monoid required" );
	const auto & op = input.monoid.getOperator();
	grb::wait( input.A );
	grb::wait( input.B );

	auto & rc = output.rc;

	utils::printSparseMatrix( input.A, "A" );
	utils::printSparseMatrix( input.B, "B" );

	{ // Operator variant
		std::cout << "-- eWiseApply using Operator, supposed to be annihilating non-zeroes -> INTERSECTION\n";
		grb::Matrix< nz_type > C( grb::nrows( input.A ), grb::ncols( input.A ) );
		rc = grb::eWiseApply( C, input.A, input.B, op, grb::Phase::RESIZE );
		grb::wait( C );
		if( rc != grb::RC::SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		rc = grb::eWiseApply( C, input.A, input.B, op, grb::Phase::EXECUTE );
		grb::wait( C );
		if( rc != grb::RC::SUCCESS ) {
			std::cerr << "Error: Phase::EXECUTE\n";
			return;
		}

		if( ! utils::equals_matrix( C, input.C_operator ) ) {
			std::cerr << "Error: Wrong result\n";
			utils::printSparseMatrix( C, "Obtained (operator)", std::cerr );
			utils::printSparseMatrix( input.C_operator, "Truth (operator)", std::cerr );
			rc = grb::RC::FAILED;
			return;
		}

		std::cout << "Result (operator) is correct\n";
	}

	{ // Monoid variant
		std::cout << "-- eWiseApply using Monoid, supposed to consider non-zeroes as the identity -> UNION\n";
		grb::Matrix< nz_type > C( grb::nrows( input.A ), grb::ncols( input.A ) );
		rc = grb::eWiseApply( C, input.A, input.B, input.monoid, grb::Phase::RESIZE );
		grb::wait( C );
		if( rc != grb::RC::SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		rc = grb::eWiseApply( C, input.A, input.B, input.monoid, grb::Phase::EXECUTE );
		grb::wait( C );
		if( rc != grb::RC::SUCCESS ) {
			std::cerr << "Error: Phase::EXECUTE\n";
			return;
		}

		if( ! utils::equals_matrix( C, input.C_monoid ) ) {
			std::cerr << "Error: Wrong result\n";
			utils::printSparseMatrix( C, "Obtained (monoid)", std::cerr );
			utils::printSparseMatrix( input.C_monoid, "Truth (monoid)", std::cerr );
			rc = grb::RC::FAILED;
			return;
		}

		std::cout << "Result (monoid) is correct\n";
	}

	rc = grb::RC::SUCCESS;
}

int main( int argc, char ** argv ) {
	(void) argc;
	(void) argv;

	if(argc > 1) std::cout << "Usage: " << argv[ 0 ] << std::endl;

	std::cout << "This is functional test " << argv[ 0 ] << std::endl;
	grb::Launcher< grb::EXEC_MODE::AUTOMATIC > launcher;
	grb::RC rc = grb::RC::SUCCESS;

	// Create input data
	/** Matrix A: Row matrix filled with A_INITIAL_VALUE
	 *  X X X X X
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _ (...)
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _
	 * 	  (...)
	 */
	grb::Matrix< nz_type > A( M, N, N );
	std::vector< size_t > A_rows( N, 0 ), A_cols( N, 0 );
	std::vector< nz_type > A_values( N, A_INITIAL_VALUE );
	std::iota( A_cols.begin(), A_cols.end(), 0 );
	rc = grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), grb::IOMode::SEQUENTIAL );
	assert( rc == grb::RC::SUCCESS );

	/** Matrix B: Column matrix filled with B_INITIAL_VALUE
	 *  Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _ (...)
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	  (...)
	 */
	grb::Matrix< nz_type > B( M, N, N );
	std::vector< size_t > B_rows( M, 0 ), B_cols( M, 0 );
	std::vector< nz_type > B_values( M, B_INITIAL_VALUE );
	std::iota( B_rows.begin(), B_rows.end(), 0 );
	rc = grb::buildMatrixUnique( B, B_rows.data(), B_cols.data(), B_values.data(), B_values.size(), grb::IOMode::SEQUENTIAL );
	assert( rc == grb::RC::SUCCESS );

	{
		/** Matrix C_monoid_truth: Union of A and B
		 * X+Y  X   X   X   X
		 * Y  ___ ___ ___ ___
		 * Y  ___ ___ ___ ___ (...)
		 * Y  ___ ___ ___ ___
		 * Y  ___ ___ ___ ___
		 * 	      (...)
		 */
		grb::Matrix< nz_type > C_monoid_truth( M, N );
		size_t nvalues = grb::nrows( A ) + grb::ncols( B ) - 1;
		std::vector< size_t > C_monoid_truth_rows( nvalues, 0 ), C_monoid_truth_cols( nvalues, 0 );
		std::vector< nz_type > C_monoid_truth_values( nvalues, 0 );
		C_monoid_truth_values[ 0 ] = A_INITIAL_VALUE + B_INITIAL_VALUE;
		std::iota( C_monoid_truth_rows.begin() + grb::nrows( A ), C_monoid_truth_rows.end(), 1 );
		std::iota( C_monoid_truth_cols.begin() + 1, C_monoid_truth_cols.begin() + grb::nrows( A ), 1 );
		std::fill( C_monoid_truth_values.begin() + 1, C_monoid_truth_values.begin() + grb::nrows( A ), A_INITIAL_VALUE );
		std::fill( C_monoid_truth_values.begin() + grb::nrows( A ), C_monoid_truth_values.end(), B_INITIAL_VALUE );
		rc = grb::buildMatrixUnique( C_monoid_truth, C_monoid_truth_rows.data(), C_monoid_truth_cols.data(), C_monoid_truth_values.data(), C_monoid_truth_values.size(), grb::IOMode::SEQUENTIAL );
		assert( rc == grb::RC::SUCCESS );

		/** Matrix C_op_truth: Intersection of A and B
		 *  X+Y ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___ (...)
		 * 	___ ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___
		 * 	       (...)
		 */
		grb::Matrix< nz_type > C_op_truth( M, N );
		std::vector< size_t > C_op_truth_rows( 1, 0 ), C_op_truth_cols( 1, 0 );
		std::vector< nz_type > C_op_truth_values( 1, A_INITIAL_VALUE + B_INITIAL_VALUE );
		rc = grb::buildMatrixUnique( C_op_truth, C_op_truth_rows.data(), C_op_truth_cols.data(), C_op_truth_values.data(), C_op_truth_values.size(), grb::IOMode::SEQUENTIAL );
		assert( rc == grb::RC::SUCCESS );

		{ /** Test using addition operator, same type for lhs and rhs
		   */
			input_t< grb::Monoid< grb::operators::add< nz_type >, grb::identities::zero > > input { A, B, C_monoid_truth, C_op_truth,
				grb::Monoid< grb::operators::add< nz_type >, grb::identities::zero >() };
			output_t output { grb::RC::SUCCESS };
			// Run the test
			rc = launcher.exec( &grb_program, input, output, false );
			// Check the result
			assert( rc == grb::RC::SUCCESS );
			if( output.rc != grb::RC::SUCCESS ) {
				std::cout << "Test FAILED (" << grb::toString( output.rc ) << ")" << std::endl;
				return 1;
			}
		}
	}

	std::cout << "Test OK" << std::endl;
	return 0;
}
