
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

using namespace grb;

using nz_type = int;

constexpr nz_type A_INITIAL_VALUE = 1;
constexpr nz_type B_INITIAL_VALUE = 3;


template< typename D >
bool equals_matrix(
	const Matrix< D > & A,
	const Matrix< D > & B
) {
	if( nrows( A ) != nrows( B ) || ncols( A ) != ncols( B ) ){
		return false;
	}

	wait( A );
	wait( B );

	std::vector< std::pair< std::pair< size_t, size_t >, D > > A_vec( A.cbegin(), A.cend() );
	std::vector< std::pair< std::pair< size_t, size_t >, D > > B_vec( B.cbegin(), B.cend() );
	return std::is_permutation( A_vec.cbegin(), A_vec.cend(), B_vec.cbegin() );
}

template< class Monoid >
struct input_t {
	const Matrix< nz_type > & A;
	const Matrix< nz_type > & B;
	const Matrix< nz_type > & C_monoid;
	const Matrix< nz_type > & C_operator;
	const Monoid & monoid;

	input_t(
		const Matrix< nz_type > & A = {0,0},
		const Matrix< nz_type > & B = {0,0},
		const Matrix< nz_type > & C_monoid = {0,0},
		const Matrix< nz_type > & C_operator = {0,0},
		const Monoid & monoid = Monoid() 
	) : A( A ), 
		B( B ), 
		C_monoid( C_monoid ),
		C_operator( C_operator ), 
		monoid( monoid ) {}
};

struct output_t {
	RC rc;
};

template< class Monoid >
void grb_program( const input_t< Monoid > & input, output_t & output ) {
	static_assert( is_monoid< Monoid >::value, "Monoid required" );
	const auto &op = input.monoid.getOperator();
	wait( input.A );
	wait( input.B );

	RC &rc = output.rc;

	{ // Operator variant
		std::cout << "-- eWiseApply using Operator, supposed to be"
					<< " annihilating non-zeroes -> INTERSECTION\n";
		Matrix< nz_type > C( nrows( input.A ), ncols( input.A ) );
		rc = eWiseApply( C, input.A, input.B, op, RESIZE );
		wait( C );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		rc = eWiseApply( C, input.A, input.B, op, EXECUTE );
		wait( C );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::EXECUTE\n";
			return;
		}

		if( !equals_matrix( C, input.C_operator ) ) {
			std::cerr << "Error: Wrong result\n";
			rc = FAILED;
			return;
		}

		std::cout << "Result (operator) is correct\n";
	}

	{ // Monoid variant
		std::cout << "-- eWiseApply using Monoid, supposed to consider"
					<< " non-zeroes as the identity -> UNION\n";
		Matrix< nz_type > C( nrows( input.A ), ncols( input.A ) );
		rc = eWiseApply( C, input.A, input.B, input.monoid, RESIZE );
		wait( C );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		rc = eWiseApply( C, input.A, input.B, input.monoid, EXECUTE );
		wait( C );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::EXECUTE\n";
			return;
		}

		if( !equals_matrix( C, input.C_monoid ) ) {
			std::cerr << "Error: Wrong result\n";
			rc = FAILED;
			return;
		}

		std::cout << "Result (monoid) is correct\n";
	}

	rc = SUCCESS;
}

int main( int argc, char ** argv ) {
	(void) argc;
	(void) argv;

	size_t N = 10;

	if( argc > 2 ) {
		std::cout << "Usage: " << argv[ 0 ] << std::endl;
		return 1;
	}
	if( argc == 2 ) {
		N = std::stoul( argv[ 1 ] );
	}

	std::cout << "This is functional test " << argv[ 0 ] << std::endl << std::flush;

	Launcher< AUTOMATIC > launcher;

	// Create input data
	/** Matrix A: Row matrix filled with A_INITIAL_VALUE
	 *  X X X X X
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _ (...)
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _
	 * 	  (...)
	 */
	Matrix< nz_type > A( N, N, N );
	std::vector< size_t > A_rows( N, 0 ), A_cols( N, 0 );
	std::vector< nz_type > A_values( N, A_INITIAL_VALUE );
	std::iota( A_cols.begin(), A_cols.end(), 0 );
	if( SUCCESS !=
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), SEQUENTIAL )
	) { return 2; }

	/** Matrix B: Column matrix filled with B_INITIAL_VALUE
	 *  Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _ (...)
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	  (...)
	 */
	Matrix< nz_type > B( N, N, N );
	std::vector< size_t > B_rows( N, 0 ), B_cols( N, 0 );
	std::vector< nz_type > B_values( N, B_INITIAL_VALUE );
	std::iota( B_rows.begin(), B_rows.end(), 0 );
	if( SUCCESS !=
		buildMatrixUnique( B, B_rows.data(), B_cols.data(), B_values.data(), B_values.size(), SEQUENTIAL )
	) { return 3; }

	{
		/** Matrix C_monoid_truth: Union of A and B
		 * X+Y  X   X   X   X
		 * Y  ___ ___ ___ ___
		 * Y  ___ ___ ___ ___ (...)
		 * Y  ___ ___ ___ ___
		 * Y  ___ ___ ___ ___
		 * 	      (...)
		 */
		Matrix< nz_type > C_monoid_truth( N, N );
		size_t nvalues = nrows( A ) + ncols( B ) - 1;
		std::vector< size_t > C_monoid_truth_rows( nvalues, 0 ), C_monoid_truth_cols( nvalues, 0 );
		std::vector< nz_type > C_monoid_truth_values( nvalues, 0 );
		C_monoid_truth_values[ 0 ] = A_INITIAL_VALUE + B_INITIAL_VALUE;
		std::iota( C_monoid_truth_rows.begin() + nrows( A ), C_monoid_truth_rows.end(), 1 );
		std::iota( C_monoid_truth_cols.begin() + 1, C_monoid_truth_cols.begin() + nrows( A ), 1 );
		std::fill( C_monoid_truth_values.begin() + 1, C_monoid_truth_values.begin() + nrows( A ), A_INITIAL_VALUE );
		std::fill( C_monoid_truth_values.begin() + nrows( A ), C_monoid_truth_values.end(), B_INITIAL_VALUE );
		if( SUCCESS !=
			buildMatrixUnique(
				C_monoid_truth,
				C_monoid_truth_rows.data(),
				C_monoid_truth_cols.data(),
				C_monoid_truth_values.data(),
				C_monoid_truth_values.size(),
				SEQUENTIAL
			)
		) { return 4; }

		/** Matrix C_op_truth: Intersection of A and B
		 *  X+Y ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___ (...)
		 * 	___ ___ ___ ___ ___
		 * 	___ ___ ___ ___ ___
		 * 	       (...)
		 */
		Matrix< nz_type > C_op_truth( N, N );
		std::vector< size_t > C_op_truth_rows( 1, 0 ), C_op_truth_cols( 1, 0 );
		std::vector< nz_type > C_op_truth_values( 1, A_INITIAL_VALUE + B_INITIAL_VALUE );
		if( SUCCESS !=
			buildMatrixUnique(
				C_op_truth,
				C_op_truth_rows.data(),
				C_op_truth_cols.data(),
				C_op_truth_values.data(),
				C_op_truth_values.size(),
				SEQUENTIAL
			)
		) { return 5; }

		{ /** Test using addition operator, same type for lhs and rhs
		   */
			input_t<
				Monoid< operators::add< nz_type >, identities::zero >
			> input { A, B, C_monoid_truth, C_op_truth };
			output_t output { SUCCESS };
			// Run the test
			RC rc = launcher.exec( &grb_program, input, output, false );
			// Check the result
			if( rc != SUCCESS ) {
				std::cerr << "Error: Launcher::exec\n";
				return 6;
			}
			if( output.rc != SUCCESS ) {
				std::cerr << "Test FAILED (" << toString( output.rc ) << ")" << std::endl;
				return 7;
			}
		}
	}

	std::cerr << std::flush;
	std::cout << "Test OK" << std::endl << std::flush;
	
	return 0;
}
