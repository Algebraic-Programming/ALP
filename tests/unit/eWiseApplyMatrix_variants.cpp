
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
#include <utils/print_vec_mat.hpp>

using namespace grb;

using nz_type = int;

constexpr nz_type A_INITIAL_VALUE = 1;
constexpr nz_type B_INITIAL_VALUE = 3;

// #define _DEBUG


template< typename D >
bool equals_matrix(
	const Matrix< D > &A,
	const Matrix< D > &B
) {
	if( nrows( A ) != nrows( B ) ||
		ncols( A ) != ncols( B ) ||
		nnz( A ) != nnz( B )
	) {
		return false;
	}

	wait( A );
	wait( B );

	std::vector<
		std::pair< std::pair< size_t, size_t >, D >
	> A_vec( A.cbegin(), A.cend() );
	std::vector<
		std::pair< std::pair< size_t, size_t >, D >
	> B_vec( B.cbegin(), B.cend() );
	return std::is_permutation( A_vec.cbegin(), A_vec.cend(), B_vec.cbegin() );
}


template<
	class Monoid,
	typename ValueTypeA,
	typename ValueTypeB,
	typename ValueTypeC,
	Descriptor descr = descriptors::no_operation
>
struct input_t {
	const Matrix< ValueTypeA > &A;
	const Matrix< ValueTypeB > &B;
	const Matrix< ValueTypeC > &C_monoid;
	const Matrix< ValueTypeC > &C_operator;
	const Monoid &monoid;

	input_t(
		const Matrix< ValueTypeA > &A = {0,0},
		const Matrix< ValueTypeB > &B = {0,0},
		const Matrix< ValueTypeC > &C_monoid = {0,0},
		const Matrix< ValueTypeC > &C_operator = {0,0},
		const Monoid &monoid = Monoid()
	) : A( A ),
		B( B ),
		C_monoid( C_monoid ),
		C_operator( C_operator ),
		monoid( monoid ) {}
};


struct output_t {
	RC rc;
};

template<
	class Monoid,
	typename ValueTypeA,
	typename ValueTypeB,
	typename ValueTypeC,
	Descriptor descr
>
void grb_program(
	const input_t< Monoid, ValueTypeA, ValueTypeB, ValueTypeC, descr > &input,
	output_t &output
) {
	static_assert( is_monoid< Monoid >::value, "Monoid required" );
	const auto &op = input.monoid.getOperator();

	RC &rc = output.rc;

	{ // Operator variant
		std::cout << "  -- eWiseApply using Operator, supposed to be"
					<< " annihilating non-zeroes -> INTERSECTION\n";
		Matrix< ValueTypeC > C( nrows( input.A ), ncols( input.A ) );

		rc = eWiseApply<descr>( C, input.A, input.B, op, RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		if( capacity( C ) < nnz( input.C_operator ) ) {
			std::cerr << "Error: Capacity should be at least " << nnz( input.C_operator ) << "\n";
			rc = FAILED;
			return;
		}

		rc = eWiseApply<descr>( C, input.A, input.B, op, EXECUTE );
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
		std::cout << "  -- eWiseApply using Monoid, supposed to consider"
					<< " non-zeroes as the identity -> UNION\n";
		Matrix< ValueTypeC > C( nrows( input.A ), ncols( input.A ) );

		rc = eWiseApply<descr>( C, input.A, input.B, input.monoid, RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "Error: Phase::RESIZE\n";
			return;
		}
		if( capacity( C ) < nnz( input.C_operator ) ) {
			std::cerr << "Error: Capacity should be at least " << nnz( input.C_monoid ) << "\n";
			rc = FAILED;
			return;
		}

		rc = eWiseApply<descr>( C, input.A, input.B, input.monoid, EXECUTE );
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

void test_program( const size_t& N, size_t& ) {
	/** Matrix A: Matrix filled with A_INITIAL_VALUE
	 *  X X X X X
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _ (...)
	 * 	_ _ _ _ _
	 * 	_ _ _ _ _
	 * 	  (...)
	 */
	Matrix< nz_type > A( N, N, N );
	Matrix< void > A_void( N, N, N );
	{
		std::vector< size_t > A_rows( N, 0 ), A_cols( N, 0 );
		std::vector< nz_type > A_values( N, A_INITIAL_VALUE );
		std::iota( A_cols.begin(), A_cols.end(), 0 );
		if(
			SUCCESS !=
			buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), SEQUENTIAL )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
			+ ": Test FAILED: buildMatrixUnique" );
		}
		if(
			SUCCESS !=
			buildMatrixUnique( A_void, A_rows.data(), A_cols.data(), A_rows.size(), SEQUENTIAL )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
			+ ": Test FAILED: buildMatrixUnique" );
		}
	}


	/** Matrix B: Column matrix filled with B_INITIAL_VALUE
	 *  Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _ (...)
	 * 	Y _ _ _ _
	 * 	Y _ _ _ _
	 * 	  (...)
	 */
	Matrix< nz_type > B( N, N, N );
	Matrix< void > B_void( N, N, N );
	{
		std::vector< size_t > B_rows( N, 0 ), B_cols( N, 0 );
		std::vector< nz_type > B_values( N, B_INITIAL_VALUE );
		std::iota( B_rows.begin(), B_rows.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( B, B_rows.data(), B_cols.data(), B_values.data(), B_values.size(), SEQUENTIAL)
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED: buildMatrixUnique" );
		}
		if(
			SUCCESS !=
			buildMatrixUnique( B_void, B_rows.data(), B_cols.data(), B_rows.size(), SEQUENTIAL )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
			+ ": Test FAILED: buildMatrixUnique" );
		}
	}

	{ // C = A .+ B
		std::cout << "-- Test C = A .+ B\n";
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
				    SEQUENTIAL)
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED: buildMatrixUnique" );
		}

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
				    SEQUENTIAL)
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED: buildMatrixUnique" );
		}

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				nz_type, nz_type, nz_type
		> input { A, B, C_monoid_truth, C_op_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}

	{ // C = A .+ A
		std::cout << "-- Test C = A .+ A\n";
		/** Matrix C_truth: Union/intersection of A and A
		 * X+X X+X X+X X+X X+X
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___(...)
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___
		 * 	      (...)
		 */
		Matrix< nz_type > C_truth( N, N );
		size_t nvalues = ncols( A );
		std::vector< size_t > C_truth_rows( nvalues, 0 ), C_truth_cols( nvalues, 0 );
		std::vector< nz_type > C_truth_values( nvalues, A_INITIAL_VALUE+A_INITIAL_VALUE );
		std::iota( C_truth_cols.begin(), C_truth_cols.end(), 0 );
		if( SUCCESS !=
		    buildMatrixUnique(
				    C_truth,
				    C_truth_rows.data(),
				    C_truth_cols.data(),
				    C_truth_values.data(),
				    C_truth_values.size(),
				    SEQUENTIAL
		    )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
					+ ": Test FAILED: buildMatrixUnique" );
		}

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				nz_type, nz_type, nz_type
		> input { A, A, C_truth, C_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}

	{ // C = A .+ A(void)
		std::cout << "-- Test C = A .+ A(void)\n";
		/** Matrix C_truth: Union/intersection of A and A
		 * X+0 X+0 X+0 X+0 X+0
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___(...)
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___
		 * 	      (...)
		 */
		const Matrix< nz_type >& C_truth = A;

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				nz_type, void, nz_type
		> input { A, A_void, C_truth, C_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}

	{ // C = A(void) .+ A
		std::cout << "-- Test C = A(void) .+ A\n";
		/** Matrix C_truth: Union/intersection of A and A
		 * 0+X 0+X 0+X 0+X 0+X
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___(...)
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___
		 * 	      (...)
		 */
		const Matrix< nz_type >& C_truth = A;

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				void, nz_type, nz_type
		> input { A_void, A, C_truth, C_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}

	{ // C = A(void) .+ A
		std::cout << "-- Test C = A(void) .+ A(void)\n";
		/** Matrix C_truth: Union/intersection of A and A
		 * 0+0 0+0 0+0 0+0 0+0
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___(...)
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___
		 * 	      (...)
		 */
		Matrix< nz_type > C_truth( N, N );
		size_t nvalues = ncols( A );
		std::vector< size_t > C_truth_rows( nvalues, 0 ), C_truth_cols( nvalues, 0 );
		std::vector< nz_type > C_truth_values( nvalues, 0 );
		std::iota( C_truth_cols.begin(), C_truth_cols.end(), 0 );
		if( SUCCESS !=
		    buildMatrixUnique(
				    C_truth,
				    C_truth_rows.data(),
				    C_truth_cols.data(),
				    C_truth_values.data(),
				    C_truth_values.size(),
				    SEQUENTIAL
		    )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
					+ ": Test FAILED: buildMatrixUnique" );
		}

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				void, void, nz_type
		> input { A_void, A_void, C_truth, C_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}


	{ // C = A .+ Bt
		std::cout << "-- Test C = A .+ Bt\n";
		/** Matrix C_truth: Union/intersection of A and Bt
		 * X+Y X+Y X+Y X+Y X+Y
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___(...)
		 * ___ ___ ___ ___ ___
		 * ___ ___ ___ ___ ___
		 * 	      (...)
		 */
		Matrix< nz_type > C_truth( N, N );
		size_t nvalues = ncols( A );
		std::vector< size_t > C_truth_rows( nvalues, 0 ), C_truth_cols( nvalues, 0 );
		std::vector< nz_type > C_truth_values( nvalues, A_INITIAL_VALUE+B_INITIAL_VALUE );
		std::iota( C_truth_cols.begin(), C_truth_cols.end(), 0 );
		if( SUCCESS !=
		    buildMatrixUnique(
				    C_truth,
				    C_truth_rows.data(),
				    C_truth_cols.data(),
				    C_truth_values.data(),
				    C_truth_values.size(),
				    SEQUENTIAL
		    )
		) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED: buildMatrixUnique" );
		}

		input_t<
				Monoid< operators::add< nz_type >, identities::zero >,
				nz_type, nz_type, nz_type,
				descriptors::transpose_right
		> input { A, B, C_truth, C_truth };
		output_t output { SUCCESS };
		// Run the test
		grb_program(input, output );
		if( output.rc != SUCCESS ) {
			throw std::runtime_error("(LINE " + std::to_string(__LINE__)
				+ ": Test FAILED (" + toString( output.rc ) + ")" );
		}
	}


}

int main( int argc, char ** argv ) {
	(void) argc;
	(void) argv;

	size_t N = 1000;

	if( argc > 2 ) {
		std::cout << "Usage: " << argv[ 0 ] << " [n=" << N << "]" << std::endl;
		return 1;
	}
	if( argc == 2 ) {
		N = std::stoul( argv[ 1 ] );
	}

	std::cout << "This is functional test " << argv[ 0 ] << std::endl << std::flush;

	// Launch the test
	Launcher< AUTOMATIC > launcher;
	RC rc = launcher.exec( &test_program, N, N, true );
	if( rc != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( rc ) << ")" << std::endl;
		return static_cast<int>( rc );
	}

	std::cerr << std::flush;
	std::cout << std::flush << "Test OK" << std::endl;
	return 0;
}
