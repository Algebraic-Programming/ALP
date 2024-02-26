/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License" );
 * you may !use this file except in compliance with the License.
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
#include <utils/print_vec_mat.hpp>

using namespace grb;

#define STDERR_WITH_LINE std::cerr << "[Line " << __LINE__ << "]  "

constexpr bool Debug = false;

namespace {

	template< bool enabled, typename D >
	void printSparseMatrix(
		const Matrix<D> &mat,
		const std::string &name = "",
		std::ostream &os = std::cerr
	) {
		if( !enabled ) return;
		wait( mat );
		print_matrix( mat, 256, name, os );
	}

} // namespace

template< typename D >
std::tuple< const size_t, const size_t, D > getMatrixEntry(
	const std::pair< const std::pair<size_t, size_t >, D > &entry,
	const typename std::enable_if<
		!std::is_void< D >::value
	>::type * = nullptr
) {
	return std::make_tuple(
		entry.first.first,
		entry.first.second,
		entry.second
	);
}

template< typename D >
std::tuple< const size_t, const size_t, bool > getMatrixEntry(
	const std::pair< const size_t, const size_t > &entry,
	const typename std::enable_if< std::is_void< D >::value >::type * = nullptr
) {
	return std::make_tuple(
		entry.first,
		entry.second,
		true
	);
}

template<
	typename D,
	typename Func,
	typename RIT, typename CIT,
	Backend implementation
>
bool matrix_validate_predicate(
	const Matrix< D, implementation, RIT, CIT > &B,
	Func predicate
) {
	/*
	NOTE:
	This function will fail for distributed backend because the local iterator of the matrix
	does !reflect the global coordinates, which can lead to false negatives.
	*/
	bool valid = true;
	for( const auto &each : B ) {
		const auto entry = getMatrixEntry<D>(each);
		const bool match = predicate(
			std::get<0>( entry ), std::get<1>( entry ), std::get<2>( entry )
		);
		if( !match ) {
			std::cerr << "  /!\\ Predicate failed for ("
				<< std::get<0>( entry ) << ", " << std::get<1>( entry )
				<< ", " << std::get<2>( entry ) << ")"
				<< std::endl;
			valid = false;
			break;
		}
	}
	if(
		collectives<>::allreduce( valid, operators::logical_and< bool >() )
		!= SUCCESS
	) {
		return false;
	}

	return valid;
}

template<
	typename D,
	typename SelectionOperator,
	typename RIT, typename CIT,
	Backend implementation
>
RC test_case(
	const Matrix< D, implementation, RIT, CIT > &input,
	const SelectionOperator op,
	const std::string &test_name
) {
	std::cout << test_name << std::endl;

	{ // Non-lambda variant
		Matrix< D, implementation, RIT, CIT > output(
			nrows( input ), ncols( input ), 0
		);

		RC rc = select( output, input, op, RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "(non-lambda variant): RESIZE phase of test <"
				<< test_name << "> failed, rc is \""
				<< toString(rc) << "\"" << std::endl;
			return rc;
		}

		rc = select( output, input, op, EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "(non-lambda variant): EXECUTE phase of test <"
				<< test_name << "> failed, rc is \""
				<< toString(rc) << "\"" << std::endl;
			return rc;
		}

		printSparseMatrix< Debug >( output );

		const bool valid = matrix_validate_predicate( output, op );
		if( !valid ) {
			std::cerr << "(non-lambda variant): Test <"
				<< test_name << "> failed, output matrix is invalid"
				<< std::endl;
			return FAILED;
		}
	}

	{ // Lambda variant
		Matrix< D, implementation, RIT, CIT > output(
			nrows(input), ncols(input), 0
		);

		auto lambda = []( const RIT &x, const CIT &y, const D &v ) {
			return SelectionOperator()( x, y, v );
		};

		RC rc = selectLambda( output, input, lambda, RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "(lambda variant): RESIZE phase of test <"
				<< test_name << "> failed, rc is \""
				<< toString(rc) << "\"" << std::endl;
			return rc;
		}

		rc = selectLambda( output, input, lambda, EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "(lambda variant): EXECUTE phase of test <"
				<< test_name << "> failed, rc is \""
				<< toString(rc) << "\"" << std::endl;
			return rc;
		}

		grb::wait( output );
		printSparseMatrix< Debug >( output );

		const bool valid = matrix_validate_predicate( output, op );
		if( !valid ) {
			std::cerr << "(lambda variant): Test <" << test_name
				<< "> failed, output matrix is invalid" << std::endl;
			return FAILED;
		}
	}

	return SUCCESS;
}

template<
	typename D,
	typename RIT, typename CIT, typename NIT,
	Backend implementation
>
RC buildMatrixUniqueWrapper(
	Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const size_t * row_indices,
	const size_t * col_indices,
	const size_t nvals,
	const IOMode io_mode,
	const typename std::enable_if<
		!std::is_void<D>::value
	>::type * const = nullptr
) {
	std::vector< D > values( nvals, 1 );
	return buildMatrixUnique( mat, row_indices,
		col_indices, values.data(), nvals, io_mode );
}

template<
	typename D,
	typename RIT, typename CIT, typename NIT,
	Backend implementation
>
RC buildMatrixUniqueWrapper(
	Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const size_t * row_indices,
	const size_t * col_indices,
	const size_t nvals,
	const IOMode io_mode,
	const typename std::enable_if<
		std::is_void<D>::value
	>::type * const = nullptr
) {
	return buildMatrixUnique( mat, row_indices,
		col_indices, nvals, io_mode );
}


template< typename D >
void grb_program( const size_t &n, RC &rc ) {
	const std::string D_name = std::is_void< D >::value
		? "void"
		: "non-void";
	rc = SUCCESS;

	Matrix< D >
		I( n, n, n ),
		I_transposed( n, n, n ),
		One_row( n, n, n ),
		One_col( n, n, n );

	{ // Build matrices
		std::vector< size_t > const_indices_zero( n, 0 );
		std::vector< size_t > iota_indices( n, 0 );
		std::iota( iota_indices.begin(), iota_indices.end(), 0 );
		std::vector< size_t > reverse_iota_indices( n, 0 );
		for ( auto i = 0; i < n; ++i ) {
			reverse_iota_indices[i] = n - i - 1;
		}

		rc = rc
			? rc
			: buildMatrixUniqueWrapper(
				I, iota_indices.data(),
				iota_indices.data(), n, SEQUENTIAL
			);
		printSparseMatrix< Debug >( I, "identity" );

		rc = rc
			? rc
			: buildMatrixUniqueWrapper(
				I_transposed, iota_indices.data(),
				reverse_iota_indices.data(), n, SEQUENTIAL
			);
		printSparseMatrix< Debug >( I_transposed, "transposed-identity" );

		rc = rc
			? rc
			: buildMatrixUniqueWrapper(
				One_row, const_indices_zero.data(),
				iota_indices.data(), n, SEQUENTIAL
			);
		printSparseMatrix< Debug >( One_row, "one-row" );

		rc = rc
			? rc
			: buildMatrixUniqueWrapper(
				One_col, iota_indices.data(),
				const_indices_zero.data(), n, SEQUENTIAL
			);
		printSparseMatrix< Debug >( One_col, "one-column" );
	}

	// Test 01: Select <diagonal>
	rc = rc ? rc : test_case( I, operators::select::is_diagonal<D>(),
	          "Test 01: Select <is_diagonal<" + D_name + ">> out of <identity>" );
	rc = rc ? rc : test_case( I_transposed, operators::select::is_diagonal<D>(),
	          "Test 01: Select <is_diagonal<" + D_name + ">> out of <transposed-identity>" );
	rc = rc ? rc : test_case( One_row, operators::select::is_diagonal<D>(),
	          "Test 01: Select <is_diagonal<" + D_name + ">> out of <one-row>" );
	rc = rc ? rc : test_case( One_col, operators::select::is_diagonal<D>(),
	          "Test 01: Select <is_diagonal<" + D_name + ">> out of <one-column>" );

	// Test 02: Select <strict-lower>
	rc = rc ? rc : test_case( I, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <is_strictly_lower<" + D_name + ">> out of <identity>" );
	rc = rc ? rc : test_case( I_transposed, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <is_strictly_lower<" + D_name + ">> out of <transposed-identity>" );
	rc = rc ? rc : test_case( One_row, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <is_strictly_lower<" + D_name + ">> out of <one-row>" );
	rc = rc ? rc : test_case( One_col, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <is_strictly_lower<" + D_name + ">> out of <one-column>" );

	// Test 03: Select <strict-upper>
	rc = rc ? rc : test_case( I, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <is_strictly_upper<" + D_name + ">> out of <identity>" );
	rc = rc ? rc : test_case( I_transposed, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <is_strictly_upper<" + D_name + ">> out of <transposed-identity>" );
	rc = rc ? rc : test_case( One_row, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <is_strictly_upper<" + D_name + ">> out of <one-row>" );
	rc = rc ? rc : test_case( One_col, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <is_strictly_upper<" + D_name + ">> out of <one-column>" );

	// Test 04: Select <lower-or-diag>
	rc = rc ? rc : test_case( I, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <is_lower_or_diagonal<" + D_name + ">> out of <identity>" );
	rc = rc ? rc : test_case( I_transposed, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <is_lower_or_diagonal<" + D_name + ">> out of <transposed-identity>" );
	rc = rc ? rc : test_case( One_row, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <is_lower_or_diagonal<" + D_name + ">> out of <one-row>" );
	rc = rc ? rc : test_case( One_col, operators::select::is_lower_or_diagonal<D>(),
			  "Test 04: Select <is_lower_or_diagonal<" + D_name + ">> out of <one-column>" );

	// Test 05: Select <upper-or-diag>
	rc = rc ? rc : test_case( I, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <is_lower_or_diagonal<" + D_name + ">> out of <identity>" );
	rc = rc ? rc : test_case( I_transposed, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <is_lower_or_diagonal<" + D_name + ">> out of <transposed-identity>" );
	rc = rc ? rc : test_case( One_row, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <is_lower_or_diagonal<" + D_name + ">> out of <one-row>" );
	rc = rc ? rc : test_case( One_col, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <is_lower_or_diagonal<" + D_name + ">> out of <one-column>" );

	assert(
		collectives<>::allreduce( rc, operators::any_or< RC >() ) == SUCCESS
	);
}

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	RC out = SUCCESS;

	std::cout << "This is functional test " << argv[0] << "\n";
	Launcher< AUTOMATIC > launcher;

	const size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10 ) : 1000;

	{
		std::cout << "-- -- Running test with using matrix-type: int" << std::endl;
		if (launcher.exec( &grb_program<int>, n, out, true ) != SUCCESS) {
			STDERR_WITH_LINE << "Launching test FAILED\n";
			return 255;
		}
		if (out != SUCCESS) {
			STDERR_WITH_LINE << "Test FAILED (" << toString(out) << ")" << std::endl;
			return out;
		}
	}

	// { // To be implemented
	// 	std::cout << "-- -- Running test with using matrix-type: void" << std::endl;
	// 	if (launcher.exec(&grb_program<void>, n, out, true) != SUCCESS) {
	// 		STDERR_WITH_LINE << "Launching test FAILED\n";
	// 		return 255;
	// 	}
	// 	if (out != SUCCESS) {
	// 		STDERR_WITH_LINE << "Test FAILED (" << toString(out) << ")" << std::endl;
	// 		return out;
	// 	}
	// }

	std::cout << std::flush;
	std::cerr << std::flush << "Test OK" << std::endl;
	return 0;
}
