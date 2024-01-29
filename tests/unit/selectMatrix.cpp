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

#define STDERR_WITH_LINE std::cerr << "[Line " << __LINE__ << "]  "

constexpr bool Debug = false;

namespace {
	template<class Iterator>
	void printSparseMatrixIterator(size_t rows, size_t cols, Iterator begin, Iterator end, const std::string& name = "",
	                               std::ostream& os = std::cout) {
		std::vector<bool> assigned(rows * cols, false);
		for( auto it = begin; it != end; ++it ) {
			assigned[ it->first.first * cols + it->first.second ] = true;
		}

		os << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
		if (rows > 256 || cols > 256) {
			os << "   Matrix too large to print" << std::endl;
		} else {
			// os.precision( 3 );
			for (size_t y = 0; y < rows; y++) {
				os << std::string(3, ' ');
				for (size_t x = 0; x < cols; x++) {
					if (assigned[y * cols + x])
						os << 'X';
					else
						os << '_';
					os << " ";
				}
				os << std::endl;
			}
		}
		os << "]" << std::endl;
	}

	template<bool enabled, typename D>
	void printSparseMatrix(const Matrix<D>& mat, const std::string& name = "", std::ostream& os = std::cout) {
		if (not enabled) return;
		wait(mat);
		printSparseMatrixIterator(nrows(mat), ncols(mat), mat.cbegin(), mat.cend(), name, os);
	}
} // namespace

template<typename D, typename Func, typename RIT, typename CIT, Backend implementation>
bool matrix_validate_predicate(
	const Matrix<D, implementation, RIT, CIT>& B,
	Func predicate
) {
	/*
	NOTE:
	This function will fail for distributed backend because the local iterator of the matrix
	does not reflect the global coordinates, which can lead to false negatives.
	*/
	bool valid = true;
	for( const auto &each : B ) {
		if( not predicate.apply( each.first.first, each.first.second, each.second ) ) {
			std::cerr << "  /!\\ Predicate failed for ("
				<< each.first.first << ", " << each.first.second << ", " << each.second << ")"
				<< std::endl;
			valid = false;
			break;
		}
	}
	assert( collectives<>::allreduce( valid, operators::logical_and<bool>() ) == SUCCESS );

	return valid;
}

template<typename D, typename SelectionOperator, typename RIT, typename CIT, Backend implementation>
RC test_case(
	const Matrix<D, implementation, RIT, CIT>& input,
	const SelectionOperator op,
	const std::string& test_name
) {
	std::cout << test_name << std::endl;

	{ // Non-lambda variant
		Matrix<D, implementation, RIT, CIT> output(nrows(input), ncols(input), 0);

		RC rc = select(output, input, op, RESIZE);
		if( rc != SUCCESS ) {
			std::cerr << "(non-lambda variant): RESIZE phase of test <" << test_name <<
				"> failed, rc is \"" << toString(rc) << "\"" << std::endl;
			return rc;
		}

		rc = select(output, input, op, EXECUTE);
		if( rc != SUCCESS ) {
			std::cerr << "(non-lambda variant): EXECUTE phase of test <" << test_name <<
				"> failed, rc is \"" << toString(rc) << "\"" << std::endl;
			return rc;
		}

		grb::wait( output );
		printSparseMatrix<Debug>(output);


		const bool valid = matrix_validate_predicate(output, op);
		if( not valid ) {
			std::cerr << "(non-lambda variant): Test <" << test_name << "> failed, output matrix is invalid" << std::endl;
			return FAILED;
		}
	}

	{ // Lambda variant
		Matrix<D, implementation, RIT, CIT> output(nrows(input), ncols(input), 0);

		auto lambda = [&op](const RIT & x, const CIT & y, const D & v) {
			return op.apply(x, y, v);
		};

		RC rc = selectLambda(output, input, lambda, RESIZE);
		if( rc != SUCCESS ) {
			std::cerr << "(lambda variant): RESIZE phase of test <" << test_name <<
				"> failed, rc is \"" << toString(rc) << "\"" << std::endl;
			return rc;
		}

		rc = selectLambda(output, input, lambda, EXECUTE);
		if( rc != SUCCESS ) {
			std::cerr << "(lambda variant): EXECUTE phase of test <" << test_name <<
				"> failed, rc is \"" << toString(rc) << "\"" << std::endl;
			return rc;
		}

		grb::wait( output );
		printSparseMatrix<Debug>(output);

		const bool valid = matrix_validate_predicate(output, op);
		if( not valid ) {
			std::cerr << "(lambda variant): Test <" << test_name << "> failed, output matrix is invalid" << std::endl;
			return FAILED;
		}
	}

	return SUCCESS;
}

template< typename D >
void grb_program(const long& n, RC& rc) {
	rc = SUCCESS;

	Matrix<D>  I(n, n, n),
	             I_transposed(n, n, n),
	             One_row(n, n, n),
	             One_col(n, n, n);

	{ // Build matrices
		std::vector<D> values(n, 1);
		std::vector<D> const_indices_zero(n, 0);
		std::vector<size_t> iota_indices(n, 0);
		std::iota(iota_indices.begin(), iota_indices.end(), 0);
		std::vector<size_t> reverse_iota_indices(n, 0);
		for (long i = n-1; i >= 0; i--) { reverse_iota_indices[i] = i; }

		buildMatrixUnique(I, iota_indices.data(), iota_indices.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(I, "identity");

		buildMatrixUnique(I_transposed, iota_indices.data(), reverse_iota_indices.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(I_transposed, "transposed-identity");

		buildMatrixUnique(One_row, iota_indices.data(), const_indices_zero.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(One_row, "one-row");

		buildMatrixUnique(One_col, const_indices_zero.data(), iota_indices.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(One_col, "one-column");
	}

	// Test 01: Select <diagonal>
	rc = rc ? rc : test_case(I, operators::select::is_diagonal<D>(),
	          "Test 01: Select <diagonal> out of <identity>");
	rc = rc ? rc : test_case(I_transposed, operators::select::is_diagonal<D>(),
	          "Test 01: Select <diagonal> out of <transposed-identity>");
	rc = rc ? rc : test_case(One_row, operators::select::is_diagonal<D>(),
	          "Test 01: Select <diagonal> out of <one-row>");
	rc = rc ? rc : test_case(One_col, operators::select::is_diagonal<D>(),
	          "Test 01: Select <diagonal> out of <one-column>");

	// Test 02: Select <strict-lower>
	rc = rc ? rc : test_case(I, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <strict-lower> out of <identity>");
	rc = rc ? rc : test_case(I_transposed, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <strict-lower> out of <transposed-identity>");
	rc = rc ? rc : test_case(One_row, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <strict-lower> out of <one-row>");
	rc = rc ? rc : test_case(One_col, operators::select::is_strictly_lower<D>(),
	          "Test 02: Select <strict-lower> out of <one-column>");

	// Test 03: Select <strict-upper>
	rc = rc ? rc : test_case(I, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <strict-lower> out of <identity>");
	rc = rc ? rc : test_case(I_transposed, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <strict-lower> out of <transposed-identity>");
	rc = rc ? rc : test_case(One_row, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <strict-lower> out of <one-row>");
	rc = rc ? rc : test_case(One_col, operators::select::is_strictly_upper<D>(),
	          "Test 03: Select <strict-lower> out of <one-column>");

	// Test 04: Select <lower-or-diag>
	rc = rc ? rc : test_case(I, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <lower-or-diag> out of <identity>");
	rc = rc ? rc : test_case(I_transposed, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <lower-or-diag> out of <transposed-identity>");
	rc = rc ? rc : test_case(One_row, operators::select::is_lower_or_diagonal<D>(),
	          "Test 04: Select <lower-or-diag> out of <one-row>");
	rc = rc ? rc : test_case(One_col, operators::select::is_lower_or_diagonal<D>(),
			  "Test 04: Select <lower-or-diag> out of <one-column>");

	// Test 05: Select <upper-or-diag>
	rc = rc ? rc : test_case(I, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <upper-or-diag> out of <identity>");
	rc = rc ? rc : test_case(I_transposed, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <upper-or-diag> out of <transposed-identity>");
	rc = rc ? rc : test_case(One_row, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <upper-or-diag> out of <one-row>");
	rc = rc ? rc : test_case(One_col, operators::select::is_upper_or_diagonal<D>(),
	          "Test 05: Select <upper-or-diag> out of <one-column>");

	assert(
		collectives<>::allreduce( rc, operators::any_or<RC>() ) == SUCCESS
	);
}

int main(int argc, char** argv) {
	(void) argc;
	(void) argv;

	RC out = SUCCESS;

	std::cout << "This is functional test " << argv[0] << "\n";
	Launcher<AUTOMATIC> launcher;

	const long n = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 1000;

	{
		std::cout << "-- -- Running test with using matrix-type: int" << std::endl;
		if (launcher.exec(&grb_program<int>, n, out, true) != SUCCESS) {
			STDERR_WITH_LINE << "Launching test FAILED\n";
			return 255;
		}
		if (out != SUCCESS) {
			STDERR_WITH_LINE << "Test FAILED (" << toString(out) << ")" << std::endl;
			return out;
		}
	}
	// NOTE: grb::select does not support pattern matrices for the moment
	// {
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
