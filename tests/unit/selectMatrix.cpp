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

constexpr bool Debug = false;

namespace {
	template<class Iterator>
	void printSparseMatrixIterator(size_t rows, size_t cols, Iterator begin, Iterator end, const std::string& name = "",
	                               std::ostream& os = std::cout) {
		std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
		if (rows > 256 || cols > 256) {
			os << "   Matrix too large to print" << std::endl;
		} else {
			// os.precision( 3 );
			for (size_t y = 0; y < rows; y++) {
				os << std::string(3, ' ');
				for (size_t x = 0; x < cols; x++) {
					auto nnz_val = std::find_if(
						begin, end, [ y, x ](const typename std::iterator_traits<Iterator>::value_type& a) {
							return a.first.first == y && a.first.second == x;
						});
					if (nnz_val != end)
						os << std::fixed << (*nnz_val).second;
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

void grb_program(const long& n, RC& rc) {
	rc = SUCCESS;

	Matrix<int> I(n, n, n), I_tr(n, n, n);
	{ // Build matrices I and I_tr
		std::vector<int> values(n, 1);
		std::vector<size_t> rows_indices(n, 0);
		std::iota(rows_indices.begin(), rows_indices.end(), 0);
		buildMatrixUnique(I, rows_indices.data(), rows_indices.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(I, "identity");

		std::vector<size_t> cols_indices(n, 0);
		std::iota(cols_indices.begin(), cols_indices.end(), 0);
		std::reverse(cols_indices.begin(), cols_indices.end());
		buildMatrixUnique(I_tr, rows_indices.data(), cols_indices.data(), values.data(), n, SEQUENTIAL);
		printSparseMatrix<Debug>(I_tr, "transposed-identity");
	}

	// Test 01: Select <diagonal> out of <identity>
	rc = rc ? rc : test_case(I, operators::select::is_diagonal<int>(),
	          "Test 01: Select <diagonal> out of <identity>");

	// Test 02: Select <diagonal> out of <transposed-identity>
	rc = rc ? rc : test_case(I_tr, operators::select::is_diagonal<int>(),
	          "Test 02: Select <diagonal> out of <transposed-identity>");

	// Test 03: Select <strict-lower> out of <identity>
	rc = rc ? rc : test_case(I, operators::select::is_strictly_lower<int>(),
	          "Test 03: Select <strict-lower> out of <identity>");

	// Test 04: Select <strict-lower> out of <transposed-identity>
	rc = rc ? rc : test_case(I_tr, operators::select::is_strictly_lower<int>(),
	          "Test 04: Select <strict-lower> out of <transposed-identity>");

	// Test 05: Select <strict-upper> out of <identity>
	rc = rc ? rc : test_case(I, operators::select::is_strictly_upper<int>(),
	          "Test 05: Select <strict-lower> out of <identity>");

	// Test 06: Select <strict-upper> out of <transposed-identity>
	rc = rc ? rc : test_case(I_tr, operators::select::is_strictly_upper<int>(),
	          "Test 06: Select <strict-lower> out of <transposed-identity>");

	// Test 07: Select <lower-or-diag> out of <identity>
	rc = rc ? rc : test_case(I, operators::select::is_lower_or_diagonal<int>(),
	          "Test 07: Select <lower-or-diag> out of <identity>");

	// Test 08: Select <lower-or-diag> out of <transposed-identity>
	rc = rc ? rc : test_case(I_tr, operators::select::is_lower_or_diagonal<int>(),
	          "Test 08: Select <lower-or-diag> out of <transposed-identity>");

	// Test 09: Select <upper-or-diag> out of <identity>
	rc = rc ? rc : test_case(I, operators::select::is_upper_or_diagonal<int>(),
	          "Test 09: Select <upper-or-diag> out of <identity>");

	// Test 10: Select <upper-or-diag> out of <transposed-identity>
	rc = rc ? rc : test_case(I_tr, operators::select::is_upper_or_diagonal<int>(),
	          "Test 10: Select <upper-or-diag> out of <transposed-identity>");

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

	const long n = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 10;
	std::cout << "-- Running test with n=" << n << std::endl;

	if (launcher.exec(&grb_program, n, out, true) != SUCCESS) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}

	if (out != SUCCESS) {
		std::cout << "Test FAILED (" << toString(out) << ")" << std::endl;
		return out;
	}

	std::cout << std::flush;
	std::cerr << std::flush << "Test OK" << std::endl;
	return 0;
}
