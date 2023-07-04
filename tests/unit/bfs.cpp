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

#include <iostream>
#include <vector>

#include <graphblas/algorithms/bfs.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>

using namespace grb;

grb::Vector< long > stdToGrbVector( const std::vector< long > & in ) {
	grb::Vector< long > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ ) {
		grb::setElement( out, in[ i ], i );
	}
	return out;
}

grb::Vector< long > createGrbVector( const std::initializer_list< long > & in ) {
	grb::Vector< long > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ ) {
		setElement( out, *( in.begin() + i ), i );
	}
	return out;
}

template< typename D >
void printSparseVector( const Vector< D > & v, const std::string & name ) {
	wait( v );
	std::cout << "  [  ";
	if( size( v ) > 50 ) {
		std::cout << "too large to print " << std::endl;
	} else if( nnz( v ) <= 0 ) {
		for( size_t i = 0; i < size( v ); i++ )
			std::cout << "_ ";
	} else {
		size_t nnz_idx = 0;
		auto it = v.cbegin();
		for( size_t i = 0; i < size( v ); i++ ) {
			if( nnz_idx < nnz( v ) && i == it->first ) {
				std::cout << it->second << " ";
				nnz_idx++;
				if( nnz_idx < nnz( v ) )
					++it;
			} else {
				std::cout << "_ ";
			}
		}
	}
	std::cout << " ]  -  "
			  << "Vector \"" << name << "\" (" << size( v ) << ")" << std::endl;
}

struct input_t {
	algorithms::AlgorithmBFS algorithm;
	const Matrix< void > & A;
	const size_t root;
	bool expected_explored_all;
	long expected_max_level;
	const Vector< long > & expected_values;

	// Necessary for distributed backends
	input_t( algorithms::AlgorithmBFS algorithm = algorithms::AlgorithmBFS::LEVELS,
		const Matrix< void > & A = { 0, 0 },
		size_t root = 0,
		bool expected_explored_all = true,
		long expected_max_level = 0,
		const Vector< long > & expected_values = { 0 } ) :
		algorithm( algorithm ),
		A( A ), root( root ), expected_explored_all( expected_explored_all ), expected_max_level( expected_max_level ), expected_values( expected_values ) {}
};

struct output_t {
	RC rc = RC::SUCCESS;
};

void grbProgram( const struct input_t & input, struct output_t & output ) {
	utils::Timer timer;
	long max_level;
	bool explored_all;

	// Allocate output vector
	Vector< long > values( nrows( input.A ) );

	// Run the BFS algorithm
	output.rc = output.rc ? output.rc : algorithms::bfs( input.algorithm, input.A, input.root, explored_all, max_level, values );
	wait();

	{ // Check the outputs
		if( explored_all == input.expected_explored_all ) {
			std::cout << "SUCCESS: explored_all = " << explored_all << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected explored_all = " << input.expected_explored_all << " but got " << explored_all << std::endl;
			output.rc = RC::FAILED;
			return;
		}

		if( max_level == input.expected_max_level ) {
			std::cout << "SUCCESS: max_level = " << max_level << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected max_level " << input.expected_max_level << " but got " << max_level << std::endl;
			output.rc = RC::FAILED;
			return;
		}

		// Check levels by comparing it with the expected one
		if( not std::equal( input.expected_values.cbegin(), input.expected_values.cend(), values.cbegin() ) ) {
			std::cerr << "FAILED: values are incorrect" << std::endl;
			std::cerr << "values != expected_values" << std::endl;
			printSparseVector( values, "values" );
			printSparseVector( input.expected_values, "expected_values" );
			output.rc = RC::FAILED;
			return;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;

	Launcher< EXEC_MODE::AUTOMATIC > launcher;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	/** Matrix A1:
	 *
	 *  2 ───── 0 ───── 1
	 *          │
	 *          │
	 *          │
	 *          3
	 */
	{ /*
	   * Directed version, pattern matrix, root = 0
	   * => 1 step(s) to explore all nodes
	   */
		size_t root = 0;
		std::cout << "-- Running test on A1 (directed, non-pattern, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = true;
		long expected_max_level = 1;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 0 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::SEQUENTIAL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, 1, 1 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, 0, 0 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}

	/** Matrix A2:
	 *
	 *  1 ───── 0 ───── 2 ───── 3
	 */
	{ /*
	   * Directed version, pattern matrix, root = 0
	   * => 2 step(s) to explore all nodes
	   */
		size_t root = 0;
		std::cout << "-- Running test on A2 (directed, pattern, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = true;
		long expected_max_level = 2;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::SEQUENTIAL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, 1, 2 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, 0, 2 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}

	/** Matrix A3:
	 *
	 *  0 ───── 1 ───── 2 ───── 3
	 *  └───────────────────────┘
	 */
	{ /*
	   * Directed version, non-pattern matrix, root = 0
	   * => 3 step(s) to explore all nodes
	   */
		size_t root = 0;
		std::cout << "-- Running test on A3 (directed, non-pattern: int, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = true;
		long expected_max_level = 3;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2, 3 } };
		std::vector< size_t > A_cols { { 1, 2, 3, 0 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::PARALLEL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, 2, 3 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, 1, 2 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}
	{ /*
	   * Undirected version, pattern matrix, root = 0
	   * => 2 step(s) to explore all nodes
	   */
		size_t root = 0;
		std::cout << "-- Running test on A3 (undirected, pattern, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = true;
		long expected_max_level = 2;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 1, 1, 2, 2, 3, 3 } };
		std::vector< size_t > A_cols { { 3, 1, 0, 2, 1, 3, 2, 0 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::PARALLEL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, 2, 1 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, 3, 0 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}

	/** Matrix A4:
	 *
	 *  0 ───── 1 ───── 3
	 *  		│       │
	 *          2 ──────┘
	 */
	{ /*
	   * Directed version, pattern matrix, root = 0
	   * => 3 step(s) to explore all nodes
	   */
		size_t root = 0;
		std::cout << "-- Running test on A4 (directed, pattern, one cycle, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = true;
		long expected_max_level = 3;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2, 3 } };
		std::vector< size_t > A_cols { { 1, 2, 3, 1 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::PARALLEL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, 2, 3 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, 1, 2 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}
	{ /*
	   * Directed version, pattern matrix, root = 1
	   * => Impossible to reach vertex 0
	   */
		size_t root = 1;
		std::cout << "-- Running test on A4 (directed, pattern, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = false;
		long expected_max_level = 2;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2, 3 } };
		std::vector< size_t > A_cols { { 1, 2, 3, 1 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::PARALLEL );
		grb::Vector< long > expected_levels = createGrbVector( { -1, 0, 1, 2 } );
		grb::Vector< long > expected_parents = createGrbVector( { -1, 1, 1, 2 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}

	/** Matrix A5:
	 *
	 *  0 ───── 1 ──x── 2 ───── 3
	 */
	{ /*
	   * Undirected version, pattern matrix, root = 0
	   * => Impossible to reach vertices 2 and 3
	   */
		size_t root = 0;
		std::cout << "-- Running test on A5 (undirected, pattern, root " + std::to_string( root ) + ")" << std::endl;
		bool expected_explored_all = false;
		long expected_max_level = 1;
		Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2, 3 } };
		std::vector< size_t > A_cols { { 1, 0, 3, 2 } };
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), IOMode::PARALLEL );
		grb::Vector< long > expected_levels = createGrbVector( { 0, 1, -1, -1 } );
		grb::Vector< long > expected_parents = createGrbVector( { 0, 0, -1, -1 } );

		{ // Levels
			input_t input( algorithms::AlgorithmBFS::LEVELS, A, root, expected_explored_all, expected_max_level, expected_levels );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		{ // Parents
			input_t input( algorithms::AlgorithmBFS::PARENTS, A, root, expected_explored_all, expected_max_level, expected_parents );
			output_t output;
			RC bench_rc = launcher.exec( &grbProgram, input, output );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}

	std::cout << "Test OK" << std::endl;

	return 0;
}
