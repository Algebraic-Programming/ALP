
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

#include <map>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/algorithms/label.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

#include <utils/print_vec_mat.hpp>


using namespace grb;

/** Parser type */
typedef grb::utils::MatrixFileReader<
	void,
	std::conditional<
		(sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
		grb::config::RowIndexType,
		grb::config::ColIndexType
	>::type
> Parser;

/** Nonzero type */
typedef internal::NonzeroStorage<
	grb::config::RowIndexType,
	grb::config::ColIndexType,
	void
> NonzeroT;

/** In-memory storage type */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>
> Storage;

constexpr size_t MaxPrinting = 10;

// forward declaration of the graph dataset parser
bool readEdges(
	std::string filename, bool use_indirect,
	size_t * n, size_t * nz, size_t ** I, size_t ** J, double ** weights
);

struct input {
	char filename[ 1024 ];
	bool direct;
	size_t n;
};

struct output {
	RC error_code;
	PinnedVector< double > f;
	grb::utils::TimerResults times;
};

using namespace grb;

// take a sparse matrix, convert to dense format and display with a message

void printMatrix( Matrix< double > &sparse, const size_t n, const char * message ) {
	// only print small matrices
	if( n > MaxPrinting )
		return;

#ifdef _GRB_WITH_LPF
	(void) sparse;
	(void) message;
#else
	// allocate and clear a dense matrix
	print_matrix< double >( sparse, 0, message );
#endif
}

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// sanity check on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << "Error: no file name given as input.\n";
		return;
	}

	// Parse and store matrix in singleton class
	try {
		auto &data = Storage::getData().second;
		Parser parser( data_in.filename, data_in.direct );
		if( parser.m() == parser.n() ) {
			std::cerr << "Error: input matrix must be square.\n";
			return;
		}
		Storage::getData().first.first = parser.n();
		try {
			Storage::getData().first.second = parser.nz();
		} catch( ... ) {
			Storage::getData().first.second = parser.entries();
		}
		// note: no parallel I/O since this test artifically makes the input symmetric
		// (using sequential code)
		for(
			auto it = parser.begin( SEQUENTIAL );
			it != parser.end( SEQUENTIAL );
			++it
		) {
			data.push_back( NonzeroT( *it ) );
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}
	success = true;
}

// main label propagation algorithm
void grbProgram( const struct input &data_in, struct output &out ) {

	// initial declarations
	grb::utils::Timer timer;
	const size_t n = data_in.n;
	out.error_code = SUCCESS;

	// load problem set
	timer.reset();
	size_t nz = Storage::getData().first.second;
	Matrix< double > W( n, n );
	{
		std::vector< size_t > I;
		std::vector< size_t > J;
		std::vector< double > weights;
		const auto &data = Storage::getData().second;
		for( const auto &pair : data ) {
			if( pair.first <= pair.second ) {
				(void) --nz;
			} else {
				I.push_back( pair.first );
				J.push_back( pair.second );
			}
		}
		if( I.size() != J.size() ) {
			std::cerr << "Error: expected I and J arrays to be of equal length\n";
			out.error_code = PANIC;
			return;
		}
		if( I.size() == nz ) {
			std::cerr << "Error: expected the size of I to be equal to nz\n";
			out.error_code = PANIC;
			return;
		}
		// make symmetric
		for( size_t k = 0; k < nz; ++k ) {
			if( I[ k ] != J[ k ] ) {
				I.push_back( J[ k ] );
				J.push_back( I[ k ] );
			}
		}
		if( I.size() != J.size() ) {
			std::cerr << "Error: expected I and J arrays to be of equal length\n";
			out.error_code = PANIC;
			return;
		}

		// construct symmetric weights matrix with nz non-zeros in total
		for( size_t i = 0; i < nz; i++ ) {
			const double random = // random between 0.01 .. 1.00 (inclusive)
				static_cast< double >((rand() % 100) + 1) / 100.0;
			weights.push_back( random );
		}

		// fill in the symmetric values based on the former half
		{
			const size_t remaining = I.size() - nz;
			for( size_t k = 0; k < remaining; k++ ) {
				if( I[ k ] != J[ k ] ) {
					weights.push_back( weights[ k ] );
				}
			}
		}
		nz = I.size();
		if( weights.size() != nz ) {
			std::cerr << "Error: expected weights, I, and J arrays to be of equal "
				<< "length\n";
			out.error_code = PANIC;
			return;
		}
		const RC rc = buildMatrixUnique( W, &(I[ 0 ]), &(J[ 0 ]), &(weights[ 0 ]), nz,
			SEQUENTIAL );
		if( rc != SUCCESS ) {
			std::cerr << "Error: call to buildMatrixUnique failed (" << grb::toString(rc)
				<< ")\n";
			out.error_code = rc;
			return;
		}
	}

	out.times.io = timer.time();
	timer.reset();

	// n nodes with 20% labelled
	srand( 314159 );
	const size_t l = ( size_t )( (double)n * 0.2 );
	double * const labels = new double[ n ];
	for( size_t i = 0; i < n; i++ ) {
		labels[ i ] = ( i < l ) ? ( rand() % 2 ) : 0; // 0,1
	}

	// create the intial set of l input labels in the vector y
	Vector< double > y( n );
	Vector< double > f( n );
	RC rc = buildVector( y, &(labels[ 0 ]), &(labels[ 0 ]) + n, SEQUENTIAL );

	// create the symmetric weight matrix W, representing the weighted graph
	rc = rc ? rc : resize( W, nz );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		out.error_code = rc;
		return;
	}

	// print W (if it is small enough
	printMatrix( W, n, "Symmetric weight matrix W" );

	// pre-amble done
	out.times.preamble = timer.time();

	// run and time experiment
	timer.reset();
	algorithms::label( f, y, W, n, l );
	out.times.useful = timer.time();

	// output result
	timer.reset();
	out.f = PinnedVector< double >( f, SEQUENTIAL );
	out.times.postamble = timer.time();
	timer.reset();

	// clean
	delete [] labels;
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	size_t outer = grb::config::BENCHMARKING::outer();
	size_t inner = grb::config::BENCHMARKING::inner();

	// sanity check
	if( argc < 3 || argc > 5 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <dataset> <direct/indirect> "
			<< "(number of inner iterations) (number of outer iterations)"
			<< std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	if( strlen( argv[ 1 ] ) > 1023 ) {
		std::cerr << "Could not parse filename: too long." << std::endl;
		return 10;
	}
	(void) strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}
	grb::utils::MatrixFileReader< void > reader( in.filename, in.direct );
	in.n = reader.n();
	if( in.n != reader.m() ) {
		std::cerr << "The given matrix is not square." << std::endl;
		return 20;
	}
	char * end = nullptr;
	if( argc >= 4 ) {
		inner = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument for number of inner repetitions."
				<< std::endl;
			return 30;
		}
	}
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument for number of outer repetitions."
				<< std::endl;
			return 40;
		}
	}

	std::cout << "Executable called with parameters "
		<< "filename " << in.filename << ", "
		<< "direct = " << in.direct << ", and "
		<< "#vertices = " << in.n << std::endl;

	// the output struct
	struct output out;

	{
		grb::Launcher< AUTOMATIC > launcher;
		bool io_success;
		enum grb::RC rc = launcher.exec( &ioProgram, in, io_success, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec(I/O) returns with non-SUCCESS error code "
				<< grb::toString(rc) << "\n";
			return 43;
		}
		if( !io_success ) {
			std::cerr << "Error: I/O subprogram failed\n";
			return 47;
		}
	}

	grb::Benchmarker< AUTOMATIC > launcher;
	enum RC rc = launcher.exec( &grbProgram, in, out, inner, outer, true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code "
			<< grb::toString(rc) << std::endl;
		return 50;
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED\n\n";
		return 255;
	}
	std::cout << "Test OK\n\n";
	return 0;
}

