
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

#include <exception>
#include <iostream>
#include <vector>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/algorithms/bicgstab.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

#include <utils/output_verification.hpp>


using namespace grb;
using namespace algorithms;

/** Parser type */
typedef grb::utils::MatrixFileReader<
	double,
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
	double
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

constexpr const double tol = 0.0001;

/** Default maximum number of solver iterations */
constexpr const size_t max_iters = 10000;

constexpr const double c1 = 0.001;

constexpr const double c2 = 0.001;

struct input {
	char filename[ 1024 ];
	bool direct;
	size_t rep;
	size_t solver_iterations;
};

struct output {
	int error_code;
	size_t rep;
	size_t iterations;
	double residual;
	grb::utils::TimerResults times;
	PinnedVector< double > pinnedVector;
};

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << "Error: no input file given\n";
		return;
	}
	// Parse and store matrix in singleton class
	auto &data = Storage::getData().second;
	try {
		Parser parser( data_in.filename, data_in.direct );
		assert( parser.m() == parser.n() );
		Storage::getData().first.first = parser.n();
		try {
			Storage::getData().first.second = parser.nz();
		} catch( ... ) {
			Storage::getData().first.second = parser.entries();
		}
		/* Once internal issue #342 is resolved this can be re-enabled
		for(
			auto it = parser.begin( PARALLEL );
			it != parser.end( PARALLEL );
			++it
		) {
			data.push_back( *it );
		}*/
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

void grbProgram( const struct input &data_in, struct output &out ) {

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	// assume successful run
	out.error_code = 0;

	// load into GraphBLAS
	const size_t n = Storage::getData().first.first;
	Matrix< double > L( n, n );
	{
		const auto &data = Storage::getData().second;
		const RC rc = buildMatrixUnique(
			L,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			SEQUENTIAL
		);
		/* Once internal issue #342 is resolved this can be re-enabled
		const RC rc = buildMatrixUnique(
			L,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			PARALLEL
		);*/
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
				<< "(" << toString( rc ) << ")." << std::endl;
			return;
		}
	}

	// check number of nonzeroes
	const size_t global_nnz = nnz( L );
	const size_t parser_nnz = Storage::getData().first.second;
	if( global_nnz != parser_nnz ) {
		std::cerr << "Warning: global nnz (" << global_nnz << ") does not equal "
			<< "parser nnz (" << parser_nnz << "). " << "This could naturally occur "
			<< "if the input matrix file employs symmetric storage; in that case, the "
			<< "number of entries is roughly half of the number of nonzeroes.\n";
	}

	// I/O done
	out.times.io = timer.time();
	timer.reset();

	// set up test of default BiCGstab
	Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	grb::operators::subtract< double > minus;
	grb::operators::divide< double > divide;
	Vector< double > x( n ), b( n ), r( n ),
		buf1( n ), buf2( n ), buf3( n ), buf4( n ), buf5( n );

	set( x, static_cast< double >( 1 ) / static_cast< double >( n ) );
	set( b, static_cast< double >( 1 ) );

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	RC rc = SUCCESS;
	if( out.rep == 0 ) {
		timer.reset();
		rc = bicgstab(
			x, L, b,
			data_in.solver_iterations, tol,
			out.iterations, out.residual,
			r, buf1, buf2, buf3, buf4, buf5,
			ring, minus, divide
		);
		double single_time = timer.time();
		if( !(rc == SUCCESS || rc == FAILED) ) {
			std::cerr << "Failure: call to BiCGstab not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == FAILED ) {
			std::cout << "Warning: call to BiCGstab did not converge\n";
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( rc == SUCCESS || rc == FAILED ) {
			if( s == 0 ) {
				if( rc == FAILED ) {
					std::cout << "Info: cold BiCGstab did not converge within ";
				} else {
					std::cout << "Info: cold BiCGstab completed within ";
				}
				std::cout << out.iterations << " iterations. Last computed residual is "
					<< out.residual << ". Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {

			rc = set( x, static_cast< double >( 1 ) / static_cast< double >( n ) );

			if( rc == SUCCESS ) {
				rc = bicgstab(
					x, L, b,
					data_in.solver_iterations, tol,
					out.iterations, out.residual,
					r, buf1, buf2, buf3, buf4, buf5
				);
			}
		}
		time_taken = timer.time();
		out.times.useful = time_taken / static_cast< double >( out.rep );
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for " << out.rep << " BiCGstab "
				<< "calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << grb::toString( rc ) << std::endl;
			std::cout << "\tnumber of BiCGstab iterations: " << out.iterations << "\n";
			std::cout << "\tmilliseconds per iteration: "
				<< ( out.times.useful / static_cast< double >( out.iterations ) )
				<< "\n";
		}
		sleep( 1 );
	}

	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 30;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}

	// output
	out.pinnedVector = PinnedVector< double >( x, SEQUENTIAL );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 8 ) {
		std::cout << "Usage: " << argv[ 0 ]
			<< " <dataset> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (solver iterations) "
			<< "(verification <truth-file>)\n";
		std::cout << "<dataset> and <direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::inner() << ". "
			<< "If this integer is set to zero, the program will select a number of "
			<< "inner iterations that results in at least one second of computation "
			<< "time.\n";
		std::cout << "(outer iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::outer()
			<< ". This integer must be strictly larger than 0.\n";
		std::cout << "(solver iterations) is optional, the default is "
			<< max_iters
			<< ". This integer must be strictly larger than 0.\n";
		std::cout << "(verification <truth-file>) is optional." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get file name
	(void) strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			return 40;
		}
	}

	in.solver_iterations = max_iters;
	if( argc >= 6 ) {
		in.solver_iterations = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 5 ] << " "
				<< "for the maximum number of solver iterations." << std::endl;
			return 50;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ 1024 ];
	if( argc >= 7 ) {
		if( strncmp( argv[ 6 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 8 ) {
				(void) strncpy( truth_filename, argv[ 7 ], 1023 );
				truth_filename[ 1023 ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument."
					<< std::endl;
				return 60;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 6 ] << "\", "
				<< "the optional \"verification\" argument was expected." << std::endl;
			return 70;
		}
	}

	std::cout << "Executable called with parameters " << in.filename << ", "
		<< "inner repititions = " << in.rep << ", "
		<< "outer reptitions = " << outer << ", and "
		<< "solver iterations = " << in.solver_iterations << "."
		<< std::endl;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch I/O
	{
		bool success;
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &ioProgram, in, success, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec(I/O) returns with non-SUCCESS error code \""
				<< grb::toString( rc ) << "\"\n";
			return 73;
		}
		if( !success ) {
			std::cerr << "I/O program caught an exception\n";
			return 77;
		}
	}

	// the output struct
	struct output out;

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< (int)rc << std::endl;
			return 80;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code "
			<< grb::toString( rc ) << std::endl;
		return 90;
	} else if( out.error_code == 0 ) {
		std::cout << "Benchmark completed successfully and took " << out.iterations
			<< " iterations to converge with residual " << out.residual << ".\n";
	}

	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of x is " << out.pinnedVector.size() << ".\n";
	if( out.error_code == 0 && out.pinnedVector.size() > 0 ) {
		std::cout << "First 10 nonzeroes of x are: ( ";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes() && k < 10; ++k ) {
			const auto &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
			std::cout << nonzeroValue << " ";
		}
		std::cout << ")" << std::endl;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		if( verification ) {
			out.error_code = vector_verification(
				out.pinnedVector, truth_filename,
				c1, c2
			);
			if( out.error_code == 0 ) {
				std::cout << "Output vector verificaton was successful!\n";
				std::cout << "Test OK\n";
			} else {
				std::cerr << std::flush;
				std::cout << "Verification FAILED\n";
				std::cout << "Test FAILED\n";
			}
		} else {
			std::cout << "Test OK\n";
		}
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}

