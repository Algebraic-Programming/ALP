
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
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <inttypes.h>

#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include "graphblas.hpp"

#define MAX_FN_LENGTH 512

using namespace grb;

struct input {
	size_t test;
	size_t rep;
	char fn[ MAX_FN_LENGTH ];
	bool direct;
};

struct output {
	enum grb::RC error_code;
	grb::utils::TimerResults times;
};

// main label propagation algorithm
void grbProgram( const struct input &data, struct output &out ) {

	grb::utils::Timer timer;

	const size_t s = spmd<>::pid();
#ifndef NDEBUG
	assert( s < spmd<>::nprocs() );
#else
	(void)s;
#endif

	// get input n and test case
	const size_t test = data.test;

	// parse matrix
	timer.reset();
	out.times.io = timer.time();
	grb::utils::MatrixFileReader<
		double,
		std::conditional<
			(sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType )),
			grb::config::RowIndexType,
			grb::config::ColIndexType
		>::type
	> parser(
		data.fn, data.direct
	);
	out.times.io = timer.time();

	// setup containers
	timer.reset();
	grb::Vector< double > vleft( parser.m() ), vright( parser.n() );
	grb::Matrix< double > mx( parser.m(), parser.n() );
	out.times.preamble = timer.time();

	// ingest matrix
	timer.reset();
	out.error_code = buildMatrixUnique( mx, parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
	out.times.io += timer.time();

	if( out.error_code != SUCCESS ) { return; }

	// setup
	Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	switch( test ) {

		// Ax
		case 1: {
			// prepare experiment
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vright, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vleft, 0 );
			}
			out.times.preamble += timer.time();
			if( out.error_code != SUCCESS ) { return; }

			// functional check
			out.error_code = mxv< grb::descriptors::dense >( vleft, mx, vright, ring );
			double * chk = new double[ parser.m() ];
			size_t * cnt = new size_t[ parser.m() ];
			double * mag = new double[ parser.m() ];
			for( size_t i = 0; i < parser.m(); ++i ) { chk[ i ] = mag[ i ] = 0.0; cnt[ i ] = 0;}
			for( const auto &triple : parser ) {
				const size_t i = triple.first.first;
				chk[ i ] += triple.second;
				(void) cnt[ i ]++;
				if( mag[ i ] < fabs( triple.second) ) { mag[ i ] = fabs(triple.second); }
			}
			for( const auto &pair : vleft ) {
				if( cnt[ pair.first ] == 0 ) {
					if( pair.second != 0.0 ) {
						std::cerr << "Verification FAILED; nonzero " << pair.second << " at output vector position " << pair.first << " while no contribution to that index was expected\n";
						out.error_code = FAILED;
					}
				} else {
					const size_t epsilons = mag[ pair.first ] < 1
						? cnt[ pair.first ] + 1
						: cnt[ pair.first ] * ceil(mag[ pair.first ]) + 1;
					const double allowedError =
						epsilons * std::numeric_limits< double >::epsilon();
					if( std::fabs( pair.second - chk[ pair.first ] ) > allowedError ) {
						std::cerr << "Verification FAILED ( " << pair.second << " does not equal " << chk[ pair.first ] << " at output vector position " << pair.first << " )\n";
						out.error_code = FAILED;
					}
				}
			}
			delete [] chk;
			delete [] cnt;
			delete [] mag;
			if( out.error_code != SUCCESS ) { return; }

			// do experiment
			out.error_code = mxv< grb::descriptors::dense >( vleft, mx, vright, ring );
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data.rep; ++i ) {
				out.error_code = mxv< grb::descriptors::dense >( vleft, mx, vright, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// A^Tx
		case 2: {
			// prepare experiment
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vleft, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vright, 0 );
			}
			out.times.preamble += timer.time();
			if( out.error_code != SUCCESS ) { return; }

			// functional check
			out.error_code = mxv< grb::descriptors::dense | grb::descriptors::transpose_matrix >( vright, mx, vleft, ring );
			double * chk = new double[ parser.n() ];
			size_t * cnt = new size_t[ parser.n() ];
			double * mag = new double[ parser.n() ];
			for( size_t i = 0; i < parser.n(); ++i ) { chk[ i ] = mag[ i ] = 0.0; cnt[ i ] = 0;}
			for( const auto &triple : parser ) {
				const size_t i = triple.first.second;
				chk[ i ] += triple.second;
				(void) cnt[ i ]++;
				if( mag[ i ] < fabs( triple.second ) ) { mag[ i ] = fabs( triple.second ); }
			}
			for( const auto &pair : vright ) {
				if( cnt[ pair.first ] == 0 ) {
					if( pair.second != 0.0 ) {
						std::cerr << "Verification FAILED; nonzero " << pair.second << " at output vector position " << pair.first << " while no contribution to that index was expected\n";
						out.error_code = FAILED;
					}
				} else {
					const double releps = pair.second == 0 ? 0 : mag[ pair.first ] / fabs(pair.second);
					const size_t epsilons = static_cast<size_t>(releps * static_cast<double>(cnt[ pair.first ])) + 1;
					if( ! grb::utils::equals( pair.second, chk[ pair.first ], epsilons ) ) {
						std::cerr << "Verification FAILED ( " << pair.second << " does not equal " << chk[ pair.first ] << " at output vector position " << pair.first << " )\n";
						out.error_code = FAILED;
					}
				}
			}
			delete [] chk;
			delete [] cnt;
			delete [] mag;
			if( out.error_code != SUCCESS ) { return; }

			// do experiment
			out.error_code = mxv< grb::descriptors::dense | descriptors::transpose_matrix >( vright, mx, vleft, ring );
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data.rep; ++i ) {
				out.error_code = mxv< grb::descriptors::dense | descriptors::transpose_matrix >( vright, mx, vleft, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// xA
		case 3: {
			// do experiment
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vleft, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vright, 0 );
			}
			out.times.preamble += timer.time();
			if( out.error_code != SUCCESS ) { return; }

			// functional check
			out.error_code = vxm< grb::descriptors::dense >( vright, vleft, mx, ring );
			double * chk = new double[ parser.n() ];
			size_t * cnt = new size_t[ parser.n() ];
			double * mag = new double[ parser.n() ];
			for( size_t i = 0; i < parser.n(); ++i ) { chk[ i ] = mag[ i ] = 0.0; cnt[ i ] = 0;}
			for( const auto &triple : parser ) {
				const size_t i = triple.first.second;
				chk[ i ] += triple.second;
				(void) cnt[ i ]++;
				if( mag[ i ] < fabs(triple.second) ) { mag[ i ] = fabs(triple.second); }
			}
			for( const auto &pair : vright ) {
				if( cnt[ pair.first ] == 0 ) {
					if( pair.second != 0.0 ) {
						std::cerr << "Verification FAILED; nonzero " << pair.second << " at output vector position " << pair.first << " while no contribution to that index was expected\n";
						out.error_code = FAILED;
					}
				} else {
					const double releps = pair.second == 0 ? 0 : mag[ pair.first ] / fabs(pair.second);
					const size_t epsilons = static_cast<size_t>(releps * static_cast<double>(cnt[ pair.first ])) + 1;
					if( ! grb::utils::equals( pair.second, chk[ pair.first ], epsilons ) ) {
						std::cerr << "Verification FAILED ( " << pair.second << " does not equal " << chk[ pair.first ] << " at output vector position " << pair.first << " )\n";
						out.error_code = FAILED;
					}
				}
			}
			delete [] chk;
			delete [] cnt;
			delete [] mag;
			if( out.error_code != SUCCESS ) { return; }

			// do experiment
			out.error_code = vxm< grb::descriptors::dense >( vright, vleft, mx, ring );
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data.rep; ++i ) {
				out.error_code = vxm< grb::descriptors::dense >( vright, vleft, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// xA^T
		case 4: {
			// prepare experiment
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vright, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vleft, 0 );
			}
			out.times.preamble += timer.time();
			if( out.error_code != SUCCESS ) { return; }

			// functional check
			out.error_code = vxm< grb::descriptors::dense | grb::descriptors::transpose_matrix >( vleft, vright, mx, ring );
			double * chk = new double[ parser.m() ];
			size_t * cnt = new size_t[ parser.m() ];
			double * mag = new double[ parser.m() ];
			for( size_t i = 0; i < parser.m(); ++i ) { chk[ i ] = mag[ i ] = 0.0; cnt[ i ] = 0;}
			for( const auto &triple : parser ) {
				const size_t i = triple.first.first;
				chk[ i ] += triple.second;
				(void) cnt[ i ]++;
				if( fabs(triple.second) > mag[ i ] ) { mag[ i ] = fabs(triple.second); }
			}
			for( const auto &pair : vleft ) {
				if( cnt[ pair.first ] == 0 ) {
					if( pair.second != 0.0 ) {
						std::cerr << "Verification FAILED; nonzero " << pair.second << " at output vector position " << pair.first << " while no contribution to that index was expected\n";
						out.error_code = FAILED;
					}
				} else {
					const double releps = pair.second == 0 ? 0 : mag[ pair.first ] / fabs(pair.second);
					const size_t epsilons = static_cast<size_t>(releps * static_cast<double>(cnt[ pair.first ])) + 1;
					if( ! grb::utils::equals( pair.second, chk[ pair.first ], epsilons ) ) {
						std::cerr << "Verification FAILED ( " << pair.second << " does not equal " << chk[ pair.first ] << " at output vector position " << pair.first << " )\n";
						out.error_code = FAILED;
					}
				}
			}
			delete [] chk;
			delete [] cnt;
			delete [] mag;
			if( out.error_code != SUCCESS ) { return; }

			// do experiment
			out.error_code = vxm< grb::descriptors::dense | grb::descriptors::transpose_matrix >( vleft, vright, mx, ring );
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data.rep; ++i ) {
				out.error_code = vxm< grb::descriptors::dense | grb::descriptors::transpose_matrix >( vleft, vright, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		default:
			std::cerr << "Unknown test case " << test << std::endl;
			break;
	}
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 6 ) {
		std::cout << "Usage: " << argv[ 0 ]
			  << " <matrix file> <direct/indirect> <test case> (inner repititions) (outer repititions)"
			  << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	(void) strncpy( in.fn, argv[ 1 ], MAX_FN_LENGTH-1 );
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}
	in.test = atoi( argv[ 3 ] );
	in.rep = grb::config::BENCHMARKING::inner();
	size_t outer = grb::config::BENCHMARKING::outer();
	char * end = NULL;
	if( argc >= 5 ) {
		in.rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument for number of inner "
						 "repititions."
					  << std::endl;
			return 25;
		}
	}
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument for number of outer "
						 "reptitions."
					  << std::endl;
			return 25;
		}
	}

	std::cout << "Executable called with parameters: filename " << in.fn << " (" << (in.direct?"direct":"indirect") << "), test case ";
	switch( in.test ) {
		case 1:
			std::cout << "Ax";
			break;
		case 2:
			std::cout << "A^Tx";
			break;
		case 3:
			std::cout << "xA";
			break;
		case 4:
			std::cout << "xA^T";
			break;
		default:
			std::cout << " UNRECOGNISED TEST CASE, ABORTING.\nTest FAILED.\n" << std::endl;
			return 30;
	}
	std::cout << ", inner = " << in.rep << ", outer = " << outer << "." << std::endl;

	// the output struct
	struct output out;

	// run the program one time to infer number of inner repititions
	if( in.rep == 0 ) {
		in.rep = 1;
		grb::Launcher< AUTOMATIC > launcher;
		const enum grb::RC rc = launcher.exec( &grbProgram, in, out, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
			return 40;
		}
		// set guesstimate for inner repititions: a single experiment should take at least a second
		in.rep = static_cast< double >( 1000.0 / out.times.useful ) + 1;
		std::cout << "Auto-selected number of inner repetitions is " << in.rep << " (at an estimated time of " << out.times.useful << " ms. of useful work per benchmark).\n";
	}

	// start benchmarks
	grb::Benchmarker< AUTOMATIC > benchmarker;
	const enum grb::RC rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
		return 50;
	}

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED.\n\n";
		return out.error_code;
	}
	std::cout << "Test OK.\n\n";
	return 0;
}

