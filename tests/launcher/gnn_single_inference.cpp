
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

#include <graphblas/algorithms/gnn_single_inference.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>

using namespace grb;
using namespace algorithms;

#define MAX_LEN 1000

struct input {
	char dataset_path[ MAX_LEN + 1 ];
	size_t neurons;
	size_t layers;
	size_t input_vector_offset;
	bool direct;
	size_t rep;
};

struct output {
	int error_code;
	size_t rep;
	size_t iterations;
	grb::utils::TimerResults times;
	PinnedVector< double > pinnedVector;
};

void grbProgram( const struct input & data_in, struct output & out ) {

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// assume successful run
	out.error_code = 0;

	char weights_path[ MAX_LEN + 1 ];
	if( strlen( data_in.dataset_path ) + strlen( "/WEIGHTS-HPEC" ) > MAX_LEN ) {
		std::cerr << "Failure: given dataset path is too long (please use a "
					 "shorter dataset path)"
				  << std::endl;
		return;
	}
	strcpy( weights_path, data_in.dataset_path );
	strcat( weights_path, "/WEIGHTS-HPEC" );

	char input_vector_path[ MAX_LEN + 1 ];
	if( strlen( data_in.dataset_path ) + strlen( "/MNIST-HPEC" ) > MAX_LEN ) {
		std::cerr << "Failure: given dataset path is too long (please use a "
					 "shorter dataset path)"
				  << std::endl;
		return;
	}
	strcpy( input_vector_path, data_in.dataset_path );
	strcat( input_vector_path, "/MNIST-HPEC" );

	double biases[ data_in.layers ];

	if( data_in.neurons == 1024 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases[ i ] = -0.30;
		}
	} else if( data_in.neurons == 4096 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases[ i ] = -0.35;
		}
	} else if( data_in.neurons == 16384 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases[ i ] = -0.40;
		}
	} else if( data_in.neurons == 65536 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases[ i ] = -0.45;
		}
	} else {
		std::cerr << "Failure: the number of neurons does not correspond to a "
					 "known dataset"
				  << std::endl;
		return;
	}

	size_t n;
	out.times.io = timer.time();
	timer.reset();

	Matrix< double > ** L = new Matrix< double > *[ data_in.layers ];

	for( size_t i = 0; i < data_in.layers; i++ ) {

		// get the names of the input files for all layers correct
		std::ostringstream oss;
		oss << weights_path << "/neuron" << data_in.neurons << "/n" << data_in.neurons << "-l" << i + 1 << ".mtx";
		std::string filename = oss.str();

		// create local parser
		grb::utils::MatrixFileReader< double,
			std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
			parser( filename.c_str(), data_in.direct );
		assert( parser.m() == parser.n() );
		assert( data_in.neurons == parser.n() );
		n = parser.n();

		// load into GraphBLAS
		L[ i ] = new Matrix< double >( n, n );
		{
			const RC rc = buildMatrixUnique( *( L[ i ] ), parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
			//			const RC rc = buildMatrixUnique( *L[i], parser.begin( PARALLEL ), parser.end( PARALLEL), PARALLEL);
			if( rc != SUCCESS ) {
				std::cerr << "Failure: call to buildMatrixUnique did not "
							 "succeed ("
						  << toString( rc ) << ")." << std::endl;
				return;
			}
		}

		// check number of nonzeroes
		try {
			const size_t global_nnz = nnz( *( L[ i ] ) );
			const size_t parser_nnz = parser.nz();
			if( global_nnz != parser_nnz ) {
				std::cerr << "Failure: global nnz (" << global_nnz << ") does not equal parser nnz (" << parser_nnz << ")." << std::endl;
				return;
			}
		} catch( const std::runtime_error & ) {
			std::cout << "Info: nonzero check skipped as the number of "
						 "nonzeroes cannot be derived from the matrix file "
						 "header. The grb::Matrix reports "
					  << nnz( *( L[ i ] ) ) << "nonzeroes.\n";
		}
	}

	// get the name of the input files for the vector correct
	std::ostringstream oss;
	oss << input_vector_path << "/test" << data_in.neurons << "/sparse-images-" << data_in.neurons << "_" << data_in.input_vector_offset << ".mtx";
	std::string vector_filename = oss.str();

	// create local parser
	grb::utils::MatrixFileReader< double,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parser( vector_filename, data_in.direct );
	assert( data_in.neurons == parser.n() );
	n = parser.n();

	// load into GraphBLAS
	Matrix< double > Lvin( n, n );
	{
		const RC rc = buildMatrixUnique( Lvin, parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
		//		const RC rc = buildMatrixUnique( Lvin, parser.begin( PARALLEL ), parser.end( PARALLEL), PARALLEL);
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed (" << toString( rc ) << ")." << std::endl;
			return;
		}
	}

	grb::Vector< double > vout( n ), vin( n ), temp( n );

	// that's a simple way to get the input vector by reading it as a matrix using the existing
	// parser and then apply the vxm operation on the matrix and on a vector of ones
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > realRing;

	set( temp, 1 );
	grb::vxm( vin, temp, Lvin, realRing );

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	RC rc = SUCCESS;
	if( out.rep == 0 ) {
		timer.reset();
		rc = gnn_single_inference( vout, vin, L, biases, data_in.layers, temp );
		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to conjugate_gradient did not succeed (" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( rc == SUCCESS ) {
			if( s == 0 ) {
				std::cout << "Info: cold gnn_single_inference completed within " << out.iterations
						  << " iterations. Last computed residual is \" << "
							 "out.residual << \". Time taken was "
						  << single_time << " ms. Deduced inner repetitions parameter of " << out.rep << " to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( rc == SUCCESS ) {
				rc = gnn_single_inference( vout, vin, L, biases, data_in.layers, temp );
			}
		}
		time_taken = timer.time();
		if( rc == SUCCESS ) {
			out.times.useful = time_taken / static_cast< double >( out.rep );
		}
		sleep( 1 );
#ifndef NDEBUG
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for a " << out.rep << " GNN Single Inference calls (hot start): " << out.times.useful << ". Error code is " << out.error_code << std::endl;
		}
#endif
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
	out.pinnedVector = PinnedVector< double >( vout, SEQUENTIAL );

	// free memory
	for( size_t i = 0; i < data_in.layers; i++ ) {
		delete L[ i ];
	}

	delete[] L;

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 6 || argc > 8 ) {
		std::cout << "Usage: " << argv[ 0 ]
				  << " <dataset path> <neurons> <layers> <input vector offset> "
					 "<direct/indirect> (inner iterations) (outer "
					 "iterations)\n";
		std::cout << "<dataset path> <neurons> <layers> <input vector offset> "
					 "and <direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner()
				  << ". If set to zero, the program will select a number of "
					 "iterations approximately required to take at least one "
					 "second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". This value must be strictly larger than 0." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get the dataset path
	if( strlen( argv[ 1 ] ) > MAX_LEN ) {
		std::cerr << "Given dataset path is too long (please use a shorter "
					 "dataset path)"
				  << std::endl;
		return 1;
	}
	strcpy( in.dataset_path, argv[ 1 ] );

	// get the number of neurons
	in.neurons = atoi( argv[ 2 ] );

	// get the number of layers
	in.layers = atoi( argv[ 3 ] );

	// get the offset of the input vector
	in.input_vector_offset = atoi( argv[ 4 ] );

	// get direct or indirect addressing
	if( strncmp( argv[ 5 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = NULL;
	if( argc >= 7 ) {
		in.rep = strtoumax( argv[ 6 ], &end, 10 );
		if( argv[ 6 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 6 ] << " for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 8 ) {
		outer = strtoumax( argv[ 7 ], &end, 10 );
		if( argv[ 7 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 7 ] << " for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	std::cout << "Executable called with parameters: neurons = " << in.neurons << ", layers = " << in.layers << ", input vector offset = " << in.input_vector_offset
			  << ", inner repititions = " << in.rep << ", and outer reptitions = " << outer << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
			return 6;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code " << grb::toString( rc ) << std::endl;
		return 8;
	} else if( out.error_code == 0 ) {
		std::cout << "Benchmark completed successfully.\n";
	}

	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of out is " << out.pinnedVector.length() << ".\n";
	if( out.error_code == 0 && out.pinnedVector.length() > 0 ) {
		std::cout << "First 10 elements of out are: ( ";
		if( out.pinnedVector.mask( 0 ) ) {
			std::cout << out.pinnedVector[ 0 ];
		} else {
			std::cout << "0";
		}
		for( size_t i = 1; i < out.pinnedVector.length() && i < 10; ++i ) {
			std::cout << ", ";
			if( out.pinnedVector.mask( i ) ) {
				std::cout << out.pinnedVector[ i ];
			} else {
				std::cout << "0";
			}
		}
		std::cout << " )" << std::endl;
	}

	double sum_out = 0.0;
	for( size_t i = 0; i < out.pinnedVector.length(); ++i ) {
		sum_out += out.pinnedVector[ i ];
	}
	std::cout << "SUM = " << sum_out << std::endl;

	if( out.error_code != 0 ) {
		std::cout << "Test FAILED.\n";
	} else {
		std::cout << "Test OK.\n";
	}
	std::cout << std::endl;

	// done
	return 0;
}
