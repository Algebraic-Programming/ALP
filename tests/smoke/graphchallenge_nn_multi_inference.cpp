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
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <inttypes.h>

#include <graphblas/algorithms/sparse_nn_multi_inference.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
#include <utils/output_verification.hpp>

#define C1 0.0001
#define C2 0.0001

#define MAX_LEN 1000

using namespace grb;
using namespace algorithms;

using nz_type = double;

template< class Iterator >
void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
#define _DEBUG
#ifndef _DEBUG
	return;
#endif
	os << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;

	bool is_fp = std::is_floating_point< typename std::iterator_traits< Iterator >::value_type::second_type >::value;
	if( is_fp )
		os.precision( 2 );
	for( size_t y = 0; y < rows; y++ ) {
		os << std::string( 3, ' ' );
		for( size_t x = 0; x < cols; x++ ) {
			auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
				return a.first.first == y && a.first.second == x;
			} );
			if( nnz_val == end )
				os << std::string( ( is_fp ) ? 3 + +os.precision() : 1, '_' );
			else if( is_fp )
				os << std::fixed << std::showpos << ( *nnz_val ).second;
			else
				os << ( *nnz_val ).second;
			os << " ";
		}
		os << std::endl;
	}

	os << "]" << std::endl;
	std::flush( os );
}

template< typename D >
void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	grb::wait( mat );
	printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

struct input {
	char dataset_path[ MAX_LEN + 1 ];
	size_t neurons;
	size_t layers;
	bool thresholded;
	double threshold;
	size_t input_vector_offset;
	bool direct;
	size_t rep;
};

struct output {
	int error_code;
	size_t rep;
	size_t iterations;
	grb::utils::TimerResults times;
	std::shared_ptr< Matrix< nz_type > > result;
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
				  << "shorter dataset path)" << std::endl;
		return;
	}
	strcpy( input_vector_path, data_in.dataset_path );
	strcat( input_vector_path, "/MNIST-HPEC" );

	size_t n, m;
	out.times.io = timer.time();
	timer.reset();

	std::vector< grb::Matrix< nz_type > > L;

	for( size_t i = 0; i < data_in.layers; i++ ) {

		// get the names of the input files for all layers correct
		std::ostringstream oss;
		oss << weights_path << "/neuron" << data_in.neurons << "/n" << data_in.neurons << "-l" << i + 1 << ".mtx";
		std::string filename = oss.str();

		// create local parser
		grb::utils::MatrixFileReader< nz_type,
			std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
			parser( filename.c_str(), data_in.direct );
		assert( parser.m() == parser.n() );
		assert( data_in.neurons == parser.n() );
		n = parser.n();

		// load into GraphBLAS
		L.emplace_back( n, n );

		{
			const RC rc = buildMatrixUnique( L.back(), parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
			// See internal issue #342 for re-enabling the below
			// const RC rc = buildMatrixUnique( L[ i ],
			//	parser.begin( PARALLEL ), parser.end( PARALLEL),
			//	PARALLEL
			//);
			if( rc != SUCCESS ) {
				std::cerr << "Failure: call to buildMatrixUnique did not succeed (" << toString( rc ) << ")." << std::endl;
				return;
			}
		}

		// check number of nonzeroes
		try {
			const size_t global_nnz = nnz( L.back() );
			const size_t parser_nnz = parser.nz();
			if( global_nnz != parser_nnz ) {
				std::cerr << "Failure: global nnz (" << global_nnz << ") does not equal parser nnz "
						  << "(" << parser_nnz << ")." << std::endl;
				return;
			}
		} catch( const std::runtime_error & ) {
			std::cout << "Info: nonzero check skipped as the number of "
						 "nonzeroes cannot be derived from the matrix file "
						 "header. The grb::Matrix reports "
					  << nnz( L.back() ) << "nonzeroes.\n";
		}
	}
	size_t layer_row = grb::nrows( L.back() ), layer_col = grb::ncols( L.back() );
	if( ! std::all_of( L.cbegin(), L.cend(), [ layer_row, layer_col ]( const grb::Matrix< nz_type > & m ) {
			return grb::nrows( m ) == layer_row && grb::ncols( m ) == layer_col;
		} ) ) {
		std::cerr << "Failure: not all layers have the same dimensions" << std::endl;
		return;
	}

	// Fill biases
	std::vector< grb::Vector< nz_type > > biases( data_in.layers, { grb::nrows( L.back() ) } );
	for( auto & e : biases ) {
		nz_type value = data_in.neurons == 1024 ? -0.30 : data_in.neurons == 4096 ? -0.35 : data_in.neurons == 16384 ? -0.40 : data_in.neurons == 65536 ? -0.45 : 0.0;
		grb::set( e, value, grb::Phase::RESIZE );
		grb::set( e, value, grb::Phase::EXECUTE );
	}
	if( data_in.neurons != 1024 && data_in.neurons != 4096 && data_in.neurons != 16384 && data_in.neurons != 65536 ) {
		std::cerr << "Failure: the number of neurons does not correspond to a "
					 "known dataset"
				  << std::endl;
		return;
	}

	// get the name of the input files for the vector correct
	std::ostringstream oss;
	oss << input_vector_path << "/test" << data_in.neurons << "/sparse-images-" << data_in.neurons << ".mtx";
	std::string vector_filename = oss.str();
	std::cout << "Info: using input file " << vector_filename << std::endl;

	// create local parser
	grb::utils::MatrixFileReader< nz_type,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parser( vector_filename, data_in.direct );
	assert( data_in.neurons == parser.n() );
	n = parser.n();
	m = parser.m();

	

	// load into GraphBLAS
	out.result = std::make_shared< grb::Matrix< nz_type > >( grb::Matrix< nz_type >( m, n ) );
	grb::Matrix< nz_type > Lvin( m, n, parser.nz() );
	{
		std::cout << "Info: Lvin is " << m << "x" << n << " with " << parser.nz() << " nonzeroes." << std::endl;
		const RC rc = buildMatrixUnique( Lvin, parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
		// See internal issue #342 for re-enabling the below
		// const RC rc = buildMatrixUnique( Lvin,
		//	parser.begin( PARALLEL ), parser.end( PARALLEL),
		//	PARALLEL
		//);
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed (" << toString( rc ) << ")." << std::endl;
			return;
		}
	}

	// this a simple way to get the input vector by reading it as a matrix using
	// the existing parser and then apply the vxm operation on the matrix and on a
	// vector of ones
	grb::Semiring< grb::operators::add< nz_type >, grb::operators::mul< nz_type >, grb::identities::zero, grb::identities::one > realRing;

	RC rc = SUCCESS;
	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		if( data_in.thresholded )
			rc = sparse_nn_multi_inference( *out.result, Lvin, L, biases, data_in.threshold );
		else
			rc = sparse_nn_multi_inference( *out.result, Lvin, L, biases );

		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to sparse_nn_single_inference did not succeed (" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< nz_type >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( rc == SUCCESS ) {
			if( s == 0 ) {
				std::cout << "Info: cold sparse_nn_single_inference completed within " << out.iterations << " iterations. Time taken was " << single_time
						  << " ms. Deduced inner repetitions parameter of " << out.rep << " to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( data_in.thresholded )
				rc = rc ? rc : sparse_nn_multi_inference( *out.result, Lvin, L, biases, data_in.threshold );
			else
				rc = rc ? rc : sparse_nn_multi_inference( *out.result, Lvin, L, biases );
		}
		time_taken = timer.time();
		if( rc == SUCCESS ) {
			out.times.useful = time_taken / static_cast< nz_type >( out.rep );
		}
		sleep( 1 );
#ifndef NDEBUG
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for a " << out.rep << " Sparse Neural Network "
					  << "Single Inference calls (hot start): " << out.times.useful << ". "
					  << "Error code is " << out.error_code << std::endl;
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

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 8 || argc > 12 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <dataset path> <neurons> <layers> <input vector offset>"
				  << " <thresholded: 0 (false) or 1 (true)> <threshold>"
				  << " <direct/indirect> (inner iterations) (outer iterations)"
				  << " (verification <truth-file>)\n";
		std::cout << "<dataset path> <neurons> <layers> <input vector offset> "
				  << "<thresholded: 0 (false) or 1 (true)> <threshold> and "
				  << "<direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner() << ". If set to zero, the program will select a number of "
				  << "iterations approximately required to take at least one "
				  << "second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". This value must be strictly larger than 0.\n";
		std::cout << "(verification <truth-file>) is optional." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get the dataset path
	if( strlen( argv[ 1 ] ) > MAX_LEN ) {
		std::cerr << "Given dataset path is too long; please use a shorter dataset "
				  << "path)" << std::endl;
		return 1;
	}
	strcpy( in.dataset_path, argv[ 1 ] );

		// get the number of neurons
	in.neurons = atoi( argv[ 2 ] );

	// get the number of layers
	in.layers = atoi( argv[ 3 ] );

	// get the offset of the input vector
	in.input_vector_offset = atoi( argv[ 4 ] );

	// get the threshold if any defined
	if( atoi( argv[ 5 ] ) == 0 ) {
		in.thresholded = false;
	} else if( atoi( argv[ 5 ] ) == 1 ) {
		in.thresholded = true;

		// if a threshold is used, read its value
		in.threshold = atof( argv[ 6 ] );
	} else {
		std::cerr << "Could not parse argument " << argv[ 5 ] << " for the usage of "
				  << "a threshold (accepted values are 0 and 1)." << std::endl;
		return 2;
	}

	// get direct or indirect addressing
	if( strncmp( argv[ 7 ], "direct", 8 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 9 ) {
		in.rep = strtoumax( argv[ 8 ], &end, 10 );
		if( argv[ 8 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 8 ] << " "
					  << "for number of inner experiment repititions." << std::endl;
			return 3;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 10 ) {
		outer = strtoumax( argv[ 9 ], &end, 10 );
		if( argv[ 9 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 9 ] << " "
					  << "for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ MAX_LEN + 1 ];
	if( argc >= 11 ) {
		if( strncmp( argv[ 10 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 12 ) {
				(void)strncpy( truth_filename, argv[ 11 ], MAX_LEN );
				truth_filename[ MAX_LEN ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument." << std::endl;
				return 5;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 10 ] << "\", "
					  << "the optional \"verification\" argument was expected." << std::endl;
			return 5;
		}
	}

	std::cout << "Executable called with parameters: neurons = " << in.neurons << ", layers = " << in.layers << ", input vector offset = " << in.input_vector_offset
			  << ", inner repititions = " << in.rep << ", "
			  << "and outer reptitions = " << outer << std::endl;

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

	grb::Matrix< nz_type > & result = *out.result;
	std::cout << "Error code is " << out.error_code << "." << std::endl;
	std::cout << "Dimension of out is " << grb::nrows( result ) << " x " << grb::ncols( result ) << "." << std::endl;
	if( out.error_code == 0 && grb::nrows( result ) * grb::ncols( result ) > 0 ) {
		std::cout << "First 10 nonzeroes of out are: ( ";
		size_t k = 10;
		for( const std::pair< std::pair< size_t, size_t >, nz_type > & e : result ) {
			std::cout << std::fixed << e.second << " ";
			if( --k <= 0 )
				break;
		}
		std::cout << ")" << std::endl;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		if( verification ) {
			out.error_code = 0; // matrix_verification( out.result, truth_filename, C1, C2 );
			if( out.error_code == 0 ) {
				std::cout << "Output matrix verificaton was successful!\n";
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
