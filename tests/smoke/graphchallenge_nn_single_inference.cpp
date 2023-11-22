
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

#include <graphblas/algorithms/sparse_nn_single_inference.hpp>

#include <graphblas/nonzeroStorage.hpp>

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
typedef grb::internal::NonzeroStorage<
	grb::config::RowIndexType,
	grb::config::ColIndexType,
	double
> NonzeroT;

/** In-memory storage type */
typedef grb::utils::Singleton<
	std::pair<
		// biases
		std::vector< double >,
		// for each input file, stores the number of nonzeroes and the nonzeroes
		std::vector< std::pair< size_t, std::vector< NonzeroT > > >
	>
> Storage;

constexpr const double c1 = 0.0001;
constexpr const double c2 = 0.0002;
constexpr const size_t max_len = 1000;

struct input {
	char dataset_path[ max_len + 1 ];
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
	PinnedVector< double > pinnedVector;
};

void ioProgram( const struct input &data_in, int &rc ) {
	rc = 0;
	char weights_path[ max_len + 1 ];
	if( strlen( data_in.dataset_path ) + strlen( "/WEIGHTS-HPEC" ) > max_len ) {
		std::cerr << "Failure: given dataset path is too long (please use a "
			"shorter dataset path)" << std::endl;
		rc = 10;
		return;
	}
	strcpy( weights_path, data_in.dataset_path );
	strcat( weights_path, "/WEIGHTS-HPEC" );

	char input_vector_path[ max_len + 1 ];
	if( strlen( data_in.dataset_path ) + strlen( "/MNIST-HPEC" ) > max_len ) {
		std::cerr << "Failure: given dataset path is too long (please use a "
			<< "shorter dataset path)" << std::endl;
		rc = 20;
		return;
	}
	strcpy( input_vector_path, data_in.dataset_path );
	strcat( input_vector_path, "/MNIST-HPEC" );

	std::vector< double > &biases = Storage::getData().first;

	if( data_in.neurons == 1024 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases.push_back( -0.30 );
		}
	} else if( data_in.neurons == 4096 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases.push_back( -0.35 );
		}
	} else if( data_in.neurons == 16384 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases.push_back( -0.40 );
		}
	} else if( data_in.neurons == 65536 ) {
		for( size_t i = 0; i < data_in.layers; i++ ) {
			biases.push_back( -0.45 );
		}
	} else {
		std::cerr << "Failure: the number of neurons does not correspond to a "
			"known dataset" << std::endl;
		rc = 30;
		return;
	}

	for( size_t i = 0; i < data_in.layers; i++ ) {
		// get the names of the input files for all layers correct
		std::ostringstream oss;
		oss << weights_path << "/neuron" << data_in.neurons << "/n"
			<< data_in.neurons << "-l" << i + 1 << ".mtx";
		std::string filename = oss.str();

		// create local parser
		Parser parser( filename.c_str(), data_in.direct );
		assert( parser.m() == parser.n() );
		assert( data_in.neurons == parser.n() );
		if( parser.m() != parser.n() ) {
			std::cerr << "Error: expected input file to be square\n";
			rc = 40;
			return;
		}
		if( data_in.neurons != parser.n() ) {
			std::cerr << "Error: expected input matrix to match #neurons\n";
			rc = 50;
			return;
		}

		// preload file into storage
		std::pair< size_t, std::vector< NonzeroT > > fileContents;
		{
			try{
				fileContents.first = parser.nz();
			} catch( ... ) {
				fileContents.first = parser.entries();
			}
			for(
				auto it = parser.cbegin( SEQUENTIAL );
				it != parser.cend( SEQUENTIAL );
				++it
			) {
				fileContents.second.push_back( NonzeroT( *it ) );
			}
			// See internal issue #342 for re-enabling the below
			/*for(
				auto it = parser.cbegin( PARALLEL );
				it != parser.cend( PARALLEL );
				++it
			) {
				fileContents.second.push_back( NonzeroT( *it ) );
			}*/
		}
		Storage::getData().second.push_back( fileContents );
	}

	// get the name of the input files for the vector correct
	std::ostringstream oss;
	oss << input_vector_path << "/test" << data_in.neurons << "/sparse-images-"
		<< data_in.neurons << "_" << data_in.input_vector_offset << ".mtx";
	std::string vector_filename = oss.str();

	// create local parser
	Parser parser( vector_filename, data_in.direct );
	assert( data_in.neurons == parser.n() );
	if( data_in.neurons != parser.n() ) {
		std::cerr << "Error: expected input vector size to match #neurons\n";
		rc = 60;
		return;
	}
	// preload vector into storage
	// note that we here read vector data by reusing the more general matrix parser
	std::pair< size_t, std::vector< NonzeroT > > fileContents;
	{
		try{
			fileContents.first = parser.nz();
		} catch( ... ) {
			fileContents.first = parser.entries();
		}
		for(
			auto it = parser.cbegin( SEQUENTIAL );
			it != parser.cend( SEQUENTIAL );
			++it
		) {
			fileContents.second.push_back( NonzeroT( *it ) );
		}
		// See internal issue #342 for re-enabling the below
		/*for(
			auto it = parser.cbegin( PARALLEL );
			it != parser.cend( PARALLEL );
			++it
		) {
			fileContents.second.push_back( NonzeroT( *it ) );
		}*/
	}
	Storage::getData().second.push_back( fileContents );
	if( Storage::getData().second.size() != data_in.layers + 1 ) {
		std::cerr << "Error: expected " << (data_in.layers + 1) << " matrices would be "
			<< "parsed, got " << Storage::getData().second.size() << " instead\n";
		rc = 70;
		return;
	}
	std::cout << "Info: I/O subprogram cached " << Storage::getData().second.size()
		<< " files.\n";
}

void grbProgram( const struct input &data_in, struct output &out ) {

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// assume successful run
	out.error_code = 0;

	out.times.io = timer.time();
	timer.reset();

	std::vector< double > &biases = Storage::getData().first;
	std::vector< grb::Matrix< double > > L;

	// load into GraphBLAS
	const size_t n = data_in.neurons;
	for( size_t i = 0; i < data_in.layers; i++ ) {
		L.push_back( grb::Matrix< double >( n, n ) );
		assert( Storage::getData().second.size() == data_in.layers + 1 );
		const size_t parser_nz = (Storage::getData().second)[ i ].first;
		const auto &data = (Storage::getData().second)[ i ].second;
		const RC rc = buildMatrixUnique(
			L[ i ],
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			SEQUENTIAL
		);
		// See internal issue #342 for re-enabling the below
		//const RC rc = buildMatrixUnique(
		//	L[ i ],
		//	utils::makeNonzeroIterator<
		//		grb::config::RowIndexType, grb::config::ColIndexType, double
		//	>( data.cbegin() ),
		//	utils::makeNonzeroIterator<
		//		grb::config::RowIndexType, grb::config::ColIndexType, double
		//	>( data.cend() ),
		//	PARALLEL
		//);
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 5;
			return;
		}
		// check number of nonzeroes
		const size_t global_nnz = nnz( L[ i ] );
		if( global_nnz != parser_nz ) {
			std::cerr << "Failure: ALP/GraphBLAS matrix nnz (" << global_nnz << ") "
				<< "does not equal parser nnz (" << parser_nz << ")!\n";
			out.error_code = 10;
			return;
		}
	}

	// this a simple way to get the input vector by reading it as a matrix using
	// the existing parser and then apply the vxm operation on the matrix and on a
	// vector of ones
	grb::Vector< double > vout( n ), vin( n ), temp( n );
	{
		grb::Matrix< double > Lvin( n, n );
		const size_t parser_nz = (Storage::getData().second)[ data_in.layers ].first;
		const auto &data = (Storage::getData().second)[ data_in.layers ].second;
		RC rc = buildMatrixUnique(
			Lvin,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			SEQUENTIAL
		);
		// See internal issue #342 for re-enabling the below
		//const RC rc = buildMatrixUnique(
		//	Lvin,
		//	utils::makeNonzeroIterator<
		//		grb::config::RowIndexType, grb::config::ColIndexType, double
		//	>( data.cbegin() ),
		//	utils::makeNonzeroIterator<
		//		grb::config::RowIndexType, grb::config::ColIndexType, double
		//	>( data.cend() ),
		//	PARALLEL
		//);
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			return;
		}
		// check number of nonzeroes
		const size_t global_nnz = nnz( Lvin );
		if( global_nnz != parser_nz ) {
			std::cerr << "Failure: ALP/GraphBLAS matrix nnz (" << global_nnz << ") "
				<< "does not equal parser nnz (" << parser_nz << ")!\n";
			out.error_code = 15;
			return;
		}

		// now we have the matrix form of the input vector, turn it into an actual
		// vector:
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> realRing;

		rc = grb::set( vout, 1.0 );
		assert( rc == SUCCESS );

		rc = rc ? rc : grb::clear( vin );
		assert( rc == SUCCESS );

		rc = rc ? rc : grb::vxm( vin, vout, Lvin, realRing );
		assert( rc == SUCCESS );

		rc = rc ? rc : grb::clear( vout );
		assert( rc == SUCCESS );

		if( rc != SUCCESS ) {
			std::cerr << "Error: could not convert the input vector format\n";
			out.error_code = 17;
			return;
		}
	}

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;

	// time a single call
	RC rc = SUCCESS;
	if( out.rep == 0 ) {
		timer.reset();
		if( data_in.thresholded ) {
			rc = sparse_nn_single_inference(
				vout, vin, L,
				biases, data_in.threshold,
				temp
			);
		} else {
			rc = sparse_nn_single_inference(
				vout, vin, L,
				biases,
				temp
			);
		}
		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to sparse_nn_single_inference did not succeed ("
				<< toString( rc ) << ")." << std::endl;
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
				std::cout << "Info: cold sparse_nn_single_inference completed within "
					<< out.iterations << " iterations. Time taken was "
					<< single_time << " ms. Deduced inner repetitions parameter of "
					<< out.rep << " to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( rc == SUCCESS ) {
				if( data_in.thresholded ) {
					rc = sparse_nn_single_inference(
						vout, vin, L,
						biases, data_in.threshold,
						temp
					);
				} else {
					rc = sparse_nn_single_inference(
						vout, vin, L,
						biases, temp
					);
				}
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

	// output
	out.pinnedVector = PinnedVector< double >( vout, SEQUENTIAL );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 8 || argc > 12 ) {
		std::cout << "Usage: " << argv[ 0 ]
			<< " <dataset path> <neurons> <layers> <input vector offset>"
			<< " <thresholded: 0 (false) or 1 (true)> <threshold>"
			<< " <direct/indirect> (inner iterations) (outer iterations)"
			<< " (verification <truth-file>)\n";
		std::cout << "<dataset path> <neurons> <layers> <input vector offset> "
			<< "<thresholded: 0 (false) or 1 (true)> <threshold> and "
			<< "<direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::inner()
			<< ". If set to zero, the program will select a number of "
			<< "iterations approximately required to take at least one "
			<< "second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::outer()
			<< ". This value must be strictly larger than 0.\n";
		std::cout << "(verification <truth-file>) is optional." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get the dataset path
	if( strlen( argv[ 1 ] ) > max_len ) {
		std::cerr << "Given dataset path is too long; please use a shorter dataset "
			<< "path)" << std::endl;
		return 10;
	}
	strcpy( in.dataset_path, argv[ 1 ] );

	// get the number of neurons
	in.neurons = atoi( argv[ 2 ] );

	// get the number of layers
	in.layers = atoi( argv[ 3 ] );

	// get the offset of the input vector
	in.input_vector_offset = atoi( argv[ 4 ] );

	// get the threshold if any defined
	if ( atoi( argv[ 5 ] ) == 0 ) {
		in.thresholded = false;
	} else if ( atoi( argv[ 5 ] ) == 1 ) {
		in.thresholded = true;

		// if a threshold is used, read its value
		in.threshold = atof( argv[ 6 ] );
	} else {
		std::cerr << "Could not parse argument " << argv[ 5 ] << " for the usage of "
			<< "a threshold (accepted values are 0 and 1)." << std::endl;
		return 20;
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
			return 30;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 10 ) {
		outer = strtoumax( argv[ 9 ], &end, 10 );
		if( argv[ 9 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 9 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			return 40;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ max_len + 1 ];
	if( argc >= 11 ) {
		if( strncmp( argv[ 10 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 12 ) {
				(void) strncpy( truth_filename, argv[ 11 ], max_len );
				truth_filename[ max_len ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument."
					<< std::endl;
				return 50;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 10 ] << "\", "
				<< "the optional \"verification\" argument was expected." << std::endl;
			return 60;
		}
	}

	std::cout << "Executable called with parameters: neurons = " << in.neurons
		<< ", layers = " << in.layers << ", input vector offset = "
		<< in.input_vector_offset << ", inner repititions = " << in.rep << ", "
		<< "and outer reptitions = " << outer << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// perform I/O
	{
		int error_code;
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &ioProgram, in, error_code, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec(I/O) returns with non-SUCCESS error code \""
				<< grb::toString( rc ) << "\"\n";
			return 73;
		}
		if( error_code != 0 ) {
			std::cerr << "I/O sub-program caught an error (code " << error_code << ")\n";
			return 77;
		}
	}

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
		std::cout << "Benchmark completed successfully.\n";
	}

	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of out is " << out.pinnedVector.size() << ".\n";
	if( out.error_code == 0 && out.pinnedVector.size() > 0 ) {
		std::cout << "First 10 nonzeroes of out are: ( ";
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
				out.pinnedVector, truth_filename, c1, c2
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
	if( out.error_code == 0 ) {
		return 0;
	} else {
		return (100 + out.error_code);
	}
}

