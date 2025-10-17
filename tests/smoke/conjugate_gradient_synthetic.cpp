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
#include <string>

#ifdef _CG_COMPLEX
 #include <complex>
#endif

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/algorithms/conjugate_gradient.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>
#include <utils/matrix_generators.hpp>

#include <utils/output_verification.hpp>


using namespace grb;
using namespace algorithms;

using BaseScalarType = double;
#ifdef _CG_COMPLEX
 using ScalarType = std::complex< BaseScalarType >;
#else
 using ScalarType = BaseScalarType;
#endif

/** Parser type */
typedef grb::utils::MatrixFileReader<
	ScalarType,
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
	ScalarType
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

constexpr const BaseScalarType tol = 0.000001;

/** The default number of maximum iterations. */
constexpr const size_t max_iters = 100;

constexpr const double c1 = 0.0001;
constexpr const double c2 = 0.0001;

struct input {
	char filename[ 1024 ];
	bool direct;
	bool jacobi_precond;
	size_t rep;
	size_t solver_iterations;
	bool synthetic_mode;
	size_t synthetic_size;
	size_t synthetic_band;
};

struct output {
	int error_code;
	size_t rep;
	size_t iterations;
	double residual;
	grb::utils::TimerResults times;
	PinnedVector< ScalarType > pinnedVector;
};

/**
 * Generate a banded matrix with compile-time band width using BandIterator
 */
template<size_t BAND_WIDTH>
grb::Matrix< ScalarType > 
generateBandedMatrixTemplate(size_t matrix_size) {
	// Create matrix
	grb::Matrix< ScalarType > A(matrix_size, matrix_size);
	
	// Use BandIterator to build the matrix
	grb::RC rc = buildMatrixUnique(
		A,
		grb::utils::BandIterator< BAND_WIDTH, false >::make_begin(matrix_size),
		grb::utils::BandIterator< BAND_WIDTH, false >::make_end(matrix_size),
		SEQUENTIAL
	);
	
	if (rc != grb::SUCCESS) {
		throw std::runtime_error("Failed to build banded matrix: " + grb::toString(rc));
	}
	
	return A;
}

/**
 * Generate a banded matrix with runtime band width (1, 3, or 5)
 */
grb::Matrix< ScalarType > 
generateBandedMatrixRuntime(size_t matrix_size, size_t band_width) {
	switch (band_width) {
		case 1:
			return generateBandedMatrixTemplate<1>(matrix_size);
		case 3:
			return generateBandedMatrixTemplate<3>(matrix_size);
		case 5:
			return generateBandedMatrixTemplate<5>(matrix_size);
		default:
			throw std::runtime_error("Unsupported band width: " + std::to_string(band_width) + 
				". Supported values are 1, 3, 5");
	}
}

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;
	
	// Generate matrix BEFORE tracing starts
	Storage::getData().first.first = data_in.synthetic_size;
	
	try {
		// Generate synthetic banded matrix using ALP BandIterator
		// This happens in ioProgram() before tracing starts, so no overhead
		grb::Matrix< ScalarType > L = generateBandedMatrixRuntime(data_in.synthetic_size, data_in.synthetic_band);
		
		// Store matrix data by converting to vector of NonzeroT
		Storage::getData().first.second = nnz(L);
		auto &data = Storage::getData().second;
		data.clear();
		
		// Convert matrix to vector of NonzeroT objects
		for (auto it = L.cbegin(); it != L.cend(); ++it) {
			data.push_back(NonzeroT(*it));
		}
		
		std::cout << "Generated synthetic banded matrix: " << data_in.synthetic_size << "x" << data_in.synthetic_size 
				  << " with band width " << data_in.synthetic_band << " and " << Storage::getData().first.second << " non-zeros" << std::endl;
		
		success = true;
		return;
	} catch (std::exception &e) {
		std::cerr << "Synthetic matrix generation failed: " << e.what() << std::endl;
		return;
	}
}

void grbProgram( const struct input &data_in, struct output &out ) {

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// sanity checks on input (synthetic mode always has valid parameters)

	// assume successful run
	out.error_code = 0;

	// load into GraphBLAS
	const size_t n = Storage::getData().first.first;
	Matrix< ScalarType > L( n, n );
	Vector< ScalarType > diag = data_in.jacobi_precond
		? Vector< ScalarType >( n )
		: Vector< ScalarType >( 0 );
		
	// Load matrix from stored data
	const auto &data = Storage::getData().second;
	RC io_rc = buildMatrixUnique(
		L,
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, ScalarType
		>( data.cbegin() ),
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, ScalarType
		>( data.cend() ),
		SEQUENTIAL
	);
	io_rc = io_rc ? io_rc : wait();
	if( io_rc != SUCCESS ) {
		std::cerr << "Failure: call to buildMatrixUnique did not succeed "
			<< "(" << toString( io_rc ) << ")." << std::endl;
		out.error_code = 1;
		return;
	}
	
	if( data_in.jacobi_precond ) {
		assert( true ); // Always true for synthetic mode
		RC io_rc = grb::set( diag, 0 );
		io_rc = io_rc ? io_rc :
			eWiseLambda( [&diag,&L](
					const size_t i, const size_t j, ScalarType &v
				) {
					if( i == j ) {
						diag[ i ] = utils::is_complex< ScalarType >::inverse( v );
					}
				}, L, diag
			);
		io_rc = io_rc ? io_rc : wait();
		if( io_rc != SUCCESS ) {
			std::cerr << "Failure: extracting diagonal did not succeed ("
				<< toString( io_rc ) << ").\n";
			out.error_code = 10;
			return;
		}
	}

	// check number of nonzeroes
	const size_t global_nnz = nnz( L );
	const size_t parser_nnz = Storage::getData().first.second;
	if( global_nnz != parser_nnz ) {
		std::cerr << "Warning: global nnz (" << global_nnz << ") does not equal "
			<< "parser nnz (" << parser_nnz << "). This could naturally occur if the "
			<< "input file employs symmetric storage, in which case only roughly one "
			<< "half of the input is stored.\n";
	}

	// I/O done
	out.times.io = timer.time();
	timer.reset();

	// set up default CG test
	Vector< ScalarType > x( n ), b( n ), r( n ), u( n ), temp( n );
	Vector< ScalarType > optional_temp = data_in.jacobi_precond
		? Vector< ScalarType >( n )
		: Vector< ScalarType >( 0 );
	std::function< RC( Vector< ScalarType > &, const Vector< ScalarType > & ) >
		jacobi_preconditioner =
			[&diag](grb::Vector< ScalarType > &out, const grb::Vector< ScalarType > &in) {
					return grb::eWiseApply< descriptors::dense >( out, in, diag,
						grb::operators::mul< ScalarType >() );
			};

	// Set up default vectors (same as original)
	RC rc = set( x,
		static_cast< ScalarType >( 1 ) / static_cast< ScalarType >( n ) );
	rc = rc ? rc : set( b, static_cast< ScalarType >( 1 ) );
	rc = rc ? rc : wait();
	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		if( data_in.jacobi_precond ) {
			rc = preconditioned_conjugate_gradient(
				x, L, b,
				jacobi_preconditioner,
				data_in.solver_iterations, tol,
				out.iterations, out.residual,
				r, u, temp, optional_temp
			);
		} else {
			rc = conjugate_gradient(
				x, L, b,
				data_in.solver_iterations, tol,
				out.iterations, out.residual,
				r, u, temp
			);
		}
		rc = rc ? rc : wait();
		double single_time = timer.time();
		if( !(rc == SUCCESS || rc == FAILED) ) {
			std::cerr << "Failure: call to conjugate_gradient did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == FAILED ) {
			std::cout << "Warning: call to conjugate_gradient did not converge\n";
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
					std::cout << "Info: cold conjugate_gradient did not converge within ";
				} else {
					std::cout << "Info: cold conjugate_gradient completed within ";
				}
				std::cout << out.iterations << " iterations. Last computed residual is "
					<< out.residual << ". Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {

			rc = set( x, static_cast< ScalarType >( 1 )
					  / static_cast< ScalarType >( n ) );

			if( rc == SUCCESS ) {
				if( data_in.jacobi_precond ) {
					rc = preconditioned_conjugate_gradient(
						x, L, b,
						jacobi_preconditioner,
						data_in.solver_iterations, tol,
						out.iterations, out.residual,
						r, u, temp, optional_temp
					);
				} else {
					rc = conjugate_gradient(
						x, L, b,
						data_in.solver_iterations, tol,
						out.iterations, out.residual,
						r, u, temp
					);
				}
			}
			if( Properties<>::isNonblockingExecution ) {
				rc = rc ? rc : wait();
			}
		}
		const double time_taken = timer.time();
		out.times.useful = time_taken / static_cast< double >( out.rep );
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for " << out.rep << " "
				<< "Conjugate Gradients calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << grb::toString( rc ) << std::endl;
			std::cout << "\tnumber of CG iterations: " << out.iterations << "\n";
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
	out.pinnedVector = PinnedVector< ScalarType >( x, SEQUENTIAL );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 9 ) {
		std::cout << "Usage: " << argv[ 0 ]
			<< " <size> <band_width> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (solver iterations) (Jacobi) "
			<< "(verification <truth-file>)\n";
		std::cout << "<size> and <band_width> and <direct/indirect> are mandatory arguments.\n";
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
		std::cout << "(Jacobi) is an optional boolean value, with default false. "
			<< "The only possible other value is true, which, if sets, will apply "
			<< "Jacobi preconditioning to the CG solve.\n";
		std::cout << "(verification <truth-file>) is optional." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	in.synthetic_mode = true;  // Always synthetic mode
	in.filename[ 0 ] = '\0';   // No filename needed

	// get synthetic parameters
	in.synthetic_size = strtoumax( argv[ 1 ], nullptr, 10 );
	in.synthetic_band = strtoumax( argv[ 2 ], nullptr, 10 );

	// get direct or indirect addressing
	if( strncmp( argv[ 3 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 5 ) {
		in.rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 5 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			return 40;
		}
	}

	in.solver_iterations = max_iters;
	if( argc >= 7 ) {
		in.solver_iterations = strtoumax( argv[ 6 ], &end, 10 );
		if( argv[ 6 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 6 ] << " "
				<< "for the maximum number of solver iterations." << std::endl;
			return 50;
		}
	}

	in.jacobi_precond = false;
	if( argc >= 8 ) {
		if( strncmp( argv[ 7 ], "true", 5 ) == 0 ) {
			in.jacobi_precond = true;
		} else if( strncmp( argv[ 7 ], "false", 6 ) != 0 ) {
			std::cerr << "Could not parse argument " << argv[ 7 ] << ", for whether "
				<< "Jacobi preconditioning should be enabled (expected true or false).\n";
			return 55;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ 1024 ];
	if( argc >= 9 ) {
		if( strncmp( argv[ 8 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 10 ) {
				(void) strncpy( truth_filename, argv[ 9 ], 1023 );
				truth_filename[ 1023 ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument."
					<< std::endl;
				return 60;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 8 ] << "\", "
				<< "the optional \"verification\" argument was expected." << std::endl;
			return 70;
		}
	}

	std::cout << "Executable called with synthetic parameters: size = " << in.synthetic_size 
			  << ", band_width = " << in.synthetic_band << ", "
			  << "inner repititions = " << in.rep << ", "
			  << "outer reptitions = " << outer << ", "
			  << "solver iterations = " << in.solver_iterations << ", and "
			  << "Jacobi preconditioning = " << in.jacobi_precond << "."
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
			const ScalarType &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
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