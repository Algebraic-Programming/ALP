
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

#include <chrono>
#include <limits>
#include <vector>
#include <iostream>
#include <exception>

#include <numa.h>
#include <inttypes.h>

#include <solver.h>

#include <graphblas/config.hpp>
#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/timerResults.hpp>


/** Parser type */
typedef grb::utils::MatrixFileReader< double, int > Parser;

/** Nonzero type */
typedef grb::internal::NonzeroStorage< int, int, double > NonzeroT;

/** In-memory storage type */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>
> Storage;

/** The default number of maximum iterations. */
constexpr const size_t max_iters = 10000;

struct input {
	char filename[ 1024 ];
	bool direct;
	bool jacobi_precond;
	size_t rep;
	size_t solver_iterations;
};

struct output {
	int error_code;
	size_t rep;
	size_t iterations;
	double residual;
	grb::utils::TimerResults times;
	double * pinnedVector;
	size_t pinnedVector_size;
};

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;
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
			auto it = parser.begin( grb::SEQUENTIAL );
			it != parser.end( grb::SEQUENTIAL );
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

struct jacobi_preconditioner_data {
	double * invDiag;
	size_t n;
};

int jacobi_preconditioner(
	double * const out, const double * const in,
	void * const data_p
) {
	struct jacobi_preconditioner_data * const data =
		static_cast< struct jacobi_preconditioner_data * >( data_p );
	#pragma omp parallel
	{
		const size_t n = data->n;
		const double * const invDiag = data->invDiag;
		#pragma omp for schedule(static,8)
		for( size_t i = 0; i < n; ++i ) {
			out[ i ] = invDiag[ i ] * in[ i ];
		}
	}
	return 0;
}

void test_tp( const struct input &data_in, struct output &out ) {

	// initialise outputs, assume successful run
	out.error_code = 0;
	out.pinnedVector = nullptr;
	out.pinnedVector_size = 0;

	// initialise timer
	grb::utils::Timer timer;
	timer.reset();

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << "Error: no file name given as input." << std::endl;
		out.error_code = 10;
		return;
	}

	// load into CRS
	const size_t n = Storage::getData().first.first;
	double * A_vals = nullptr;
	int * A_cind = nullptr;
	int * A_offs = nullptr;
	double * jacobiPreconditioner = nullptr;
	{
		const auto &data = Storage::getData().second;
		const size_t nnz = data.size();
		bool parse_error = false;
		A_vals = static_cast< double * >(
			numa_alloc_interleaved( sizeof( double ) * nnz ) );
		A_cind = static_cast< int * >(
			numa_alloc_interleaved( sizeof( int ) * nnz ) );
		A_offs = static_cast< int * >(
			numa_alloc_interleaved( sizeof( int ) * ( n + 1 ) ) );
		if( A_vals == nullptr || A_cind == nullptr || A_offs == nullptr ) {
			std::cerr << "Error: CRS allocation failed\n";
			out.error_code = 10;
			return;
		}
		if( data_in.jacobi_precond ) {
			jacobiPreconditioner = static_cast< double * >(
				numa_alloc_interleaved( sizeof( double ) * n ) );
			if( jacobiPreconditioner == nullptr ) {
				std::cerr << "Error: preconditioner allocation failed\n";
				out.error_code = 15;
				return;
			}
			for( size_t i = 0; i < n; ++i ) {
				jacobiPreconditioner[ i ] = 0;
			}
		}
		for( size_t i = 0; i <= n; ++i ) {
			A_offs[ i ] = 0;
		}
		for( const auto &nonzero : data ) {
			assert( nonzero.i() >= 0 );
			assert( nonzero.j() >= 0 );
			const size_t i = static_cast< size_t >( nonzero.i() );
			const size_t j = static_cast< size_t >( nonzero.j() );
			if( i > n ) {
				std::cerr << "Error: nonzero with row index " << i << " out of "
					<< "range ( " << n << " )\n";
				parse_error = true;
			} else {
				(void) ++(A_offs[ i ]);
			}
			if( j > n ) {
				std::cerr << "Error: nonzero with column index " << j << " out "
					<< "range ( " << n << " )\n";
				parse_error = true;
			}
		}
		for( size_t i = 1; i <= n; ++i ) {
			A_offs[ i ] += A_offs[ i - 1 ];
		}
		if( parse_error ) {
			std::cerr << "Error: invalid data detected\n";
			out.error_code = 20;
		} else if( static_cast< size_t >(A_offs[ n ]) != nnz ) {
			std::cerr << "Error: ingestion checksum failed\n";
			out.error_code = 30;
		}
		if( out.error_code ) { return; }
		for( const auto &nonzero : data ) {
			const size_t i = static_cast< size_t >( nonzero.i() );
			const size_t j = static_cast< size_t >( nonzero.j() );
			const size_t index = --(A_offs[ i + 1 ]);
			A_vals[ index ] = nonzero.v();
			A_cind[ index ] = nonzero.j();
			if( data_in.jacobi_precond && j == i ) {
				jacobiPreconditioner[ i ] = static_cast< double >( 1 ) / nonzero.v();
			}
		}
		for( size_t i = 1; i <= n; ++i ) {
			A_offs[ i - 1 ] = A_offs[ i ];
			if( data_in.jacobi_precond ) {
				if( jacobiPreconditioner[ i - 1 ] == 0 ) {
					std::cerr << "Error: invalid preconditioner detected (zero at index "
						<< (i-1) << ")\n";
					out.error_code = 35;
					return;
				}
			}
		}
		A_offs[ n ] = nnz;
	}

	// I/O done
	out.times.io = timer.time();
	timer.reset();

	// set up default CG test
	sparse_cg_handle_t cg_handle;
	double * x, * b;
	x = b = nullptr;
	int err = sparse_cg_init_dii( &cg_handle, n, A_vals, A_cind, A_offs );
	err = err ? err :
		sparse_cg_set_max_iter_count_dii( cg_handle, data_in.solver_iterations );
	x = static_cast< double * >(
		numa_alloc_interleaved( sizeof(double) * n ) );
	b = static_cast< double * >(
		numa_alloc_interleaved( sizeof(double) * n ) );
	if( err != 0 || x == nullptr || b == nullptr ) {
		std::cerr << "Setting up CG solver failed\n";
		if( err == 0 ) {
			(void) sparse_cg_destroy_dii( cg_handle );
		}
		out.error_code = 40;
		return;
	}
	for( size_t i = 0; i < n; ++i ) {
		x[ i ] = static_cast< double >( 1 ) / static_cast< double >( n );
		b[ i ] = static_cast< double >( 1 );
	}
	struct jacobi_preconditioner_data * jPData = nullptr;
	if( data_in.jacobi_precond ) {
		jPData = new struct jacobi_preconditioner_data;
		jPData->n = n;
		jPData->invDiag = jacobiPreconditioner;
		err = sparse_cg_set_preconditioner_dii( cg_handle, &jacobi_preconditioner,
			jPData );
		if( err != NO_ERROR ) {
			std::cerr << "Error: set up of preconditioner failed with code " << err
				<< "\n";
			out.error_code = 45;
			return;
		}
	}

	// setup done
	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		err = sparse_cg_solve_dii( cg_handle, x, b );
		double single_time = timer.time();
		if( !(err == NO_ERROR || err == FAILED) ) {
			std::cerr << "Failure: call to conjugate_gradient did not succeed ("
				<< err << ")." << std::endl;
			out.error_code = 50;
			return;
		}
		if( err == FAILED ) {
			std::cout << "Warning: call to conjugate_gradient did not converge\n";
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( err == NO_ERROR || err == FAILED ) {
			if( err == FAILED ) {
				std::cout << "Info: cold conjugate_gradient did not converge within ";
			} else {
				std::cout << "Info: cold conjugate_gradient completed within ";
			}
			err = sparse_cg_get_iter_count_dii( cg_handle, &(out.iterations) );
			err = err ? err : sparse_cg_get_residual_dii( cg_handle, &(out.residual) );
			if( err != NO_ERROR ) {
				std::cerr << "Error: could not retrieve solver statistics\n";
				out.error_code = 60;
				return;
			}
			std::cout << out.iterations << " iterations. Last computed residual is "
				<< out.residual << ". Time taken was " << single_time << " ms. "
				<< "Deduced inner repetitions parameter of " << out.rep << " "
				<< "to take 1 second or more per inner benchmark.\n";
		}
	} else {
		// do benchmark
		timer.reset();
		for( size_t i = 0; i < out.rep && err == NO_ERROR; ++i ) {
			#pragma omp parallel for schedule(static,8)
			for( size_t i = 0; i < n; ++i ) {
				x[ i ] = static_cast< double >( 1 ) / static_cast< double >( n );
			}
			err = sparse_cg_solve_dii( cg_handle, x, b );
		}
		const double time_taken = timer.time();
		out.times.useful = time_taken / static_cast< double >( out.rep );
		if( !(err == NO_ERROR || err == FAILED) ) {
			std::cerr << "Error: during call to CG solver (code " << err << ")\n";
			out.error_code = 70 + err;
			return;
		}

		if( err == FAILED ) {
			std::cerr << "Warning: CG solver did not converge\n";
		}
		err = sparse_cg_get_iter_count_dii( cg_handle, &(out.iterations) );
		err = err ? err : sparse_cg_get_residual_dii( cg_handle, &(out.residual) );
		if( err != NO_ERROR ) {
			std::cerr << "Error: could not get statistics from solver handle (code "
				<< err << ")\n";
			out.error_code = 80 + err;
			return;
		}
		std::cout << "Time taken for " << out.rep << " Conjugate Gradients calls "
			<< "(hot start): " << out.times.useful << ". Error code is " << err
			<< std::endl;
		std::cout << "\tnumber of CG iterations: " << out.iterations << "\n";
		std::cout << "\tCG residual: " << out.residual << "\n";
		std::cout << "\tmilliseconds per iteration: "
			<< ( out.times.useful / static_cast< double >( out.iterations ) )
			<< "\n";
		sleep( 1 );
	}

	// start postamble
	timer.reset();

	// output
	out.pinnedVector = x;
	out.pinnedVector_size = n;
	{
		const size_t nnz = Storage::getData().second.size();
		numa_free( b, sizeof(double) * n );
		numa_free( A_vals, sizeof(double) * nnz );
		numa_free( A_cind, sizeof(int) * nnz );
		numa_free( A_offs, sizeof(int) * (n + 1) );
	}
	if( data_in.jacobi_precond ) {
		numa_free( jacobiPreconditioner, sizeof(double) * n );
		assert( jPData );
		delete jPData;
	}

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ]
			<< " <dataset> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (solver iterations) (Jacobi)\n";
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
		std::cout << "(Jacobi) is an optional boolean value, with default false. "
			<< "The only possible other value is true, which, if sets, will apply "
			<< "Jacobi preconditioning to the CG solve." << std::endl;
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

	in.jacobi_precond = false;
	if( argc >= 7 ) {
		if( strncmp( argv[ 6 ], "true", 5 ) == 0 ) {
			in.jacobi_precond = true;
		} else if( strncmp( argv[ 6 ], "false", 6 ) != 0 ) {
			std::cerr << "Could not parse argument " << argv[ 6 ] << ", for whether "
				<< "Jacobi preconditioning should be enabled (expected true or false).\n";
			return 55;
		}
	}

	std::cout << "Executable called with parameters " << in.filename << ", "
		<< "inner repititions = " << in.rep << ", "
		<< "outer reptitions = " << outer << ", "
		<< "solver iterations = " << in.solver_iterations << ", and "
		<< "Jacobi preconditioning = " << in.jacobi_precond << "."
		<< std::endl;

	// launch I/O
	bool success = false;
	ioProgram( in, success );
	if( !success ) {
		std::cerr << "I/O program caught an exception\n";
		return 75;
	}

	// the output struct
	struct output out;

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		test_tp( in, out );
		if( out.error_code ) {
			std::cerr << "Test returns non-SUCCESS error code " << out.error_code
				<< "\n";
			return 80;
		}
		in.rep = out.rep;
	}

	// launch benchmark
	grb::utils::TimerResults total_times, min_times, max_times;
	std::vector< grb::utils::TimerResults > sdev_times( outer );
	total_times.set( 0 );
	min_times.set( std::numeric_limits< double >::infinity() );
	max_times.set( 0 );
	for( size_t i = 0; i < outer; ++i ) {
		test_tp( in, out );
		if( out.error_code ) {
			std::cerr << "Test returns non-SUCCESS error code " << out.error_code
				<< "\n";
			return 90;
		}
		// compute aggregate timings
		total_times.accum( out.times );
		min_times.min( out.times );
		max_times.max( out.times );
		sdev_times[ i ] = out.times;
	}
	std::cout << "Benchmark completed successfully and took " << out.iterations
		<< " iterations and achieved residual " << out.residual << ".\n";

	// compute timing statistics
	grb::utils::TimerResults sdev;
	total_times.normalize( outer );
	sdev.set( 0 );
	for( size_t i = 0; i < outer; ++i ) {
		double diff = sdev_times[ i ].io - total_times.io;
		sdev.io += diff * diff;
		diff = sdev_times[ i ].preamble - total_times.preamble;
		sdev.preamble += diff * diff;
		diff = sdev_times[ i ].useful - total_times.useful;
		sdev.useful += diff * diff;
		diff = sdev_times[ i ].postamble - total_times.postamble;
		sdev.postamble += diff * diff;
	}
	sdev.normalize( outer - 1 );

	// output checksums
	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of x is " << out.pinnedVector_size << ".\n";
	if( out.error_code == 0 && out.pinnedVector_size > 0 ) {
		std::cout << "First 10 nonzeroes of x are: ( ";
		for( size_t k = 0; k < out.pinnedVector_size && k < 10; ++k ) {
			std::cout << out.pinnedVector[ k ] << " ";
		}
		std::cout << ")" << std::endl;
	}

	// output timing statistics
	std::cout << "Overall timings (io, preambe, useful, postamble):\n"
		<< std::scientific;
	std::cout << "Avg: " << total_times.io << ", " << total_times.preamble
		<< ", " << total_times.useful << ", " << total_times.postamble << "\n";
	std::cout << "Min: " << min_times.io << ", " << min_times.preamble << ", "
		<< min_times.useful << ", " << min_times.postamble << "\n";
	std::cout << "Max: " << max_times.io << ", " << max_times.preamble << ", "
		<< max_times.useful << ", " << max_times.postamble << "\n";
	std::cout << "Std: " << sqrt( sdev.io ) << ", " << sqrt( sdev.preamble )
		<< ", " << sqrt( sdev.useful ) << ", " << sqrt( sdev.postamble ) << "\n";
 #if __GNUC__ > 4
	std::cout << std::defaultfloat;
 #endif
	std::cout << "Time since epoch (in ms.): " << std::chrono::duration_cast<
			std::chrono::milliseconds
		>( std::chrono::system_clock::now().time_since_epoch() ).count() << "\n";

	// TODO re-enable verification

	// done
	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}

