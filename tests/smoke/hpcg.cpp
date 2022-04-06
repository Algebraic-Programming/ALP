
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

/**
 * @file hpcg_test.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Test for HPCG simulations on N-dimensional physical problems.
 *
 * This test strictly follows the parameter and the formulation of the reference HPCG
 * benchmark impementation in https://github.com/hpcg-benchmark/hpcg.
 *
 * @date 2021-04-30
 */

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>

#include <graphblas.hpp>
#include <graphblas/algorithms/hpcg/hpcg.hpp>
#include <graphblas/algorithms/hpcg/system_building_utils.hpp>

#include <graphblas/algorithms/hpcg/old_ndim_matrix_builders.hpp>

#include <chrono>

// #define TEST_ITER

// here we define a custom macro and do not use NDEBUG since the latter is not defined for smoke tests
#ifdef HPCG_PRINT_STEPS

#include <cstdio>

// HPCG_PRINT_STEPS requires defining the following symbols

/**
 * @brief simply prints \p args on a dedicated line.
 */
#define DBG_println( args ) std::cout << args << std::endl;

// forward declaration for the tracing facility
template< typename T,
	class Ring = grb::Semiring< grb::operators::add< T >, grb::operators::mul< T >, grb::identities::zero, grb::identities::one >
>
void print_norm( const grb::Vector< T > &r, const char * head, const Ring &ring = Ring() );

/**
 * @brief prints \p head and the norm of \p r.
 */
#define DBG_print_norm( vec, head ) print_norm( vec, head )
#endif

#include <graphblas/utils/Timer.hpp>

#include <utils/argument_parser.hpp>
#include <utils/assertions.hpp>
#include <utils/print_vec_mat.hpp>

//========== MAIN PROBLEM PARAMETERS =========
// values modifiable via cmd line args: default set as in reference HPCG
constexpr size_t PHYS_SYSTEM_SIZE_DEF{ 16UL };
constexpr size_t PHYS_SYSTEM_SIZE_MIN{ 4UL };
constexpr size_t DEF_COARSENING_LEVELS{ 1U };
constexpr size_t MAX_COARSENING_LEVELS{ 4U };
constexpr size_t MAX_ITERATIONS_DEF{ 56UL };
constexpr size_t SMOOTHER_STEPS_DEF{ 1 };

// internal values
constexpr double SYSTEM_DIAG_VALUE { 26.0 };
constexpr double SYSTEM_NON_DIAG_VALUE { -1.0 };
constexpr size_t BAND_WIDTH_3D { 13UL };
constexpr size_t HALO_RADIUS { 1U };
//============================================

constexpr double MAX_NORM { 4.0e-14 };

using namespace grb;
using namespace algorithms;

static const char * const TEXT_HIGHLIGHT = "===> ";
#define thcout ( std::cout << TEXT_HIGHLIGHT )
#define thcerr ( std::cerr << TEXT_HIGHLIGHT )


/**
 * @brief Container for system parameters to create the HPCG problem.
 */
struct system_input {
	size_t nx, ny, nz;
	size_t max_coarsening_levels;
};

/**
 * @brief Container for the parameters for the HPCG simulation.
 */
struct simulation_input : public system_input {
	size_t test_repetitions;
	size_t max_iterations;
	size_t smoother_steps;
	bool evaluation_run;
	bool no_preconditioning;
	bool print_iter_stats;
};

/**
 * @brief Containers for test outputs.
 */
struct output {
	RC error_code;
	size_t test_repetitions;
	size_t performed_iterations;
	double residual;
	grb::utils::TimerResults times;
	std::unique_ptr< PinnedVector< double > > pinnedVector;
	double square_norm_diff;

	output() {
		error_code = SUCCESS;
		test_repetitions = 0;
		performed_iterations = 0;
		residual = 0.0;
	}
};

/**
 * @brief Returns the closets power of 2 bigger or equal to \p n .
 */
template< typename T = size_t >
T static next_pow_2( T n ) {
	static_assert( std::is_integral< T >::value, "Integral required." );
	--n;
	n |= ( n >> 1 );
	for( unsigned i = 1; i <= sizeof( T ) * 4; i *= 2 ) {
		const unsigned shift = static_cast< T >( 1U ) << i;
		n |= ( n >> shift );
	}
	return n + 1;
}

/**
 * @brief Builds and initializes a 3D system for an HPCG simulation according to the given 3D system sizes.
 * @return RC grb::SUCCESS if the system initialization within GraphBLAS succeeded
 */
static RC build_3d_system( std::unique_ptr< hpcg_data< double, double, double > > & holder, const system_input & in ) {
	const std::array< size_t, 3 > physical_sys_sizes { in.nx, in.ny, in.nz };
	struct hpcg_system_params< 3, double > params {
		physical_sys_sizes, HALO_RADIUS, BAND_WIDTH_3D * 2 + 1, SYSTEM_DIAG_VALUE, SYSTEM_NON_DIAG_VALUE, PHYS_SYSTEM_SIZE_MIN, in.max_coarsening_levels, 2
	};

	return build_hpcg_system< 3, double >( holder, params );
}

#ifdef HPCG_PRINT_SYSTEM
static void print_system( const hpcg_data< double, double, double > & data ) {
	print_matrix( data.A, 70, "A" );
	multi_grid_data< double, double > * coarser = data.coarser_level;
	while( coarser != nullptr ) {
		print_matrix( coarser->coarsening_matrix, 50, "COARSENING MATRIX" );
		print_matrix( coarser->A, 50, "COARSER SYSTEM MATRIX" );
		coarser = coarser->coarser_level;
	}
}
#endif

#ifdef HPCG_PRINT_STEPS
template< typename T,
		class Ring = Semiring< grb::operators::add< T >, grb::operators::mul< T >, grb::identities::zero, grb::identities::one >
	>
void print_norm( const grb::Vector< T > & r, const char * head, const Ring & ring ) {
	T norm = 0;
	RC ret = grb::dot( norm, r, r, ring ); // norm = r' * r;
	(void)ret;
	assert( ret == SUCCESS );
	if( head != nullptr ) {
		std::cout << head << ": ";
		printf(">>> %s: %lf\n", head, norm );
	} else {
		printf(">>> %lf\n", norm );
	}
}
#endif

/**
 * @brief Main test, building an HPCG problem and running the simulation closely following the
 * parameters in the reference HPCG test.
 */
void grbProgram( const simulation_input & in, struct output & out ) {
	// get user process ID
	const size_t pid { spmd<>::pid() };
	assert( pid < spmd<>::nprocs() );
	grb::utils::Timer timer;

	// assume successful run
	out.error_code = SUCCESS;
	RC rc { SUCCESS };

	// wrap hpcg_data inside a unique_ptr to forget about cleaning chores
	std::unique_ptr< hpcg_data< double, double, double > > hpcg_state;
	if( pid == 0 ) {
		thcout << "beginning input generation..." << std::endl;
	}
	timer.reset();
	rc = build_3d_system( hpcg_state, in );
	double input_duration { timer.time() };

	if( rc != SUCCESS ) {
		std::cerr << "Failure to generate the system (" << toString( rc ) << ")." << std::endl;
		out.error_code = rc;
		return;
	}
	if( pid == 0 ) {
		thcout << "input generation time (ms): " << input_duration << std::endl;
	}

#ifdef HPCG_PRINT_SYSTEM
	if( pid == 0 ) {
		print_system( *hpcg_state );
	}
#endif

	Matrix< double > & A { hpcg_state->A };
	Vector< double > & x { hpcg_state->x };
	Vector< double > & b { hpcg_state->b };

	// set vectors as from standard HPCG benchmark
	set( x, 1.0 );
	set( b, 0.0 );
	rc = grb::mxv( b, A, x, grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one >() );
	set( x, 0.0 );

#ifdef HPCG_PRINT_SYSTEM
	if( pid == 0 ) {
		print_vector( x, 50, "X" );
		print_vector( b, 50, "B" );
	}
#endif

	out.times.preamble = timer.time();

	const bool with_preconditioning = ! in.no_preconditioning;
	if( in.evaluation_run ) {
		out.test_repetitions = 0;
		if( pid == 0 ) {
			thcout << "beginning evaluation run..." << std::endl;
		}
		timer.reset();
		rc = hpcg( *hpcg_state, with_preconditioning, in.smoother_steps, in.smoother_steps,
			in.max_iterations, 0.0, out.performed_iterations, out.residual, false );
		double single_time = timer.time();
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			thcerr << "error during evaluation run" << std::endl;
			out.error_code = rc;
			return;
		}
		out.times.useful = single_time;
		out.test_repetitions = static_cast< size_t >( 1000.0 / single_time ) + 1;

		if( pid == 0 ) {
			thcout << "Evaluation run" << std::endl;
		}

		std::cout << "  iterations: " << out.performed_iterations << std::endl
			<< "  computed residual: " << out.residual << std::endl
			<< "  time taken (ms): " << out.times.useful << std::endl
			<< "  deduced inner repetitions for 1s duration: " << out.test_repetitions << std::endl;
		return;
	}

	// do a cold run to warm the system up
	if( pid == 0 ) {
		thcout << "beginning cold run..." << std::endl;
	}
	timer.reset();
	rc = hpcg( *hpcg_state, with_preconditioning, in.smoother_steps, in.smoother_steps,
		1, 0.0, out.performed_iterations, out.residual, false );
	double iter_duration { timer.time() };
	if( pid == 0 ) {
		thcout << "cold run duration (ms): " << iter_duration << std::endl;
	}


	// do benchmark
	for( size_t i = 0; i < in.test_repetitions && rc == SUCCESS; ++i ) {
		rc = set( x, 0.0 );
		assert( rc == SUCCESS );
		if( pid == 0 ) {
			thcout << "beginning iteration: " << i << std::endl;
		}
		timer.reset();
		rc = hpcg( *hpcg_state, with_preconditioning, in.smoother_steps, in.smoother_steps,
			in.max_iterations, 0.0, out.performed_iterations, out.residual, in.print_iter_stats );
		iter_duration = timer.time();
		out.times.useful += iter_duration;
		if( pid == 0 ) {
			thcout << "repetition,duration (ms): " << i << "," << iter_duration << std::endl;
		}
		out.test_repetitions++;
		if( rc != SUCCESS ) {
			break;
		}
	}
	out.times.useful /= static_cast< double >( in.test_repetitions );

	if( spmd<>::pid() == 0 ) {
		if( rc == SUCCESS ) {
			thcout << "repetitions, average time (ms): " << out.test_repetitions
				<< ", " << out.times.useful << std::endl;
		} else {
			thcerr << "Failure: call to HPCG did not succeed (" << toString( rc )
				<< ")." << std::endl;
		}
	}

	// start postamble
	timer.reset();
	// set error code
	out.error_code = rc;

	Semiring< grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one > ring;
	grb::set( b, 1.0 );
	out.square_norm_diff = 0.0;
	grb::eWiseMul( b, -1.0, x, ring );
	grb::dot( out.square_norm_diff, b, b, ring );

	// output
	out.pinnedVector = std::unique_ptr< PinnedVector< double > >( new PinnedVector< double >( x, SEQUENTIAL ) );
	// finish timing
	const double time_taken { timer.time() };
	out.times.postamble = time_taken;
}

/**
 * @brief Parser the command-line arguments to extract the simulation information and checks they are valid.
 */
static void parse_arguments( simulation_input &, size_t &, double &, int, char ** );

#ifdef TEST_ITER
static void test_iters();
static void test_iters2();
#endif

int main( int argc, char ** argv ) {
	simulation_input sim_in;
	size_t test_outer_iterations;
	double max_residual_norm;

#ifdef TEST_ITER
	test_iters();
	test_iters2();
	return 0;
#endif

	parse_arguments( sim_in, test_outer_iterations, max_residual_norm, argc, argv );
	thcout << "System size x: " << sim_in.nx << std::endl;
	thcout << "System size y: " << sim_in.ny << std::endl;
	thcout << "System size z: " << sim_in.nz << std::endl;
	thcout << "System max coarsening levels " << sim_in.max_coarsening_levels << std::endl;
	thcout << "Test repetitions: " << sim_in.test_repetitions << std::endl;
	thcout << "Max iterations: " << sim_in.max_iterations << std::endl;
	thcout << "Direct launch: " << std::boolalpha << sim_in.evaluation_run << std::noboolalpha << std::endl;
	thcout << "No conditioning: " << std::boolalpha << sim_in.no_preconditioning << std::noboolalpha << std::endl;
	thcout << "Print iteration residual: " << std::boolalpha << sim_in.print_iter_stats << std::noboolalpha << std::endl;
	thcout << "Smoother steps: " << sim_in.smoother_steps << std::endl;
	thcout << "Test outer iterations: " << test_outer_iterations << std::endl;
	thcout << "Maximum norm for residual: " << max_residual_norm << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc { SUCCESS };

	// launch estimator (if requested)
	if( sim_in.evaluation_run ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, sim_in, out, true );
		if( rc == SUCCESS ) {
			sim_in.test_repetitions = out.test_repetitions;
		} else {
			thcout << "launcher.exec returns with non-SUCCESS error code " << grb::toString( rc ) << std::endl;
			std::exit( -1 );
		}
	}

	// launch full benchmark
	grb::Benchmarker< AUTOMATIC > benchmarker;
	rc = benchmarker.exec( &grbProgram, sim_in, out, 1, test_outer_iterations, true );
	ASSERT_RC_SUCCESS( rc );
	thcout << "Benchmark completed successfully and took " << out.performed_iterations << " iterations to converge with residual " << out.residual << std::endl;

	if( ! out.pinnedVector ) {
		thcerr << "no output vector to inspect" << std::endl;
	} else {
		const PinnedVector< double > &solution { *out.pinnedVector };
		thcout << "Size of x is " << solution.size() << std::endl;
		if( solution.size() > 0 ) {
			print_vector( solution, 30, "SOLUTION" );
		} else {
			thcerr << "ERROR: solution contains no values" << std::endl;
		}
	}

	ASSERT_RC_SUCCESS( out.error_code );

	double residual_norm { sqrt( out.square_norm_diff ) };
	thcout << "Residual norm: " << residual_norm << std::endl;

	ASSERT_LT( residual_norm, max_residual_norm );

	thcout << "Test OK" << std::endl;
	return 0;
}

static void parse_arguments( simulation_input & sim_in, size_t & outer_iterations, double & max_residual_norm, int argc, char ** argv ) {

	argument_parser parser;
	parser.add_optional_argument( "--nx", sim_in.nx, PHYS_SYSTEM_SIZE_DEF, "physical system size along x" )
		.add_optional_argument( "--ny", sim_in.ny, PHYS_SYSTEM_SIZE_DEF, "physical system size along y" )
		.add_optional_argument( "--nz", sim_in.nz, PHYS_SYSTEM_SIZE_DEF, "physical system size along z" )
		.add_optional_argument( "--max-coarse-levels", sim_in.max_coarsening_levels, DEF_COARSENING_LEVELS,
			"maximum level for coarsening; 0 means no coarsening; note: actual "
			"level may be limited"
			" by the minimum system dimension" )
		.add_optional_argument( "--test-rep", sim_in.test_repetitions, grb::config::BENCHMARKING::inner(), "consecutive test repetitions before benchmarking" )
		.add_optional_argument( "--init-iter", outer_iterations, grb::config::BENCHMARKING::outer(), "test repetitions with complete initialization" )
		.add_optional_argument( "--max-iter", sim_in.max_iterations, MAX_ITERATIONS_DEF, "maximum number of HPCG iterations" )
		.add_optional_argument( "--max-residual-norm", max_residual_norm, MAX_NORM,
			"maximum norm for the residual to be acceptable (does NOT limit "
			"the execution of the algorithm)" )
		.add_optional_argument( "--smoother-steps", sim_in.smoother_steps, SMOOTHER_STEPS_DEF, "number of pre/post-smoother steps; 0 disables smoothing" )
		.add_option( "--evaluation-run", sim_in.evaluation_run, false,
			"launch single run directly, without benchmarker (ignore "
			"repetitions)" )
		.add_option( "--no-preconditioning", sim_in.no_preconditioning, false, "do not apply pre-conditioning via multi-grid V cycle" )
		.add_option( "--print-iter-stats", sim_in.print_iter_stats, false, "on each iteration, print more statistics" );


	parser.parse( argc, argv );

	// check for valid values
	size_t ssize { std::max( next_pow_2( sim_in.nx ), PHYS_SYSTEM_SIZE_MIN ) };
	if( ssize != sim_in.nx ) {
		std::cout << "Setting system size x to " << ssize << " instead of " << sim_in.nx << std::endl;
		sim_in.nx = ssize;
	}
	ssize = std::max( next_pow_2( sim_in.ny ), PHYS_SYSTEM_SIZE_MIN );
	if( ssize != sim_in.ny ) {
		std::cout << "Setting system size y to " << ssize << " instead of " << sim_in.ny << std::endl;
		sim_in.ny = ssize;
	}
	ssize = std::max( next_pow_2( sim_in.nz ), PHYS_SYSTEM_SIZE_MIN );
	if( ssize != sim_in.nz ) {
		std::cout << "Setting system size z to " << ssize << " instead of " << sim_in.nz << std::endl;
		sim_in.nz = ssize;
	}
	if( sim_in.max_coarsening_levels > MAX_COARSENING_LEVELS ) {
		std::cout << "Setting max coarsening level to " << MAX_COARSENING_LEVELS << " instead of " << sim_in.max_coarsening_levels << std::endl;
		sim_in.max_coarsening_levels = MAX_COARSENING_LEVELS;
	}
	if( sim_in.test_repetitions == 0 ) {
		std::cerr << "ERROR no test runs selected: set \"--test-rep >0\"" << std::endl;
		std::exit( -1 );
	}
	if( sim_in.max_iterations == 0 ) {
		std::cout << "Setting number of iterations to 1" << std::endl;
		sim_in.max_iterations = 1;
	}
}




struct NZ {
	size_t i;
	size_t j;
	double v;

	NZ( size_t _i, size_t _j, double _v ): i(_i), j(_j), v(_v) {}

	bool operator!=( const NZ& o ) const {
		return i != o.i || j != o.j || v != o.v;
	}
};

#ifdef TEST_ITER
static void test_iters() {

	using clock = std::chrono::steady_clock;

	constexpr size_t DIMS = 3;

	std::array< unsigned, DIMS > finer_sizes{ 1024, 1024, 1024};
	std::array< unsigned, DIMS > coarser_sizes;
	for( size_t i = 0; i < finer_sizes.size(); i++ ) {
		coarser_sizes[ i ] = finer_sizes[ i ] / 2;
	}

	size_t rows { std::accumulate( coarser_sizes.cbegin(), coarser_sizes.cend(), 1UL, std::multiplies< size_t >() ) };

	std::array< size_t, DIMS > lfiner_sizes{ 1024, 1024, 1024};
	std::array< size_t, DIMS > lcoarser_sizes{};
	for( size_t i = 0; i < lfiner_sizes.size(); i++ ) {
		lcoarser_sizes[ i ] = lfiner_sizes[ i ] / 2;
	}
	grb::algorithms::old::coarsener_generator_iterator< DIMS, double > sbegin( lcoarser_sizes, lfiner_sizes, 0 );
	grb::algorithms::old::coarsener_generator_iterator< DIMS, double > send( lcoarser_sizes, lfiner_sizes, rows );


	using citer = hpcg_coarsener_builder< DIMS, unsigned, double >::hpcg_coarsener_iterator;
	hpcg_coarsener_builder< DIMS, unsigned, double > coarsener( coarser_sizes, finer_sizes );
	citer pbegin( coarsener.make_begin_iterator() );
	const citer pend( coarsener.make_end_iterator() );

	size_t num_elements = pend - pbegin;
	std::cout << "number of elements: " << num_elements << std::endl;

	std::vector< NZ > svalues;
	svalues.reserve( num_elements);
	typename clock::time_point start( clock::now() );
	for( ; sbegin != send; ++sbegin ) {
		// printf( "inserting %lu %lu\n", sbegin.i(), sbegin.j() );
		svalues.emplace_back( sbegin.i(), sbegin.j(), sbegin.v() );
	}
	typename clock::time_point finish( clock::now() );
	std::cout << "sequential generation time (ms): " <<
		std::chrono::duration< double, std::milli >( finish - start ).count() << std::endl;




	const size_t nthreads = omp_get_max_threads();
	size_t per_thread_num = ( num_elements + nthreads - 1 ) / nthreads;
	std::vector< std::vector< NZ > > tvalues( nthreads );
	for( size_t i = 0; i < nthreads; i++ ) {
		tvalues[i].reserve( per_thread_num );
	}
	start = clock::now();
	#pragma omp parallel
	{

		int t = omp_get_thread_num();
		std::vector< NZ > &tv = tvalues[ t ];
		// printf( "thread %d, size %lu\n", t, tv.size() );
		#pragma omp for schedule( static )
		for( auto it = pbegin; it != pend; ++it ) {
			tv.emplace_back( it.i(), it.j(), it.v() );
			// printf( "thread %d: inserting %lu %lu\n", t, it.i(), it.j() );
		}
	}
	finish = clock::now();
	std::cout << "parallel generation time (ms): " <<
		std::chrono::duration< double, std::milli >( finish - start ).count() << std::endl;

	std::vector< NZ > pvalues;
	for( const std::vector< NZ > &tv: tvalues ) {
		pvalues.insert( pvalues.end(), tv.cbegin(), tv.cend() );
	}


	if( svalues.size() != pvalues.size() ) {
		std::cout << "different sizes!" << std::endl;
		std::exit(-1);
	}

	for( size_t i = 0; i < svalues.size(); i++ ) {
		if( svalues[i] != pvalues[i] ) {
			std::cout << "error at position " << i << std::endl;
		}
	}
	std::cout << "all OK" << std::endl;
}

static void test_iters2() {

	using clock = std::chrono::steady_clock;

	constexpr size_t DIMS = 3, halo_size = 1;
	constexpr double diag_value = 26.0, non_diag_value = -1.0;

	std::array< unsigned, DIMS > sys_sizes{ 64, 64, 64};
	size_t n { std::accumulate( sys_sizes.cbegin(), sys_sizes.cend(), 1UL, std::multiplies< size_t >() ) };

	std::array< size_t, DIMS > large_sys_sizes{ 64, 64, 64};
	old::matrix_generator_iterator< DIMS, double > sbegin( large_sys_sizes, 0UL, halo_size, diag_value, non_diag_value );
	old::matrix_generator_iterator< DIMS, double > send( large_sys_sizes, n, halo_size, diag_value, non_diag_value );

	hpcg_builder< DIMS, unsigned, double > hpcg_system( sys_sizes, halo_size );
	matrix_generator_iterator< DIMS, unsigned, double > pbegin(
		hpcg_system.make_begin_iterator( diag_value, non_diag_value ) );
	matrix_generator_iterator< DIMS, unsigned, double > pend(
		hpcg_system.make_end_iterator( diag_value, non_diag_value )
	);

	size_t num_elements = pend - pbegin;
	std::cout << "number of elements: " << num_elements << std::endl;

	std::vector< NZ > svalues;
	svalues.reserve( num_elements);
	typename clock::time_point start( clock::now() );
	for( ; sbegin != send; ++sbegin ) {
		svalues.emplace_back( sbegin.i(), sbegin.j(), sbegin.v() );
	}
	typename clock::time_point finish( clock::now() );
	std::cout << "sequential generation time (ms): " <<
		std::chrono::duration< double, std::milli >( finish - start ).count() << std::endl;




	const size_t nthreads = omp_get_max_threads();
	size_t per_thread_num = ( num_elements + nthreads - 1 ) / nthreads;
	std::vector< std::vector< NZ > > tvalues( nthreads );
	for( size_t i = 0; i < nthreads; i++ ) {
		tvalues[i].reserve( per_thread_num );
	}
	start = clock::now();
	#pragma omp parallel
	{

		int t = omp_get_thread_num();
		std::vector< NZ > &tv = tvalues[ t ];
		// printf( "thread %d, size %lu\n", t, tv.size() );
		#pragma omp for schedule( static )
		for( auto it = pbegin; it != pend; ++it ) {
			tv.emplace_back( it.i(), it.j(), it.v() );
			// printf( "thread %d: inserting %lu %lu\n", t, it.i(), it.j() );
		}
	}
	finish = clock::now();
	std::cout << "parallel generation time (ms): " <<
		std::chrono::duration< double, std::milli >( finish - start ).count() << std::endl;

	std::vector< NZ > pvalues;
	for( const std::vector< NZ > &tv: tvalues ) {
		pvalues.insert( pvalues.end(), tv.cbegin(), tv.cend() );
	}


	if( svalues.size() != pvalues.size() ) {
		std::cout << "different sizes!" << std::endl;
		std::exit(-1);
	}

	for( size_t i = 0; i < svalues.size(); i++ ) {
		if( svalues[i] != pvalues[i] ) {
			std::cout << "error at position " << i << std::endl;
		}
	}

	std::cout << "all OK" << std::endl;
}
#endif // TEST_ITER
