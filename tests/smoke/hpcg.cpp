
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
#include <algorithm>

#include <graphblas.hpp>

//========== TRACE SOLVER STEPS =========
// to easily trace the steps of the solver, just define this symbol
// #define HPCG_PRINT_STEPS

// here we define a custom macro and do not use NDEBUG since the latter is not defined for smoke tests
#ifdef HPCG_PRINT_STEPS
#include <cstdio>

// HPCG_PRINT_STEPS requires defining the following symbols

/**
 * @brief simply prints \p args on a dedicated line.
 */
#define DBG_println( args ) std::cout << args << std::endl;
// forward declaration for the tracing facility
template< typename T > void print_norm( const grb::Vector< T > &r, const char * head );
/**
 * @brief prints \p head and the norm of \p r.
 */
#define DBG_print_norm( vec, head ) print_norm( vec, head )
#endif

#include <graphblas/algorithms/hpcg/hpcg.hpp>
#include <graphblas/algorithms/multigrid/multigrid_building_utils.hpp>
#include <graphblas/algorithms/hpcg/system_building_utils.hpp>

#include <graphblas/utils/Timer.hpp>

#include <utils/argument_parser.hpp>
#include <utils/assertions.hpp>
#include <utils/print_vec_mat.hpp>

//========== MAIN PROBLEM PARAMETERS =========
// default simulation parameters, set as in reference HPCG
// users can input different ones via the cmd line
constexpr size_t PHYS_SYSTEM_SIZE_DEF = 16UL;
constexpr size_t PHYS_SYSTEM_SIZE_MIN = 2UL;
constexpr size_t MAX_COARSENING_LEVELS = 3U;
constexpr size_t MAX_ITERATIONS_DEF = 56UL;
constexpr size_t SMOOTHER_STEPS_DEF = 1;

// internal values defining the simulated physical system
constexpr double SYSTEM_DIAG_VALUE = 26.0;
constexpr double SYSTEM_NON_DIAG_VALUE = -1.0;
constexpr size_t BAND_WIDTH_3D = 13UL;
constexpr size_t HALO_RADIUS = 1U;
constexpr double MAX_NORM = 4.0e-14;
//============================================

using namespace grb;
using namespace algorithms;

static const char * const TEXT_HIGHLIGHT = "===> ";
#define thcout ( std::cout << TEXT_HIGHLIGHT )
#define thcerr ( std::cerr << TEXT_HIGHLIGHT )
#define MASTER_PRINT( pid, txt ) if( pid == 0 ) { std::cout << txt; }

/**
 * Container for system parameters to create the HPCG problem.
 */
struct system_input {
	size_t nx, ny, nz;
	size_t max_coarsening_levels;
};

/**
 * Container for the parameters for the HPCG simulation.
 */
struct simulation_input : public system_input {
	size_t inner_test_repetitions;
	size_t max_iterations;
	size_t smoother_steps;
	bool evaluation_run;
	bool no_preconditioning;
	bool print_iter_stats;
};

using IOType = double;
using NonzeroType = double;
using InputType = double;
using ResidualType = double;
using StdRing = Semiring< grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
	grb::identities::zero, grb::identities::one >;
using StdMinus = operators::subtract< NonzeroType >;
using coord_t = size_t;

/**
 * Container for test outputs.
 */
struct output {
	RC error_code = SUCCESS;
	size_t inner_test_repetitions = 0;
	grb::utils::TimerResults times;
	std::unique_ptr< PinnedVector< IOType > > pinnedVector;
	NonzeroType square_norm_diff = 0.0;
	cg_out_data< NonzeroType > cg_out = { 0, 0.0 };
};

using hpcg_runner_t = HPCGRunnerType< IOType, NonzeroType, InputType, ResidualType,
	StdRing, StdMinus >;
using mg_data_t = multigrid_data< IOType, NonzeroType >;
using coarsening_data_t = coarsening_data< IOType, NonzeroType >;
using smoothing_data_t = smoother_data< IOType >;
using hpcg_data_t = mg_cg_data< IOType, NonzeroType, InputType >;

#ifdef HPCG_PRINT_SYSTEM
static void print_system(
	const std::vector< std::unique_ptr< mg_data_t > > &system_levels,
	const std::vector< std::unique_ptr< coarsening_data_t > > &coarsener_levels
) {
	assert( spmd<>::nprocs() == 1 ); // distributed printin of system not implemented
	print_matrix( system_levels[ 0 ]->A, 70, "A" );
	for( size_t i = 0; i < coarsener_levels.size(); i++ ) {
		print_matrix( coarsener_levels[i ] ->coarsening_matrix, 50, "COARSENING MATRIX" );
		print_matrix( system_levels[ i + 1 ]->A, 50, "COARSER SYSTEM MATRIX" );
	}
}
#endif

#ifdef HPCG_PRINT_STEPS
template<
	typename T,
	class Ring
> void print_norm( const grb::Vector< T > & r, const char * head, const Ring & ring ) {
	T norm = 0;
	RC ret = grb::dot( norm, r, r, ring ); // norm = r' * r;
	(void)ret;
	assert( ret == SUCCESS );
	if( spmd<>::pid() != 0 ) {
		return;
	}
	if( head != nullptr ) {
		printf(">>> %s: %lf\n", head, norm );
	} else {
		printf(">>> %lf\n", norm );
	}
}

template< typename T > void print_norm( const grb::Vector< T > & r, const char * head ) {
	return print_norm( r, head, StdRing() );
}
#endif


/**
 * Allocates the data structure input to the various simulation steps (CG, multi-grid, coarsening, smoothing)
 * for each level of the multi-grid. The input is the vector of system sizes \p mg_sizes, with sizes in
 * monotonically \b decreasing order (finest system first).
 *
 * This routine is algorithm-agnositc, as long as the constructors of the data types meet the requirements
 * explained in \ref multigrid_allocate_data().
 */
static void allocate_system_structures(
	const std::vector< size_t > &mg_sizes,
	std::vector< std::unique_ptr< mg_data_t > > &system_levels,
	std::vector< std::unique_ptr< coarsening_data_t > > &coarsener_levels,
	std::vector< std::unique_ptr< smoothing_data_t > > &smoother_levels,
	std::unique_ptr< hpcg_data_t > &holder
) {
	const size_t pid { spmd<>::pid() };
	grb::utils::Timer timer;

	hpcg_data_t *data{ new hpcg_data_t( mg_sizes[ 0 ] ) };
	holder = std::unique_ptr< hpcg_data_t >( data );
	MASTER_PRINT( pid, "allocating data for the MultiGrid simulation...");
	timer.reset();
	multigrid_allocate_data( mg_sizes, system_levels, coarsener_levels, smoother_levels );
	double time = timer.time();
	MASTER_PRINT( pid, " time (ms) " << time << std::endl )

	// zero all vectors
	MASTER_PRINT( pid, "zeroing all vectors...");
	timer.reset();
	grb::RC rc = data->zero_temp_vectors();
	ASSERT_RC_SUCCESS( rc );
	std::for_each( system_levels.begin(), system_levels.end(),
		[]( std::unique_ptr< mg_data_t > &s) { ASSERT_RC_SUCCESS( s->zero_temp_vectors() ); } );
	std::for_each( coarsener_levels.begin(), coarsener_levels.end(),
		[]( std::unique_ptr< coarsening_data_t > &s) { ASSERT_RC_SUCCESS( s->zero_temp_vectors() ); } );
	std::for_each( smoother_levels.begin(), smoother_levels.end(),
		[]( std::unique_ptr< smoothing_data_t > &s) { ASSERT_RC_SUCCESS( s->zero_temp_vectors() ); } );
	time = timer.time();
	MASTER_PRINT( pid, " time (ms) " << time << std::endl );
}

/**
 * Builds and initializes a 3D system for an HPCG simulation according to the given 3D system sizes.
 */
static void build_3d_system(
	const system_input & in,
	std::vector< std::unique_ptr< mg_data_t > > &system_levels,
	std::vector< std::unique_ptr< coarsening_data_t > > &coarsener_levels,
	std::vector< std::unique_ptr< smoothing_data_t > > &smoother_levels,
	std::unique_ptr< hpcg_data_t > &holder
) {
	constexpr size_t DIMS = 3;
	using builder_t = grb::algorithms::HPCGSystemBuilder< DIMS, coord_t, NonzeroType >;
	const size_t pid { spmd<>::pid() };
	grb::utils::Timer timer;

	hpcg_system_params< DIMS, NonzeroType > params {
		{ in.nx, in.ny, in.nz }, HALO_RADIUS, SYSTEM_DIAG_VALUE, SYSTEM_NON_DIAG_VALUE,
			PHYS_SYSTEM_SIZE_MIN, in.max_coarsening_levels, 2
	};

	std::vector< builder_t > mg_generators;
	MASTER_PRINT( pid, "building HPCG generators for " << ( in.max_coarsening_levels + 1 )
		<< " levels..." );
	timer.reset();
	hpcg_build_multigrid_generators( params, mg_generators );
	double time = timer.time();
	MASTER_PRINT( pid, " time (ms) " << time << std::endl );
	MASTER_PRINT( pid, "built HPCG generators for " << mg_generators.size()
		<< " levels" << std::endl );


	std::vector< size_t > mg_sizes;
	// exclude main system
	std::transform( mg_generators.cbegin(), mg_generators.cend(), std::back_inserter( mg_sizes  ),
		[] ( const builder_t &b ) { return b.system_size(); } );
	allocate_system_structures( mg_sizes, system_levels, coarsener_levels, smoother_levels, holder );
	assert( mg_generators.size() == system_levels.size() );
	assert( mg_generators.size() == smoother_levels.size() );
	assert( mg_generators.size() - 1 == coarsener_levels.size() );

	for( size_t i = 0; i < mg_generators.size(); i++) {
		MASTER_PRINT( pid, "SYSTEM LEVEL " << i << std::endl );
		MASTER_PRINT( pid, " populating system matrix: " );
		timer.reset();
		grb::RC rc = hpcg_populate_system_matrix( mg_generators[ i ], system_levels.at(i)->A );
		time = timer.time();
		ASSERT_RC_SUCCESS( rc );
		MASTER_PRINT( pid, " time (ms) " << time << std::endl )

		MASTER_PRINT( pid, " populating smoothing data: " );
		timer.reset();
		rc = hpcg_populate_smoothing_data( mg_generators[ i ], *smoother_levels[ i ] );
		time = timer.time();
		ASSERT_RC_SUCCESS( rc );
		MASTER_PRINT( pid, " time (ms) " << time << std::endl )

		if( i > 0 ) {
			MASTER_PRINT( pid, " populating coarsening data: " );
			timer.reset();
			rc = hpcg_populate_coarsener( mg_generators[ i - 1 ], mg_generators[ i ], *coarsener_levels[ i - 1 ] );
			time = timer.time();
			ASSERT_RC_SUCCESS( rc );
			MASTER_PRINT( pid, " time (ms) " << time << std::endl )
		}
	}
}

/**
 * @brief Main test, building an HPCG problem and running the simulation closely following the
 * parameters in the reference HPCG test.
 */
void grbProgram( const simulation_input & in, struct output & out ) {
	// get user process ID
	const size_t pid { spmd<>::pid() };
	MASTER_PRINT( pid, "beginning input generation..." << std::endl );

	grb::utils::Timer timer;

	// wrap hpcg_data inside a unique_ptr to forget about cleaning chores
	std::unique_ptr< hpcg_data_t > hpcg_state;

	hpcg_runner_t hpcg_runner( build_hpcg_runner< IOType, NonzeroType, InputType, ResidualType,
		StdRing, StdMinus >( in.smoother_steps ) );
	auto &mg_runner = hpcg_runner.mg_runner;
	auto &coarsener = mg_runner.coarsener_runner;
	auto &smoother = mg_runner.smoother_runner;
	hpcg_runner.cg_opts.max_iterations = in.max_iterations;
	hpcg_runner.cg_opts.tolerance = 0.0;
	hpcg_runner.cg_opts.with_preconditioning = ! in.no_preconditioning;

	timer.reset();
	build_3d_system( in, mg_runner.system_levels, coarsener.coarsener_levels, smoother.levels, hpcg_state );
	double input_duration { timer.time() };
	MASTER_PRINT( pid, "input generation time (ms): " << input_duration << std::endl );

#ifdef HPCG_PRINT_SYSTEM
	if( pid == 0 ) {
		print_system( mg_runner.system_levels, coarsener.coarsener_levels );
	}
#endif

	Matrix< NonzeroType > & A { mg_runner.system_levels[ 0 ]->A };
	Vector< NonzeroType > & x { hpcg_state->x };
	Vector< NonzeroType > & b { hpcg_state->b };

	RC rc { SUCCESS };
	// set vectors as from standard HPCG benchmark
	set( x, 1.0 );
	set( b, 0.0 );
	rc = grb::mxv( b, A, x, StdRing() );
	set( x, 0.0 );

#ifdef HPCG_PRINT_SYSTEM
	if( pid == 0 ) {
		print_vector( x, 50, "X" );
		print_vector( b, 50, "B" );
	}
#endif

	out.times.preamble = timer.time();

	cg_out_data< NonzeroType > &cg_out = out.cg_out;
	mg_data_t &grid_base = *mg_runner.system_levels[ 0 ];

	// do a cold run to warm the system up
	MASTER_PRINT( pid, TEXT_HIGHLIGHT << "beginning cold run..." << std::endl );
	hpcg_runner.cg_opts.max_iterations = 1;
	timer.reset();
	rc = hpcg_runner( grid_base, *hpcg_state, cg_out );
	double iter_duration { timer.time() };
	ASSERT_RC_SUCCESS( rc );
	MASTER_PRINT( pid, " time (ms): " << iter_duration << std::endl );

	hpcg_runner.cg_opts.max_iterations = in.max_iterations;
	hpcg_runner.cg_opts.print_iter_stats = in.print_iter_stats;
	// do benchmark
	const size_t inner_test_repetitions = in.evaluation_run ? 1 : in.inner_test_repetitions;
	if( in.evaluation_run ) {
		MASTER_PRINT( pid, TEXT_HIGHLIGHT << "beginning evaluation run..." << std::endl );
	} else {
		MASTER_PRINT( pid, TEXT_HIGHLIGHT << "beginning test run..." << std::endl );
	}
	out.inner_test_repetitions = 0;
	out.times.useful = 0.0;
	for( size_t i = 0; i < inner_test_repetitions; ++i ) {
		rc = set( x, 0.0 );
		ASSERT_RC_SUCCESS( rc );
		MASTER_PRINT( pid, TEXT_HIGHLIGHT << "beginning iteration: " << i << std::endl );
		timer.reset();
		rc = hpcg_runner( grid_base, *hpcg_state, cg_out );
		out.times.useful += timer.time();
		ASSERT_RC_SUCCESS( rc );
		MASTER_PRINT( pid, "repetition,duration (ms): " << i << "," << iter_duration << std::endl );
		out.inner_test_repetitions++;
	}
	if( in.evaluation_run ) {
		rc = collectives<>::reduce( iter_duration, 0, operators::max< double >() );
		ASSERT_RC_SUCCESS( rc );
		out.inner_test_repetitions = static_cast< size_t >( 1000.0 / out.times.useful ) + 1;
		MASTER_PRINT( pid, "Evaluation run" << std::endl
			<< "  computed residual: " << cg_out.norm_residual << std::endl
			<< "  iterations: " << cg_out.iterations << std::endl
			<< "  time taken (ms): " << out.times.useful << std::endl
			<< "  deduced inner repetitions for 1s duration: " << out.inner_test_repetitions << std::endl );
		return;
	}
	out.times.useful /= static_cast< double >( in.inner_test_repetitions );

	MASTER_PRINT( pid, TEXT_HIGHLIGHT << "repetitions,average time (ms): " << out.inner_test_repetitions
				<< ", " << out.times.useful << std::endl );

	// start postamble
	timer.reset();
	// set error code
	out.error_code = rc;

	grb::set( b, 1.0 );
	grb::eWiseMul( b, -1.0, x, StdRing() );
	out.square_norm_diff = 0.0;
	grb::dot( out.square_norm_diff, b, b, StdRing() );

	// output
	out.pinnedVector.reset( new PinnedVector< NonzeroType >( x, SEQUENTIAL ) );
	// finish timing
	out.times.postamble = timer.time();
}

/**
 * @brief Parser the command-line arguments to extract the simulation information and checks they are valid.
 */
static void parse_arguments( simulation_input &, size_t &, double &, int, char ** );

int main( int argc, char ** argv ) {
	simulation_input sim_in;
	size_t test_outer_iterations;
	double max_diff_norm;

	parse_arguments( sim_in, test_outer_iterations, max_diff_norm, argc, argv );
	thcout << "System size x: " << sim_in.nx << std::endl;
	thcout << "System size y: " << sim_in.ny << std::endl;
	thcout << "System size z: " << sim_in.nz << std::endl;
	thcout << "System max coarsening levels " << sim_in.max_coarsening_levels << std::endl;
	thcout << "Test repetitions: " << sim_in.inner_test_repetitions << std::endl;
	thcout << "Max iterations: " << sim_in.max_iterations << std::endl;
	thcout << "Direct launch: " << std::boolalpha << sim_in.evaluation_run << std::noboolalpha << std::endl;
	thcout << "No conditioning: " << std::boolalpha << sim_in.no_preconditioning << std::noboolalpha << std::endl;
	thcout << "Print iteration residual: " << std::boolalpha << sim_in.print_iter_stats << std::noboolalpha << std::endl;
	thcout << "Smoother steps: " << sim_in.smoother_steps << std::endl;
	thcout << "Test outer iterations: " << test_outer_iterations << std::endl;
	thcout << "Maximum norm for residual: " << max_diff_norm << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc { SUCCESS };

	// launch estimator (if requested)
	if( sim_in.evaluation_run ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, sim_in, out, true );
		if( rc == SUCCESS ) {
			sim_in.inner_test_repetitions = out.inner_test_repetitions;
		} else {
			thcout << "launcher.exec returns with non-SUCCESS error code " << grb::toString( rc ) << std::endl;
			std::exit( -1 );
		}
	}

	// launch full benchmark
	grb::Benchmarker< AUTOMATIC > benchmarker;
	rc = benchmarker.exec( &grbProgram, sim_in, out, 1, test_outer_iterations, true );
	ASSERT_RC_SUCCESS( rc );
	thcout << "Benchmark completed successfully and took " << out.cg_out.iterations
		<< " iterations to converge with residual " << out.cg_out.norm_residual << std::endl;

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

	double diff_norm { sqrt( out.square_norm_diff ) };
	thcout << "Norm of difference vector |<exact solution> - <actual solution>|: " << diff_norm << std::endl;

	ASSERT_LT( diff_norm, max_diff_norm );

	thcout << "Test OK" << std::endl;
	return 0;
}

static void parse_arguments(
	simulation_input & sim_in,
	size_t & outer_iterations,
	double & max_diff_norm,
	int argc,
	char ** argv
) {

	argument_parser parser;
	parser.add_optional_argument( "--nx", sim_in.nx, PHYS_SYSTEM_SIZE_DEF, "physical system size along x" )
		.add_optional_argument( "--ny", sim_in.ny, PHYS_SYSTEM_SIZE_DEF, "physical system size along y" )
		.add_optional_argument( "--nz", sim_in.nz, PHYS_SYSTEM_SIZE_DEF, "physical system size along z" )
		.add_optional_argument( "--max-coarse-levels", sim_in.max_coarsening_levels, MAX_COARSENING_LEVELS,
			"maximum level for coarsening; 0 means no coarsening; note: actual level may be limited"
			" by the minimum system dimension" )
		.add_optional_argument( "--test-rep", sim_in.inner_test_repetitions, grb::config::BENCHMARKING::inner(),
			"consecutive test repetitions before benchmarking" )
		.add_optional_argument( "--outer-iterations", outer_iterations, 1,
			"test repetitions with complete initialization" )
		.add_optional_argument( "--max-cg-iterations", sim_in.max_iterations, MAX_ITERATIONS_DEF,
			"maximum number of CG iterations" )
		.add_optional_argument( "--max-difference-norm", max_diff_norm, MAX_NORM,
			"maximum acceptable norm |<exact solution> - <actual solution>| (does NOT limit "
			"the execution of the algorithm)" )
		.add_optional_argument( "--smoother-steps", sim_in.smoother_steps, SMOOTHER_STEPS_DEF,
			"number of pre/post-smoother steps; 0 disables smoothing" )
		.add_option( "--evaluation-run", sim_in.evaluation_run, false,
			"launch single run directly, without benchmarker (ignore repetitions)" )
		.add_option( "--no-preconditioning", sim_in.no_preconditioning, false,
			"do not apply pre-conditioning via multi-grid V cycle" )
		.add_option( "--print-iter-stats", sim_in.print_iter_stats, false,
			"on each iteration, print more statistics" );

	parser.parse( argc, argv );

	if( sim_in.max_coarsening_levels > MAX_COARSENING_LEVELS ) {
		std::cout << "Setting max coarsening level to " << MAX_COARSENING_LEVELS << " instead of " << sim_in.max_coarsening_levels << std::endl;
		sim_in.max_coarsening_levels = MAX_COARSENING_LEVELS;
	}
	if( sim_in.inner_test_repetitions == 0 ) {
		std::cerr << "ERROR no test runs selected: set \"--test-rep >0\"" << std::endl;
		std::exit( -1 );
	}
	if( sim_in.max_iterations == 0 ) {
		std::cerr << "ERROR no CG iterations selected: set \"--max-cg-iterations > 0\"" << std::endl;
		std::exit( -1 );
	}

	const size_t max_system_divider = 1 << sim_in.max_coarsening_levels;
	std::cout << "max_system_divider " << max_system_divider << std::endl;
	for( size_t s : { sim_in.nx, sim_in.ny, sim_in.nz } ) {
		std::cout << "trying " << s << std::endl;
		std::lldiv_t div_res = std::div( static_cast< long long >( s ), static_cast< long long >( max_system_divider ) );
		if ( div_res.rem != 0) {
			std::cerr << "ERROR: system size " << s << " cannot be coarsened "
				<< sim_in.max_coarsening_levels << " times because it is not exactly divisible" << std::endl;
			std::exit( -1 );
		}
		std::cout << "div_res.quot " << div_res.quot << std::endl;
		if ( div_res.quot < static_cast< long long >( PHYS_SYSTEM_SIZE_MIN ) ) {
			std::cerr << "ERROR: system size " << s << " cannot be coarsened "
				<< sim_in.max_coarsening_levels << " times because it is too small" << std::endl;
			std::exit( -1 );
		}
		if ( div_res.quot % 2 != 0 ) {
			std::cerr << "ERROR: the coarsest size " << div_res.rem << " is not a multiple of 2" << std::endl;
			std::exit( -1 );
		}
	}
}
