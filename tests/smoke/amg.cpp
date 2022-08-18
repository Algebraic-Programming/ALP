
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * Test for AMG solver, levels provided by AMGCL.
 * @file amg.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @author Denis Jelovina (denis.jelovina@huawei.com)
 *
 *
 * @date 2022-10-08
 */
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>
#include <string>
#include <graphblas.hpp>
#include <graphblas/algorithms/amg/amg.hpp>
#include <graphblas/algorithms/amg/system_building_utils.hpp>
#include <graphblas/utils/Timer.hpp>
#include <utils/argument_parser.hpp>
#include <utils/assertions.hpp>
#include <utils/print_vec_mat.hpp>

#include <vector>

#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
//#include <amgcl/solver/runtime.hpp>
//#include <amgcl/make_solver.hpp>
#include <graphblas/algorithms/amg/plugin/amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>


//#include <lib/amgcl.h>

typedef amgcl::backend::builtin<double>           Backend;
typedef amgcl::amg<Backend,
				   amgcl::coarsening::ruge_stuben,
				   amgcl::relaxation::spai0> AMG;

// forward declaration for the tracing facility
template< typename T,
	class Ring = grb::Semiring<
		grb::operators::add< T >,
		grb::operators::mul< T >,
		grb::identities::zero,
		grb::identities::one
	>
>
void print_norm( const grb::Vector< T > &r, const char * head, const Ring &ring = Ring() );

#ifdef AMG_PRINT_STEPS

// AMG_PRINT_STEPS requires defining the following symbols

/**
 * simply prints \p args on a dedicated line.
 */
#define DBG_println( args ) std::cout << args << std::endl;

/**
 * prints \p head and the norm of \p r.
 */
#define DBG_print_norm( vec, head ) print_norm( vec, head )
#endif

//========== MAIN PROBLEM PARAMETERS =========
// values modifiable via cmd line args: default set as in reference AMG
constexpr size_t DEF_COARSENING_LEVELS = 1U;
constexpr size_t MAX_COARSENING_LEVELS = 10U;
constexpr size_t DEF_COARSE_ENOUGH = 100;
constexpr size_t MAX_ITERATIONS_DEF = 56UL;
constexpr size_t SMOOTHER_STEPS_DEF = 1;
//============================================

constexpr double MAX_NORM { 4.0e-14 };

using namespace grb;
using namespace algorithms;

static const char * const TEXT_HIGHLIGHT = "===> ";
#define thcout ( std::cout << TEXT_HIGHLIGHT )
#define thcerr ( std::cerr << TEXT_HIGHLIGHT )


/**
 * Container to store matrices loaded from a AMGCL.
 */
template< typename T = double >
struct mat_data {
	size_t nz, n, m;
	std::vector<size_t> i_data;
	std::vector<size_t> j_data;
	std::vector<T> v_data;
	mat_data(size_t nz, size_t n, size_t m,
			 std::vector<size_t> i_data,
			 std::vector<size_t> j_data,
			 std::vector<T> v_data):
		nz(nz), n(n), m(m), i_data(i_data), j_data(j_data), v_data(v_data)
	{}
};

static bool matloaded = false;

/**
 * Container for the parameters for the AMG simulation.
 */
struct simulation_input {
	size_t max_coarsening_levels;
	size_t coarse_enough;
	size_t test_repetitions;
	size_t max_iterations;
	size_t smoother_steps;
	const char * matAfile_c_str;
	bool evaluation_run;
	bool no_preconditioning;
};

/**
 * @brief Container to store all data for AMG hierarchy.
 */
class preloaded_matrices {

public :

	std::vector<mat_data<>>  Amat_data;
	std::vector<mat_data<>>  Pmat_data;
	std::vector<mat_data<>>  Rmat_data;
	std::vector<std::vector<double>>  Dvec_data;

	grb::RC get_vcyclehierarchy_AGMCL( const simulation_input &in ){
		grb::RC rc = SUCCESS;

		std::vector<size_t>    ptr;
		std::vector<size_t>    col;
		std::vector<double> val;
		std::vector<double> rhs;
		size_t rows, cols;

		std::string fname( in.matAfile_c_str );
        if ( fname.compare( fname.size() - 4, 4, ".mtx" ) != 0 ) {
#ifdef DEBUG
			std::cout << "reading " << fname << " file, as binary crs file.\n";
#endif
            amgcl::io::read_crs( fname, rows, ptr, col, val );
#ifdef DEBUG
			cols = rows;
			std::cout << "file " << fname << " contains " << rows << " x " << cols << " matrix\n";
#endif
        } else {
#ifdef DEBUG
			std::cout << "reading .mtx file\n";
#endif
			std::tie(rows, cols) = amgcl::io::mm_reader( in.matAfile_c_str )( ptr, col, val );
            assert( rows == cols );
        }

#ifdef DEBUG
		std::cout << " ptr.size() = " << ptr.size() << "\n";
		std::cout << " col.size() = " << col.size() << "\n";
		std::cout << " val.size() = " << val.size() << "\n";
		std::cout << " rows, cols =  " << rows << ", " << cols << "\n";
		std::cout << " in.max_coarsening_levels = " << in.max_coarsening_levels << "\n";
		std::cout << " in.coarse_enough = " << in.coarse_enough << "\n";
#endif

		auto A = std::tie(rows, ptr, col, val);
		AMG::params prm;
		prm.coarse_enough = in.coarse_enough;
		prm.direct_coarse = false;
		prm.max_levels = in.max_coarsening_levels;

		AMG amg( A, prm );
		save_levels( amg, Amat_data, Pmat_data, Rmat_data, Dvec_data );

		if ( Amat_data.size() != in.max_coarsening_levels ) {
			std::cout << " max_coarsening_levels readjusted to : ";
			std::cout << Amat_data.size() << "\n";
		}


#ifdef DEBUG
		std::cout << " --> Amat_data.size() =" << Amat_data.size() << "\n";
		for( size_t i = 0; i < Amat_data.size(); i++ ) {
			std::cout << " amgcl check data: level =" << i << "\n";
			std::cout << "    **Amat_data ** \n";
			std::cout << "    nz =" << Amat_data[ i ].nz << "\n";
			std::cout << "     n =" << Amat_data[ i ].n << "\n";
			std::cout << "     m =" << Amat_data[ i ].m << "\n";
			for( size_t k = 0; k < Amat_data[ i ].nz; k++ ) {
				if( k < 3 || k + 3 >= Amat_data[ i ].nz ){
					std::cout << "     " << std::fixed  << "[" << std::setw(5);
					std::cout << Amat_data[ i ].i_data[ k ] << " ";
					std::cout << std::setw(5) << Amat_data[ i ].j_data[ k ] << "] ";
					std::cout << std::scientific << std::setw(5);
					std::cout << Amat_data[ i ].v_data[ k ] << "\n";
				}
			}
			std::cout << "\n\n";
		}

		for( size_t i = 0; i < Pmat_data.size(); i++ ) {
			std::cout << " amgcl check data: level =" << i << "\n";
			std::cout << "    **Pmat_data ** \n";
			std::cout << "    nz =" << Pmat_data[ i ].nz << "\n";
			std::cout << "     n =" << Pmat_data[ i ].n << "\n";
			std::cout << "     m =" << Pmat_data[ i ].m << "\n";
			for( size_t k = 0; k < Pmat_data[ i ].nz; k++ ) {
				if( k < 3 || k + 3 >= Pmat_data[ i ].nz ){
					std::cout << "     " << std::fixed  << "[" << std::setw(5);
					std::cout << Pmat_data[ i ].i_data[ k ] << " ";
					std::cout << std::setw(5) << Pmat_data[ i ].j_data[ k ] << "] ";
					std::cout << std::scientific << std::setw(5);
					std::cout << Pmat_data[ i ].v_data[ k ] << "\n";
				}
			}
			std::cout << "\n\n";
		}

		for( size_t i = 0; i < Rmat_data.size(); i++ ) {
			std::cout << " amgcl check data: level =" << i << "\n";
			std::cout << "    **Rmat_data ** \n";
			std::cout << "    nz =" << Rmat_data[ i ].nz << "\n";
			std::cout << "     n =" << Rmat_data[ i ].n << "\n";
			std::cout << "     m =" << Rmat_data[ i ].m << "\n";
			for( size_t k = 0; k < Rmat_data[ i ].nz; k++ ) {
				if( k < 3 || k + 3 >= Rmat_data[ i ].nz ){
					std::cout << "     " << std::fixed  << "[" << std::setw(5);
					std::cout << Rmat_data[ i ].i_data[ k ] << " ";
					std::cout << std::setw(5) << Rmat_data[ i ].j_data[ k ] << "] ";
					std::cout << std::scientific << std::setw(5);
					std::cout << Rmat_data[ i ].v_data[ k ] << "\n";
				}
			}
			std::cout << "\n\n";
		}

		for( size_t i = 0; i < Dvec_data.size(); i++ ) {
			std::cout << " amgcl check data: level =" << i << "\n";
			std::cout << "    **Dvec_data ** \n";
			std::cout << "     n =" << Dvec_data[ i ].size() << "\n";
			for( size_t k = 0; k < Dvec_data[ i ].size(); k++ ) {
				if( k < 3 || k + 3 >= Dvec_data[ i ].size() ){
					std::cout << "     "  << std::scientific << std::setw(5);
					std::cout << Dvec_data[ i ][ k ] << "\n";
				}
			}
			std::cout << "\n\n";
		}
#endif

		return rc;
	}

};



preloaded_matrices inputData;

/**
 * Containers for test outputs.
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

#ifdef AMG_PRINT_SYSTEM
static void print_system( const amg_data< double, double, double > & data ) {
	print_matrix( data.A, 70, "A" );
	multi_grid_data< double, double > * coarser = data.coarser_level;
	while( coarser != nullptr ) {
		print_matrix( coarser->coarsening_matrix, 50, "COARSENING MATRIX" );
		print_matrix( coarser->A, 50, "COARSER SYSTEM MATRIX" );
		coarser = coarser->coarser_level;
	}
}
#endif

#ifdef AMG_PRINT_STEPS
template<
	typename T,
	class Ring = Semiring<
		grb::operators::add< T >,
		grb::operators::mul< T >,
		grb::identities::zero,
		grb::identities::one >
	>
void print_norm( const grb::Vector< T > &r, const char *head, const Ring &ring ) {
	T norm = 0;
	RC ret = grb::dot( norm, r, r, ring ); // residual = r' * r;
	(void)ret;
	assert( ret == SUCCESS );
	std::cout << ">>> ";
	if( head != nullptr ) {
		std::cout << head << ": ";
	}
	std::cout << norm << std::endl;
}
#endif

/**
 * Main test, building an AMG problem and running the simulation closely
 *  following the parameters in the reference AMG test.
 */
void grbProgram( const simulation_input &in, struct output &out ) {
	grb::utils::Timer timer;
	timer.reset();

	// get user process ID
	assert( spmd<>::pid() < spmd<>::nprocs() );

	// assume successful run
	out.error_code = SUCCESS;
	RC rc { SUCCESS };

	if( ! matloaded ) {
		rc = inputData.get_vcyclehierarchy_AGMCL( in );
		if( rc != SUCCESS ) {
			std::cerr << "Failure to read data" << std::endl;
		}
		matloaded = true ;
	}

	out.times.io = timer.time();
	timer.reset();

	// wrap amg_data inside a unique_ptr to forget about cleaning chores
	std::unique_ptr< amg_data< double, double, double > > amg_state;
	rc = build_amg_system< double >( amg_state, inputData );


	if( rc != SUCCESS ) {
		std::cerr << "Failure to generate the system (" << toString( rc ) << ")." << std::endl;
		out.error_code = rc;
		return;
	}

#ifdef AMG_PRINT_SYSTEM
	if( spmd<>::pid() == 0 ) {
		print_system( *amg_state );
	}
#endif

	Matrix< double > &A = amg_state->A;
	Vector< double > &x = amg_state->x;
	Vector< double > &b = amg_state->b;

	// set vectors as from standard AMG benchmark
	set( x, 1.0 );
	set( b, 0.0 );
	rc = grb::mxv( b, A, x,
		grb::Semiring< grb::operators::add< double >,
			grb::operators::mul< double >,
			grb::identities::zero,
			grb::identities::one >() );
	set( x, 0.0 );

	double norm_b = 0;
	rc = grb::dot( norm_b, b, b,
		grb::Semiring< grb::operators::add< double >,
			grb::operators::mul< double >,
			grb::identities::zero,
			grb::identities::one >() );
	(void)rc;
	assert( rc == SUCCESS );

#ifdef AMG_PRINT_SYSTEM
	if( spmd<>::pid() == 0 ) {
		print_vector( x, 50, " ---> X(1)" );
		print_vector( b, 50, " ---> B(1)" );
	}
#endif

	out.times.preamble = timer.time();
	timer.reset();

	const bool with_preconditioning = ! in.no_preconditioning;
	out.test_repetitions = 0;
	if( in.evaluation_run ) {
		double single_time_start = timer.time();
		rc = amg( *amg_state, with_preconditioning, in.smoother_steps,
			in.smoother_steps, in.max_iterations, 0.0, out.performed_iterations, out.residual );
		double single_time = timer.time() - single_time_start;
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		out.times.useful = single_time;
		out.test_repetitions = static_cast< size_t >( 1000.0 / single_time ) + 1;
	} else {
		// do benchmark
		double time_start = timer.time();
		for( size_t i = 0; i < in.test_repetitions && rc == SUCCESS; ++i ) {
			rc = set( x, 0.0 );
			assert( rc == SUCCESS );
			rc = amg( *amg_state, with_preconditioning, in.smoother_steps,
				in.smoother_steps, in.max_iterations, 0.0, out.performed_iterations, out.residual );

			out.test_repetitions++;
			if( rc != SUCCESS ) {
				break;
			}

		}
		double time_taken = timer.time() - time_start;
		rc = rc ? rc : collectives<>::reduce( time_taken, 0, operators::max< double >() );
		out.times.useful = time_taken / static_cast< double >( out.test_repetitions );

#ifdef AMG_PRINT_SYSTEM
		rc = rc ? rc : grb::eWiseLambda( [ &x ]( const size_t i ) { x[ i ] -= 1;}, x );
		print_norm (x, " norm(x)");
#endif
		// sleep( 1 );
	}
	timer.reset();

#ifdef AMG_PRINT_SYSTEM
	if( spmd<>::pid() == 0 ) {
		print_vector( x, 50, " x(first 50 elements)" );
		print_vector( b, 50, " b(first 50 elements)" );
	}
#endif

	if( spmd<>::pid() == 0 ) {
		if( rc == SUCCESS ) {
			if( in.evaluation_run ) {
				std::cout << "Info: cold AMG completed within " << out.performed_iterations
				          << " iterations. Last computed residual is " << out.residual
				          << ". Time taken was " << out.times.useful
				          << " ms. Deduced inner repetitions parameter of " << out.test_repetitions
				          << " to take 1 second or more per inner benchmark." << std::endl;
			} else {
				std::cout << "Final residual= "<< out.residual << " relative error= "
				          <<  out.residual/sqrt(norm_b) << "\n";
				std::cout << "Average time taken for each of " << out.test_repetitions
				          << " AMG calls (hot start): " << out.times.useful << std::endl;
			}
		} else {
			std::cerr << "Failure: call to AMG did not succeed (" << toString( rc ) << ")." << std::endl;
		}
	}

	// start postamble
	timer.reset();
	// set error code
	out.error_code = rc;

	Semiring< grb::operators::add< double >,
		grb::operators::mul< double >,
		grb::identities::zero,
		grb::identities::one > ring;
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
 * Parser the command-line arguments to extract the simulation information
 * and checks they are valid.
 */
static void parse_arguments( simulation_input &, size_t &, double &, int, char ** );

int main( int argc, char ** argv ) {
	simulation_input sim_in;
	size_t test_outer_iterations;
	double max_residual_norm;

	parse_arguments( sim_in, test_outer_iterations, max_residual_norm, argc, argv );
	thcout << "System max coarsening levels " << sim_in.max_coarsening_levels << std::endl;
	thcout << "Test repetitions: " << sim_in.test_repetitions << std::endl;
	thcout << "Max iterations: " << sim_in.max_iterations << std::endl;
	thcout << "Direct launch: " << std::boolalpha << sim_in.evaluation_run << std::noboolalpha << std::endl;
	thcout << "No conditioning: " << std::boolalpha << sim_in.no_preconditioning << std::noboolalpha << std::endl;
	thcout << "Smoother steps: " << sim_in.smoother_steps << std::endl;
	thcout << "Test outer iterations: " << test_outer_iterations << std::endl;
	thcout << "Maximum norm for residual: " << max_residual_norm << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

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
	thcout << "Benchmark completed successfully and took " << out.performed_iterations
	       << " iterations to converge with residual " << out.residual << std::endl;

	if( ! out.pinnedVector ) {
		thcerr << "no output vector to inspect" << std::endl;
	} else {
		const PinnedVector< double > &solution = *out.pinnedVector;
		thcout << "Size of x is " << solution.size() << std::endl;
		if( solution.size() > 0 ) {
			print_vector( solution, 30, "SOLUTION" );
		} else {
			thcerr << "ERROR: solution contains no values" << std::endl;
		}
	}

	ASSERT_RC_SUCCESS( out.error_code );

	double residual_norm = sqrt( out.square_norm_diff );
	thcout << "Residual norm: " << residual_norm << std::endl;

	ASSERT_LT( residual_norm, max_residual_norm );

	thcout << "Test OK" << std::endl;
	return 0;
}

static void parse_arguments( simulation_input &sim_in, size_t &outer_iterations,
	double &max_residual_norm, int argc, char **argv ) {

	argument_parser parser;
	parser.add_optional_argument( "--max_coarse-levels", sim_in.max_coarsening_levels, DEF_COARSENING_LEVELS,
			"maximum level for coarsening; 0 means no coarsening; note: actual "
			"level may be limited by the minimum system dimension" )
		.add_optional_argument( "--coarse_enough", sim_in.coarse_enough, DEF_COARSE_ENOUGH,
			"max size of the coarsest levels: stop coarsening after this matrix size" )
		.add_optional_argument( "--mat_file", sim_in.matAfile_c_str,
			"file contining matrix in matrix market format "
			"i.e. '--mat_file A.mtx'  ")
		.add_optional_argument( "--test-rep", sim_in.test_repetitions, grb::config::BENCHMARKING::inner(),
			"consecutive test repetitions before benchmarking" )
		.add_optional_argument( "--init-iter", outer_iterations, grb::config::BENCHMARKING::outer(),
			"test repetitions with complete initialization" )
		.add_optional_argument( "--max_iter", sim_in.max_iterations, MAX_ITERATIONS_DEF,
			"maximum number of AMG iterations" )
		.add_optional_argument( "--max-residual-norm", max_residual_norm, MAX_NORM,
			"maximum norm for the residual to be acceptable (does NOT limit "
			"the execution of the algorithm)" )
		.add_optional_argument( "--smoother-steps", sim_in.smoother_steps,
			SMOOTHER_STEPS_DEF, "number of pre/post-smoother steps; 0 disables smoothing" )
		.add_option( "--evaluation-run", sim_in.evaluation_run, false,
			"launch single run directly, without benchmarker (ignore repetitions)" )
		.add_option( "--no-preconditioning", sim_in.no_preconditioning, false,
			"do not apply pre-conditioning via multi-grid V cycle" );

	parser.parse( argc, argv );

	// check for valid values
	if( sim_in.max_coarsening_levels > MAX_COARSENING_LEVELS ) {
		std::cout << "Setting max coarsening level to " << MAX_COARSENING_LEVELS
		          << " instead of " << sim_in.max_coarsening_levels << std::endl;
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
