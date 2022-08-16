
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
#include <amgcl/solver/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <graphblas/algorithms/amg/plugin/amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/io/mm.hpp>


#include <lib/amgcl.h>

typedef amgcl::backend::builtin<double>           Backend;
typedef amgcl::amg<Backend,
				   amgcl::runtime::coarsening::wrapper,
				   amgcl::runtime::relaxation::wrapper> AMG;
typedef amgcl::runtime::solver::wrapper<Backend>  ISolver;
typedef amgcl::make_solver<AMG, ISolver>          Solver;



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
constexpr size_t DEF_COARSENING_LEVELS{ 1U };
constexpr size_t MAX_COARSENING_LEVELS{ 4U };
constexpr size_t MAX_ITERATIONS_DEF{ 56UL };
constexpr size_t SMOOTHER_STEPS_DEF{ 1 };
//============================================

constexpr double MAX_NORM { 4.0e-14 };

using namespace grb;
using namespace algorithms;

static const char * const TEXT_HIGHLIGHT = "===> ";
#define thcout ( std::cout << TEXT_HIGHLIGHT )
#define thcerr ( std::cerr << TEXT_HIGHLIGHT )

#define DEBUG

// /**
//  * Container to store matrices loaded from a file.
//  */
// template< typename T = double >
// struct mat_data {
// 	size_t *i_data;
// 	size_t *j_data;
// 	T *v_data;
// 	size_t nz, n, m;

// 	void resize( size_t innz, size_t inn, size_t inm ){
// 		nz=innz;
// 		n=inn;
// 		m=inm;
// 		i_data = new size_t [ nz ];
// 		j_data = new size_t [ nz ];
// 		v_data = new T [ nz ];
// 	};

// 	mat_data( ){
// 		nz = 0;
// 	};

// 	mat_data( size_t in ){
// 		resize( in );
// 	};

// 	size_t size(){
// 		return nz;
// 	};

// 	size_t get_n(){
// 		return n;
// 	};

// 	size_t get_m(){
// 		return m;
// 	};

// 	~mat_data(){
// 		delete [] i_data;
// 		delete [] i_data;
// 		delete [] v_data;
// 	};
// };

// /**
//  * Container to store vectors loaded from a file.
//  */
// template< typename T = double >
// struct vec_data {
// 	T *v_data;
// 	size_t n;

// 	void resize( size_t in ){
// 		n=in;
// 		v_data = new T [ n ];
// 	};

// 	vec_data( ){
// 		n = 0;
// 	};

// 	vec_data( size_t in ){
// 		resize( in );
// 	};

// 	size_t size(){
// 		return n;
// 	};

// 	~vec_data(){
// 		delete [] v_data;
// 	};
// };

static bool matloaded = false;

/**
 * Container for the parameters for the AMG simulation.
 */
struct simulation_input {
	size_t max_coarsening_levels;
	size_t test_repetitions;
	size_t max_iterations;
	size_t smoother_steps;
	const char * matAfile_c_str;
	// // vectors of input matrix filenames
	// std::vector< std::string > matAfiles;
	// std::vector< std::string > matMfiles;
	// std::vector< std::string > matPfiles;
	// std::vector< std::string > matRfiles;
	bool evaluation_run;
	bool no_preconditioning;
};

// template<typename CharT, typename TraitsT = std::char_traits< CharT > >
// class vectorwrapbuf :
// 	public std::basic_streambuf<CharT, TraitsT> {
// public:
// 	vectorwrapbuf(  std::vector<CharT> &vec) {
// 		this->setg( vec.data(), vec.data(), vec.data() + vec.size() );
//     }
// };
// std::istream& operator>>(std::istream& is, std::string& s){
// 	std::getline(is, s);
// 	return is;
// }
 
/**
 * @brief Container to store all data for AMG hierarchy.
 */
class preloaded_matrices : public simulation_input {

public :

	amgclHandle solver;

// 	size_t nzAmt, nzMmt, nzPmt, nzRmt;
// 	mat_data< double > *matAbuffer;
// 	vec_data< double > *matMbuffer;
// 	mat_data< double > *matPbuffer;
// 	mat_data< double > *matRbuffer;

// 	grb::RC read_matrix(
// 		std::vector< std::string > &fname,
// 		mat_data<double> *data
// 	) {
// 		grb::RC rc = SUCCESS;
// 		for ( size_t i = 0; i < fname.size(); i++ ) {
// 			grb::utils::MatrixFileReader<
// 				double, std::conditional<
// 					( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
// 							grb::config::RowIndexType,
// 							grb::config::ColIndexType >::type
// 				> parser_mat( fname[ i ].c_str(), true );
// #ifdef DEBUG
// 			std::cout << " ---> parser_mat.filename()=" << parser_mat.filename() << "\n";
// 			std::cout << " ---> parser_mat.nz()=" << parser_mat.nz() << "\n";
// 			std::cout << " ---> parser_mat.n()=" << parser_mat.n() << "\n";
// 			std::cout << " ---> parser_mat.m()=" << parser_mat.m() << "\n";
// 			std::cout << " ---> parser_mat.entries()=" << parser_mat.entries() << "\n";
// #endif
// 			// very poor choice and temp solution size = 2 x nz
// 			data[i].resize( parser_mat.entries()*2, parser_mat.n(), parser_mat.m() );
// 			std::ifstream inFile( fname[ i ], std::ios::binary | std::ios::ate );
// 			if( ! inFile.is_open() ) {
// 				std::cout << " Cannot open "<< fname[ i ].c_str() <<  "\n";
// 				return( PANIC );
// 			}
// 			std::streamsize size = inFile.tellg();
// 			inFile.seekg( 0, std::ios::beg );
// 			std::vector< char > buffer( size );
// 			if ( inFile.read( buffer.data(), size ) ) {
// 				// all fine
// 			}
// 			else {
// 				std::cout << " Cannot read "<< fname[ i ].c_str() <<  "\n";
// 				return( PANIC );
// 			}
// 			inFile.close();
// 			vectorwrapbuf< char > databuf( buffer );
// 			std::istream isdata( &databuf );
// 			std::string headder;
// 			isdata >> headder;
// 			while( headder.at( 0 ) == '%' ) {
// 				isdata >> headder;
// 			}
// 			std::stringstream ss( headder );
// 			size_t n, m, nz;
// 			ss >> n >> m >> nz ;
// 			size_t k = 0;
// 			for ( size_t j = 0; j < nz; j++ ) {
// 				size_t itmp, jtmp;
// 				double vtmp;
// 				isdata >> itmp >> jtmp >> vtmp;
// 				data[ i ].i_data[ k ] = itmp - 1;
// 				data[ i ].j_data[ k ] = jtmp - 1;
// 				data[ i ].v_data[ k ] = vtmp;
// 				k += 1;
// 			}
// 			data[ i ].nz = k;
// 		}
// 		return ( rc );
// 	}

// 	grb::RC read_vector( std::vector< std::string > &fname,
// 						 vec_data< double > *data ) {
// 		grb::RC rc = SUCCESS;
// 		for ( size_t i = 0; i < fname.size(); i++ ){
// #ifdef DEBUG
// 			std::cout << " Reading " << fname[ i ].c_str() << ".\n";
// #endif
// 			std::ifstream inFile( fname[ i ], std::ios::binary | std::ios::ate );
// 			if( ! inFile.is_open() ) {
// 				std::cout << " Cannot open "<< fname[ i ].c_str() <<  "\n";
// 				return( PANIC );
// 			}
// 			std::streamsize size = inFile.tellg();
// 			inFile.seekg( 0, std::ios::beg );
// 			std::vector< char > buffer( size );
// 			if ( inFile.read( buffer.data(), size ) ) {
// 				// all fine
// 			}
// 			else {
// 				std::cout << " Cannot read "<< fname[ i ].c_str() <<  "\n";
// 				return( PANIC );
// 			}
// 			inFile.close();
// 			vectorwrapbuf< char > databuf( buffer );
// 			std::istream isdata( &databuf );
// 			std::string headder;
// 			size_t n, m;
// 			std::getline( isdata, headder ); // skip the first line
// 			while( headder.at( 0 ) == '%' ) {
// 				std::getline( isdata, headder );
// 			}
// 			std::stringstream ss( headder );
// 			ss >> n >> m;
// 			std::cout << " Reading from" << fname[ i ] << " dense matrix with dimensions: "
// 			          << " n x m = " << n << " x " << m << "\n";
// 			data[ i ].resize( n );
// 			for ( size_t j = 0; j < n; j++ ) {
// 				isdata >> data[ i ].v_data[ j ];
// 			}
// 		}
// 		return( rc );
// 	}

	grb::RC read_vec_matrics(){
		grb::RC rc = SUCCESS;

		std::vector<int>    ptr;
		std::vector<int>    col;
		std::vector<double> val;
		std::vector<double> rhs;

		amgclHandle prm = amgcl_params_create();
		size_t rows, cols;
		std::tie(rows, cols) = amgcl::io::mm_reader("/home/d/Repos/edapp2/EDApp2/saved_amg_levels/level_0_A.mtx")(ptr, col, val);
#ifdef DEBUG
		std::cout << " ptr.size() = " << ptr.size() << "\n";
		std::cout << " col.size() = " << col.size() << "\n";
		std::cout << " val.size() = " << val.size() << "\n";
		std::cout << " rows, cols =  " << rows << ", " << cols << "\n";
#endif

		amgcl_params_sets(prm, "precond.relax.type", "spai0");
		amgcl_params_sets(prm, "precond.coarsening.type","ruge_stuben");
		amgcl_params_seti(prm, "precond.max_levels",5);
		amgcl_params_seti(prm, "precond.coarse_enough",100);

		solver = amgcl_solver_create(
			rows, ptr.data(), col.data(), val.data(), prm
		);
		//TODO: extract amgcl data



		// nzAmt = matAfiles.size() ;
		// nzMmt = matMfiles.size() ;
		// nzPmt = matPfiles.size() ;
		// nzRmt = matRfiles.size() ;

		// matAbuffer = new mat_data< double > [ nzAmt ];
		// matMbuffer = new vec_data< double > [ nzMmt ];
		// matPbuffer = new mat_data< double > [ nzPmt ];
		// matRbuffer = new mat_data< double > [ nzRmt ];

		// rc = rc ? rc : read_matrix( matAfiles, matAbuffer );
		// rc = rc ? rc : read_matrix( matRfiles, matRbuffer );
		// // rc = rc ? rc : read_matrix( matPfiles, matPbuffer );
		// rc = rc ? rc : read_vector( matMfiles, matMbuffer );

		return rc;
	}


	~preloaded_matrices(){
		amgcl_solver_destroy(solver);
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
		// //preloaded_matrices inputData;
		// inputData.matAfiles = in.matAfiles;
		// inputData.matMfiles = in.matMfiles;
		// inputData.matPfiles = in.matPfiles;
		// inputData.matRfiles = in.matRfiles;

		rc = inputData.read_vec_matrics();
		if( rc != SUCCESS ) {
			std::cerr << "Failure to read data" << std::endl;
		}
		matloaded = true ;
	}

	std::cout << " ----> testing: return !\n";
	return ;

	out.times.io = timer.time();
	timer.reset();

	// wrap amg_data inside a unique_ptr to forget about cleaning chores
	std::unique_ptr< amg_data< double, double, double > > amg_state;
	rc = build_amg_system< double >( amg_state, in.max_coarsening_levels, inputData );

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
		.add_optional_argument( "--mat_files_pattern", sim_in.matAfile_c_str,
			"file pattern for files contining matrices A, M_diag, P, R "
			"i.e. '--mat_a_file_names /path/to/dir/level_  --max_coarse-levels 2' will read "
			"/path/to/dir/level_0_A.mtx,  /path/to/dir/level_1_A.mtx, /path/to/dir/level_2_A.mtx ... " )
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

	// std::string matAfile = sim_in.matAfile_c_str;
	// if ( ! matAfile.empty() ) {
	// 	std::cout << "Using  <<" << matAfile << ">> pattern to read matrices " << std::endl;
	// 	for ( size_t i = 0 ; i < sim_in.max_coarsening_levels ; i++ ){
	// 		std::string fnamebase = matAfile + std::to_string(  i );
	// 		std::string fnameA = fnamebase + "_A.mtx";
	// 		std::string fnameM = fnamebase + "_M_diag.mtx";
	// 		std::string fnameP = fnamebase + "_P.mtx";
	// 		std::string fnameR = fnamebase + "_R.mtx";
	// 		sim_in.matAfiles.push_back( fnameA );
	// 		sim_in.matMfiles.push_back( fnameM );
	// 		sim_in.matPfiles.push_back( fnameP );
	// 		sim_in.matRfiles.push_back( fnameR );
	// 	}
	// 	{
	// 		std::string fnamebase = matAfile + std::to_string( sim_in.max_coarsening_levels );
	// 		std::string fnameA = fnamebase + "_A.mtx";
	// 		sim_in.matAfiles.push_back (fnameA);
	// 		std::string fnameM = fnamebase + "_M_diag.mtx";
	// 		sim_in.matMfiles.push_back (fnameM);
	// 	}
	// 	std::cout << "files to read matrices: " << std::endl;
	// 	for ( std::string fname: sim_in.matAfiles ) {
	// 		std::cout << fname << " \n";
	// 	}

	// 	for ( std::string fname: sim_in.matMfiles ) {
	// 		std::cout << fname << " \n";
	// 	}

	// 	for ( std::string fname: sim_in.matPfiles ) {
	// 		std::cout << fname << " \n";
	// 	}

	// 	for ( std::string fname: sim_in.matRfiles ) {
	// 		std::cout << fname << " \n";
	// 	}

	// }
	// else {
	// 	std::cout << "No pattern to read matrices provided" << std::endl;
	// }

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
