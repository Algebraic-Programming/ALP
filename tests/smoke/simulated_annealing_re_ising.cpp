/*
  Minimal scaffold adapted from ising_machine_sb.cpp to drive a replica-exchange
  simulated-annealing (RE-SA) solver.  Algorithmic parts are intentionally left
  unimplemented (stubs).  This file mirrors the existing IO / launcher /
  program structure and replaces numpy arrays with grb::Vector and lists of
  numpy vectors with std::vector< grb::Vector<...> >. Sparse matrices are
  represented as grb::Matrix< JType >.

  Purpose: allow running internal tests or an external-run mode while the RE-SA
  algorithm is implemented separately.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdlib>
#include <unistd.h>

#include <graphblas/algorithms/simulated_annealing_re.hpp>
#include <graphblas/nonzeroStorage.hpp>
#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/iterators/nonzeroIterator.hpp>
#include <utils/output_verification.hpp>
#include <graphblas.hpp>
#include <utils/print_vec_mat.hpp>

using namespace grb;

// #define DEBUG_SARE 1
constexpr size_t MAX_FN_SIZE = 255;

// Types
using IOType = int8_t;   // scalar/vector element type
using JType  = float;   // coupling (matrix) value type
using EnergyType  = double;   // energy value type

template< typename T1, typename T2 >
inline bool ISCLOSE( const T1 &a, const T2 &b, const double tol = 1e-4){
	return (std::abs((b)-(a))/std::abs(a) < tol) || (std::abs((b)-(a)) < tol);
}

/** Parser type */
typedef grb::utils::MatrixFileReader<
	JType,
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
	JType
> NonzeroT;

/** In-memory storage type using tuple */
typedef grb::utils::Singleton<
    std::tuple<
        size_t,                    // n (rows/columns)
        size_t,                    // nz (nonzeros)
        size_t,                    // nsweeps
        size_t,                    // n_replicas
        size_t,                      // pt_time
        unsigned,                  // seed
        std::vector<NonzeroT>,     // matrix data
        std::vector<JType>         // h vector
    >
> Storage;

namespace test_data {
    constexpr size_t n = 16;
    constexpr size_t nsweeps = 2;
    constexpr size_t n_replicas = 3;
    constexpr size_t pt_time = 1;
    constexpr unsigned seed = 8;

    const std::vector<  std::pair< std::pair< grb::config::RowIndexType, grb::config::ColIndexType >, JType > > j_matrix_data = {
		{{0, 1}, -0.2752300610319546},
		{{1, 0}, -0.2752300610319546},
		{{1, 2}, -0.10636508505639508},
		{{2, 1}, -0.10636508505639508},
		{{2, 3}, 0.3961450048806352},
		{{3, 2}, 0.3961450048806352},
		{{3, 4}, -0.15453838800213293},
		{{3, 5}, 0.4847494372852713},
		{{4, 3}, -0.15453838800213293},
		{{4, 5}, -0.4712679510367046},
		{{5, 3}, 0.4847494372852713},
		{{5, 4}, -0.4712679510367046},
		{{5, 6}, -0.1483152637298799},
		{{6, 5}, -0.1483152637298799},
		{{7, 8}, -0.11904111079614699},
		{{8, 7}, -0.11904111079614699},
		{{9, 10}, -0.18031020353297234},
		{{10, 9}, -0.18031020353297234},
		{{10, 11}, -0.22985425840853468},
		{{11, 10}, -0.22985425840853468},
		{{11, 12}, 0.30105588632639446},
		{{11, 13}, 0.13823880612312134},
		{{12, 11}, 0.30105588632639446},
		{{13, 11}, 0.13823880612312134},
		{{13, 14}, 0.10364447636911123},
		{{14, 13}, 0.10364447636911123},
		{{14, 15}, 0.2955745584289766},
		{{15, 14}, 0.2955745584289766},
    };

    const size_t nnz = j_matrix_data.size();

    const std::vector< JType > h_array_data = {
        -0.08910436,  0.58034508,  0.97719304,  0.16792909,
		-0.9221754 , -0.10715418, -0.62365497,  0.25411129,
		-0.5693644 , -0.69805978,  0.07228861, -0.79922641,
		0.46231686 , 0.87930208 ,  0.88663637, -0.25052299,
    };
}
// --- New, minimal runner configuration and result types ---
struct input {
    size_t n = test_data::n;
    size_t n_replicas = test_data::n_replicas;
    size_t nsweeps = test_data::nsweeps;
    size_t pt_time = test_data::pt_time;
    unsigned seed = test_data::seed;
    bool use_default_data = false;
    char filename_Jmatrix [ MAX_FN_SIZE + 1 ];
    char filename_h [ MAX_FN_SIZE + 1 ];
    EnergyType reference_energy = 0.0;
    bool verify = false;
    char filename_ref_solution [ MAX_FN_SIZE + 1 ];
	bool direct;
    size_t rep = 0;
    size_t outer = 1;
};

struct output {
    int error_code = 0;
    // TODO: remove itrations if not applicable
    size_t iterations = 10; // total number of iterations performed does not make sense since the code does not have convergence criteria
    EnergyType best_energy = std::numeric_limits< EnergyType >::max();
	size_t rep;
	grb::utils::TimerResults times;
    std::unique_ptr< PinnedVector< JType > > pinnedSolutionVector;
    std::unique_ptr< PinnedVector< JType > > pinnedRefSolutionVector;
    // other things like eg: best replicas ...
};

template< typename Dtype >
void read_matrix_data(const std::string &filename, std::vector<Dtype> &data, bool direct) {
    // Implementation for reading matrix data from file
	try {
		Parser parser( filename, direct );
		assert( parser.m() == parser.n() );
		std::get<0>(Storage::getData()) = parser.n();
		try {
			std::get<1>(Storage::getData()) = parser.nz();
		} catch( ... ) {
			std::get<1>(Storage::getData()) = parser.entries();
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
			auto it = parser.begin( SEQUENTIAL );
			it != parser.end( SEQUENTIAL );
			++it
		) {
			data.push_back( Dtype( *it ) );
#ifdef DEBUG_SARE
			if( spmd<>::pid() == 0 ){
				// print last data element from std::vector<NonzeroT> data
				std::cout << "readmatrix_data: " << data.back().first.first << ", "
					<< data.back().first.second << ", " << data.back().second << "\n";
			}
#endif
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}
}

template< typename NonzeroT, typename IType, typename VType >
void read_matrix_data_from_array(
	const std::vector<std::pair< std::pair< IType, IType >, VType > > &array,
	std::vector<NonzeroT> &data
) {
	// Implementation for reading matrix data from array
    try {
        for (const auto &entry : array) {
            data.emplace_back(
                NonzeroT( entry.first.first, entry.first.second, entry.second )
            );
#ifdef DEBUG_SARE
			if( spmd<>::pid() < 1 ){
				// print last data element from std::vector<NonzeroT> data
				std::cout << "read_matrix_data_from_array: " << data.back().first.first << ", "
					<< data.back().first.second << ", " << data.back().second << "\n";
			}
#endif
        }
        std::get<0>(Storage::getData()) = test_data::n;
        std::get<1>(Storage::getData()) = data.size();
    } catch (const std::exception &e) {
        std::cerr << "Failed to read matrix data from array: " << e.what() << "\n";
        return;
    }
}

template< typename Dtype >
void read_vector_data(const std::string &filename, std::vector<Dtype> &data) {
    // Implementation for reading vector data from file
    try {
        std::ifstream file( filename );
        if( !file.is_open() ) {
            std::cerr << "Failed to open vector file: " << filename << "\n";
            return;
        }
        std::string line;
        while( std::getline( file, line ) ) {
            if( line.empty() ) continue; // skip empty lines
            std::istringstream iss( line );
            Dtype v;
            if( !(iss >> v) ) {
                throw std::runtime_error( "Failed to parse line in vector file" );
            }
            data.push_back( v );
        }
    } catch( std::exception &e ) {
        std::cerr << "I/O program failed: " << e.what() << "\n";
        return;
    }
}

template< typename Dtype >
void read_vector_data_from_array(
	const std::vector<Dtype> &array, std::vector<Dtype> &data
) {
	// Implementation for reading vector data from array
	try {
		for (size_t i = 0; i < array.size(); ++i) {
			data.push_back(array[i]);
		}
	} catch (const std::exception &e) {
		std::cerr << "Failed to read vector data from array: " << e.what() << "\n";
		return;
	}
}

template<
	Backend backend,
	grb::Descriptor descr = grb::descriptors::no_operation,
	class Ring = Semiring<
		grb::operators::add< JType >, grb::operators::mul< JType >,
		grb::identities::zero, grb::identities::one
	>,
	typename Ttmp
	>
EnergyType get_energy(
				 const grb::Matrix< JType, backend >& couplings,
				 const grb::Vector< JType, backend > &local_fields,
				 const grb::Vector< IOType, backend > &state,
				 grb::Vector< Ttmp, backend > &tmp,
				 const Ring &ring = Ring()
			  ){
	const size_t n = grb::size( local_fields );
	assert( n == grb::size( state ) );
	assert( n == grb::ncols( couplings ) );
	assert( n == grb::nrows( couplings ) );
	grb::resize( tmp, n );
	grb::RC rc = grb::SUCCESS;
	EnergyType energy = 0.0;
	constexpr auto dense_descr = descr | grb::descriptors::dense;

	rc = rc ? rc : grb::set< descr >( tmp, 0.0 );
	rc = rc ? rc : grb::mxv< dense_descr >( tmp, couplings, state, ring );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, static_cast< JType >( 0.5 ), ring.getMultiplicativeMonoid() );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, local_fields, ring.getAdditiveMonoid() );
	rc = rc ? rc : grb::dot< dense_descr >( energy, tmp, state, ring );
	assert( rc == grb::SUCCESS );

	return energy;
}

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// Parse and store matrix in singleton class
	// Map Storage tuple fields to meaningful names and wire up default data
	auto &storage = Storage::getData();
	auto &n           = std::get<0>(storage); // n (rows/cols)
	auto &nnz         = std::get<1>(storage); // nz (nonzeros)
	auto &nsweeps_st  = std::get<2>(storage); // nsweeps
	auto &n_replicas_st = std::get<3>(storage); // n_replicas
	auto &pt_time_st  = std::get<4>(storage); // pt_time
	auto &seed_st     = std::get<5>(storage); // seed
	auto &Jdata       = std::get<6>(storage); // std::vector<NonzeroT>
	auto &h           = std::get<7>(storage); // std::vector<JType>

	try {
		// Initialize metadata from input (allow CLI to override defaults)
		(void) n; // initialized by read_matrix_*
		(void) nnz; // initialized by read_matrix_*
		nsweeps_st    = data_in.nsweeps;
		n_replicas_st = data_in.n_replicas;
		pt_time_st    = data_in.pt_time;
		seed_st       = data_in.seed;

		if ( data_in.use_default_data ) {
			// if no file provided, use default data from file_content
			read_matrix_data_from_array<NonzeroT>( test_data::j_matrix_data, Jdata );
			read_vector_data_from_array<JType>( test_data::h_array_data, h );
			// other data
		} else {
			// read from files if provided
			read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
			if( std::strlen( data_in.filename_h ) > 0 ) {
				read_vector_data<JType>( data_in.filename_h, h );
			}else{
				h.resize( n );
				std::fill( h.begin(), h.end(), static_cast< JType >( 0 ) );
			}
			if(data_in.verify) {
				if( std::strlen(data_in.filename_ref_solution) == 0 ) {
					std::cerr << "Reference solution file not provided for verification\n";
					return;
				}
			}
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}

	success = true;
}

void grbProgram(
    const struct input &data_in, 
    struct output &out
) {
    std::cout<< "grbProgram: running simulated-annealing RE solver (stub)\n";

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

    grb::utils::Timer timer;
	timer.reset();

    /* --- Problem setup --- */
    const size_t n = std::get<0>(Storage::getData());
    const size_t n_replicas = std::get<3>(Storage::getData());
	if( s == 0 ){
		std::cout << "problem size n = " << n << "\n";
	}
    grb::Vector< JType > h( n );

    // populate J with test (random) values
    grb::RC rc = grb::SUCCESS;

    // load into GraphBLAS
    grb::Matrix< JType > J( n, n );
	{
		const auto &data = std::get<6>(Storage::getData());
		RC io_rc = buildMatrixUnique(
			J,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cend() ),
			SEQUENTIAL
		);
		/* Once internal issue #342 is resolved this can be re-enabled
		RC io_rc = buildMatrixUnique(
			J,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cend() ),
			PARALLEL
		);*/
		io_rc = io_rc ? io_rc : wait();
		if( io_rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
				<< "(" << toString( io_rc ) << ")." << std::endl;
			out.error_code = 5;
			return;
		}
		// make J symmetric
		// grb::Matrix< JType > Jt ( n, n );
		// Jt = J;
		// const grb::Monoid< grb::operators::add< JType >, grb::identities::zero > addMonoid;
		// const grb::Monoid< grb::operators::mul< JType >, grb::identities::one > mulMonoid;
		// grb::foldl< grb::descriptors::transpose_right >( J, Jt, addMonoid); // issue  #210
		// grb::foldl<>( J, static_cast< JType >( 0.5 ), mulMonoid);

#ifdef DEBUG_SARE
		if( s == 0 && grb::ncols( J ) < 40 ) {
			std::cout << "Matrix J:\n";
			print_matrix( J );
		}
#endif
	}

    // build vector h with data from singleton
    {
        const auto &h_data = std::get<7>(Storage::getData());
		rc = rc ? rc : buildVector(
			h,
			h_data.cbegin(),
			h_data.cend(),
			SEQUENTIAL
		);
    }

    // seed RNGs (C and C++ engines) using requested seed (hardcoded default 8 if not provided)
    std::minstd_rand rng ( data_in.seed + s ); // rng or std::mt19937

    // create states storage and initialize with random 1/0 values
    std::vector< grb::Vector<IOType> > states0;
    std::vector< grb::Vector<IOType> > states;
    for ( size_t r = 0; r < n_replicas; ++r ) {
        states0.emplace_back( grb::Vector<IOType>(n) );
        states.emplace_back( grb::Vector<IOType>(n) );
        // initialize with random values
        std::uniform_int_distribution< unsigned short > randint(0,1);
        // we use buildvectorUnique with a random set of indices
        std::vector< IOType > rand_data;
        for ( size_t i = 0; i < n; ++i ) {
            rand_data.emplace_back( static_cast<IOType>(
                randint( rng ) ) );
        }
        rc = rc ? rc : grb::buildVector(
            states0.back(),
            rand_data.cbegin(),
            rand_data.cend(),
            SEQUENTIAL
        );
		rc = rc ? rc : grb::set( states.back(), states0.back() );
    }

	grb::Vector< EnergyType > tmp_energy ( n );
	EnergyType initial_energy = get_energy(  J, h, states[0], tmp_energy );

	for ( size_t r = 0; r < n_replicas; ++r ) {
		const auto en = get_energy(  J, h, states[r], tmp_energy );
		initial_energy = std::min( en, initial_energy );
    #ifdef DEBUG_SARE
		if( s == 0 ) {
			std::cout << "Initial state replica " << r << ":\n";
			print_vector( states[r], 30 ,"states values" );
			std::cout << "With energy " << en << "\n";
			std::cout << std::endl;
		}
    #endif
	}

    // also make betas vector os size n_replicas and initialize with 10.0
    grb::Vector< JType > betas( n_replicas );
    grb::Vector< EnergyType > energies( n_replicas );
    for ( size_t r = 0; rc == grb::SUCCESS && r < n_replicas; ++r ) {
        rc = rc ? rc : grb::setElement( betas, static_cast< JType >( (10.0) * std::pow<JType>( 2, r ) ), r );
        // rc = rc ? rc : grb::setElement( energies, get_energy(  J, h, states[r], tmp_energy ), r );
    }
	assert( rc == grb::SUCCESS );

	grb::Vector< IOType > best_state ( n );

	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		rc = grb::algorithms::simulated_annealing_RE_Ising(
			 J, h, states, energies, betas, best_state, out.best_energy, data_in.nsweeps, data_in.reference_energy, data_in.pt_time, data_in.seed
        );

		rc = rc ? rc : wait();
		double single_time = timer.time();
		if( !(rc == SUCCESS || rc == FAILED) ) {
			std::cerr << "Failure: call to Simulated Annealing RE did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == FAILED ) {
			std::cout << "Warning: call to Simulated Annealing RE did not converge\n";
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
					std::cout << "Info: cold Simulated Annealing RE did not converge within ";
				} else {
					std::cout << "Info: cold Simulated Annealing RE completed within ";
				}
				std::cout << out.iterations << " iterations. "
					<< "Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		for( size_t i = 0; i < 2 ; ++i ){
			for ( size_t r = 0; r < n_replicas; ++r ) {
				rc = rc ? rc : grb::set(states[r], states0[r]);
			}
			out.best_energy = std::numeric_limits< EnergyType >::max();
			rc = rc ? rc : grb::clear( energies );

			rc = grb::algorithms::simulated_annealing_RE_Ising(
			 J, h, states, energies, betas, best_state, out.best_energy, data_in.nsweeps, data_in.reference_energy, data_in.pt_time, data_in.seed + i
			);

			assert( ISCLOSE( get_energy(  J, h, best_state, tmp_energy ), out.best_energy) );
		}
		// do benchmark
		double min_time = 1e9;
		double max_time = 0;
		double total_time = 0;
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			for ( size_t r = 0; r < n_replicas; ++r ) {
				rc = rc ? rc : grb::set(states[r], states0[r]);
			}
			out.best_energy = std::numeric_limits< EnergyType >::max();
			rc = rc ? rc : grb::clear( energies );
			timer.reset();
			if( rc == SUCCESS ) {
				out.iterations = data_in.nsweeps;

                rc = grb::algorithms::simulated_annealing_RE_Ising(
			 J, h, states, energies, betas, best_state, out.best_energy, data_in.nsweeps, data_in.reference_energy, data_in.pt_time, data_in.seed + i
                );
			}
			if( grb::Properties<>::isNonblockingExecution ) {
				rc = rc ? rc : wait();
			}
			const double time_taken = timer.time();

			assert( ISCLOSE( get_energy(  J, h, best_state, tmp_energy ), out.best_energy) );
			min_time = std::min(min_time, time_taken);
			max_time = std::max(max_time, time_taken);
			total_time +=  time_taken;
			std::cerr << n_replicas << "," << data_in.nsweeps << "," << time_taken << "," << out.best_energy << std::endl;
		}

		out.times.useful = total_time / static_cast< double >( out.rep );
		// print timing at root process
		if( s == 0 ) {
			std::cout << "Average Time taken for " << out.rep << " "
				<< "Simulated Annealing RE calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << grb::toString( rc ) << std::endl;
			std::cout << "\tnumber of IM-SB iterations: " << out.rep << "\n"; std::cout << "\tmilliseconds per iteration: "
				<< ( out.times.useful / static_cast< double >( out.iterations ) ) << "\n";;
			std::cout << "\tMin Time: " << min_time << "\n";
			std::cout << "\tMax Time: " << max_time << "\n";

			if( data_in.verify ){
				if( out.best_energy < initial_energy ){
					std::cout << "Test OK" << std::endl;
				}else{
					std::cout << "Test FAILED" << std::endl;
				}
			}
		}
		sleep( 1 );
	}

	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 30;
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}
}


// --- Simple help / CLI parser for the new runner (no backward compatibility) ---
void printhelp( char *progname ) {
    std::cout << "Usage: " << progname << " [--use-default-data] [--j-matrix-fname STR] [--h-fname STR]\n"
              << "       [--n-replicas INT] [--nsweeps INT] [--seed INT]\n"
              << "       [--rep INT] [--goal INT] [--verify] [--ref-solution-fname STR] [--help]\n\n"
              << "Options:\n"
              << "  --use-default-data         Use embedded default test data\n"
              << "  --j-matrix-fname STR       Path to J matrix file (matrix-market or supported)\n"
              << "  --h-fname STR              Path to h (local fields) vector (whitespace separated), if not provided assume zero\n"
              << "  --n-replicas INT           Number of replicas (default: 3)\n"
              << "  --nsweeps INT              Number of sweeps (default: 2)\n"
              << "  --pt-time INT              Number of iterations between exchange steps (default: 1, after each iteration)\n"
              << "  --seed INT                 RNG seed (default: 8)\n"
              << "  --rep INT                  number of times to repeat the run of the algorithm (default: 1)\n"
              << "  --goal FLOAT               The value of the energy to achieve before stopping (default: 0, no such check).\n"
              << "  --verify                   Verify output against reference solution\n"
              << "  --ref-solution-fname STR   Reference solution file (required with --verify unless using default data)\n"
              << "  --help, -h                 Print this help message\n";
}

bool parse_arguments( input &in, int argc, char ** argv ) {
	std::fill( in.filename_Jmatrix, in.filename_Jmatrix + MAX_FN_SIZE, '\0' );
	std::fill( in.filename_h, in.filename_h + MAX_FN_SIZE, '\0' );
	std::fill( in.filename_ref_solution, in.filename_ref_solution + MAX_FN_SIZE, '\0' );
    in.direct = true;
    // map benchmarking configuration to the runner's fields
    in.rep = grb::config::BENCHMARKING::inner();
    in.outer = grb::config::BENCHMARKING::outer();
    in.reference_energy = static_cast<EnergyType>( 0.0 );
    // keep verify default (false) unless overridden via CLI
    in.verify = false;

    for ( int i = 1; i < argc; ++i ) {
        std::string a = argv[i];
        if ( a == "--use-default-data" ) {
            in.use_default_data = true;
        } else if ( a == "--j-matrix-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--j-matrix-fname requires an argument\n"; return false; }
            std::strncpy( in.filename_Jmatrix, argv[++i], MAX_FN_SIZE );
        } else if ( a == "--h-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--h-fname requires an argument\n"; return false; }
			std::strncpy( in.filename_h, argv[++i], MAX_FN_SIZE );
        } else if ( a == "--n-replicas" ) {
            if ( i+1 >= argc ) { std::cerr << "--n-replicas requires an argument\n"; return false; }
            in.n_replicas = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--nsweeps" ) {
            if ( i+1 >= argc ) { std::cerr << "--nsweeps requires an argument\n"; return false; }
            in.nsweeps = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--pt-time" ) {
            if ( i+1 >= argc ) { std::cerr << "--pt-time requires an argument\n"; return false; }
            in.pt_time = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--rep" ) {
            if ( i+1 >= argc ) { std::cerr << "--rep requires an argument\n"; return false; }
            in.rep = static_cast<unsigned>( std::stoul(argv[++i]) );
        } else if ( a == "--seed" ) {
            if ( i+1 >= argc ) { std::cerr << "--seed requires an argument\n"; return false; }
            in.seed = static_cast<unsigned>( std::stoul(argv[++i]) );
        } else if ( a == "--goal" ) {
            if ( i+1 >= argc ) { std::cerr << "--goal requires an argument\n"; return false; }
            in.reference_energy = std::stof(argv[++i]);
        } else if ( a == "--verify" ) {
            in.verify = true;
        } else if ( a == "--ref-solution-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--ref-solution-fname requires an argument\n"; return false; }
			std::strncpy( in.filename_ref_solution, argv[++i], MAX_FN_SIZE );
        } else if ( a == "--help" || a == "-h" ) {
            printhelp( argv[0] );
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    // basic validation
    if ( !in.use_default_data ) {
        if ( std::strlen( in.filename_Jmatrix ) == 0 ) {
            std::cerr << "Either --use-default-data or both --j-matrix-fname must be provided\n";
            return false;
        }
    }
    if ( in.verify && !in.use_default_data
			&& std::strlen( in.filename_ref_solution ) == 0 ) {
        std::cerr << "--ref-solution-fname required when --verify is used without --use-default-data\n";
        return false;
    }
    return true;
}

// --- Minimal main that uses the existing ioProgram / grbProgram entrypoints ---
int main( int argc, char ** argv ) {
    std::cout << "simulated_anealing_re runner\n";
    input in;
    output out;

    if ( !parse_arguments( in, argc, argv ) ) {
        printhelp( argv[0] );
        return 1;
    }


    std::cout << "seed=" << in.seed << " n_replicas=" << in.n_replicas << " nsweeps=" << in.nsweeps << " sweep=ising_sweep" << "\n";

    // Run IO program (populates Storage or similar)
    {
        bool success = false;
        grb::Launcher< AUTOMATIC > launcher;
        grb::RC rc = launcher.exec( &ioProgram, in, success, true );
        if ( rc != SUCCESS ) {
            std::cerr << "I/O launcher failed: " << toString(rc) << "\n";
            return 2;
        }
        if ( !success ) {
            std::cerr << "I/O program reported failure\n";
            return 3;
        }
    }

    // Run main GraphBLAS program that builds data and calls reSA stub
    {
        grb::Launcher< AUTOMATIC > launcher;
        grb::RC rc = launcher.exec( &grbProgram, in, out, true );
        if ( rc != SUCCESS ) {
            std::cerr << "grbProgram launcher failed: " << toString(rc) << "\n";
            return 4;
        }
    }

    std::cout << "Finished: error_code=" << out.error_code << " iterations=" << out.rep << " best_energy=" << out.best_energy << "\n";
    return out.error_code;
}
