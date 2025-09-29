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
#include <cassert>

#include <graphblas/nonzeroStorage.hpp>
#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/iterators/nonzeroIterator.hpp>
#include <utils/output_verification.hpp>
#include <graphblas.hpp>
#include <utils/print_vec_mat.hpp>
#include <random>

using namespace grb;

#define DEBUG_IMSB 1

// Types
using IOType = double;   // scalar/vector element type
using JType  = double;   // coupling (matrix) value type

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
        unsigned,                  // seed
        std::string,               // sweep_name
        std::vector<NonzeroT>,     // matrix data
        std::vector<JType>         // h vector
    >
> Storage;

namespace test_data {
    constexpr size_t n = 16;
    constexpr size_t n_replicas = 3;
    constexpr size_t nsweeps = 2;
    constexpr unsigned seed = 8;

    const std::vector< std::pair< std::pair< grb::config::RowIndexType, grb::config::ColIndexType >, JType > > j_matrix_data = {
        {{0, 1}, -0.27523006},
        {{1, 0}, -0.27523006},
        {{1, 2},  0.28977992},
        {{2, 1},  0.28977992},
        {{2, 3}, -0.15453839},
        {{3, 2}, -0.15453839},
        {{3, 4},  0.48474944},
        {{3, 5}, -0.61958321},
        {{4, 3},  0.48474944},
        {{4, 5}, -0.11904111},
        {{5, 3}, -0.61958321},
        {{5, 4}, -0.11904111},
        {{5, 6},  0.70296404},
        {{6, 5},  0.70296404},
        {{7, 8}, -0.18031020},
        {{8, 7}, -0.18031020},
        {{9, 10}, 0.13823881},
        {{10, 9}, 0.13823881}
    };

    const size_t nnz = j_matrix_data.size();

    const std::vector< JType > h_array_data = {
        0.03076145, -0.06152290, 0.09228435, -0.12304580,
        0.15380725, -0.18456870, 0.21533015, -0.24609160,
        0.27685305, -0.30761450, 0.33837595, -0.36913740,
        0.39989885, -0.43066030, 0.46142175, -0.49218320
    };
}
// --- New, minimal runner configuration and result types ---
struct input {
    bool use_default_data = false;
    std::string filename_Jmatrix;
    std::string filename_h;
    size_t n_replicas = 3;
    size_t nsweeps = 2;
    unsigned seed = 8;
    std::string sweep_name = "sequential_sweep_immediate";
    bool verify = false;
    std::string filename_ref_solution;
	bool direct;
    size_t rep = 0;
    size_t outer = 1;
};

struct output {
    int error_code = 0;
    // TODO: remove itrations if not applicable
    size_t iterations = 0; // total number of iterations performed does not make sense since the code does not have convergence criteria
    double best_energy = 0.0;
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
#ifdef DEBUG_IMSB
			// print last data element from std::vector<NonzeroT> data
			std::cout << "read_matrix_data: " << data.back().first.first << ", "
				<< data.back().first.second << ", " << data.back().second << "\n";
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
#ifdef DEBUG_IMSB
            // print last data element from std::vector<NonzeroT> data
            std::cout << "read_matrix_data_from_array: " << data.back().first.first << ", "
                << data.back().first.second << ", " << data.back().second << "\n";
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


void ioProgram( const struct input &data_in, bool &success ) {

    using namespace test_data;
	success = false;
	// Parse and store matrix in singleton class
    // Map Storage tuple fields to meaningful names and wire up default data
    auto &storage = Storage::getData();
    auto &n           = std::get<0>(storage); // n (rows/cols)
    auto &nnz         = std::get<1>(storage); // nz (nonzeros)
    auto &nsweeps_st  = std::get<2>(storage); // nsweeps
    auto &n_replicas_st = std::get<3>(storage); // n_replicas
    auto &seed_st     = std::get<4>(storage); // seed
    auto &sweep_name  = std::get<5>(storage); // sweep_name
    auto &Jdata       = std::get<6>(storage); // std::vector<NonzeroT>
    auto &h           = std::get<7>(storage); // std::vector<JType>

    // Initialize metadata from input (allow CLI to override defaults)
    nsweeps_st    = data_in.nsweeps;
    n_replicas_st = data_in.n_replicas;
    seed_st       = data_in.seed;
    sweep_name    = data_in.sweep_name;

    if ( data_in.use_default_data ) {
        // if no file provided, use default data from file_content
        read_matrix_data_from_array<NonzeroT>( test_data::j_matrix_data, Jdata );
        read_vector_data_from_array<JType>( test_data::h_array_data, h );
        // other data
    } else {
        // read from files if provided
        read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
        read_vector_data<JType>( data_in.filename_h, h );
		if(data_in.verify) {
			if(data_in.filename_ref_solution.empty()) {
				std::cerr << "Reference solution file not provided for verification\n";
				return;
			}
		}
		//read_vector_data<JType>( data_in.filename_ref_solution, sol );

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
	std::cout << "problem size n = " << n << "\n";
    grb::Vector<JType> h( n );
    // populate J with test (random) values
    grb::RC rc = grb::SUCCESS;

    // load into GraphBLAS
    grb::Matrix<JType> J( n, n );
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

#ifdef DEBUG_IMSB
	if( s == 0 ) {
		std::cout << "Matrix J:\n";
		print_matrix( J);
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

    // create states storage and initialize with random 1/0 values
    const size_t n_replicas = std::get<3>(Storage::getData());
    std::vector< grb::Vector<IOType> > states;
    for ( size_t r = 0; r < n_replicas; ++r ) {
        states.emplace_back( grb::Vector<IOType>(n) );
        // initialize with random values
        std::default_random_engine generator( std::get<4>(Storage::getData()) + r );
        std::uniform_int_distribution<int> distribution(0,1);
        // we use buildvectorUnique with a random set of indices
        std::vector< IOType > rand_data;
        for ( size_t i = 0; i < n; ++i ) {
            rand_data.emplace_back( static_cast<IOType>(
                distribution(generator) ) );
        }
        rc = rc ? rc : grb::buildVector(
            states.back(),
            rand_data.cbegin(),
            rand_data.cend(),
            SEQUENTIAL
        );
    }

    #ifdef DEBUG_IMSB
    if( s == 0 ) {
        for ( size_t r = 0; r < n_replicas; ++r ) {
            std::cout << "Initial state replica " << r << ":\n";
            print_vector( states[r], 30 ,"states values" );  
            std::cout << std::endl;

        }
    }
    #endif


    // also make betas vector os size n_replicas and initialize with 10.0
    grb::Vector<IOType> betas( n_replicas );
    for ( size_t r = 0; r < n_replicas; ++r ) {
        rc = rc ? rc : grb::setElement( betas, static_cast<IOType>(10.0), r );
    }
    rc = rc ? rc : wait();

    // also make energies vector os size n_replicas and calculate values
    // in python energies = np.array([get_energy(couplings, local_fields, state) for state in states])
    // will be initalize in the algorithm
    grb::Vector<IOType> energies( n_replicas );

    // all temporary vectors and matrices should be created here

    // TODO: add times

	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		// rc = simulated_annealing_RE(
        //     energies, states, J, h, ... other params ... ,
        //     .. temp args, sol, out.iterations
        // );

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
		// do benchmark
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( rc == SUCCESS ) {
                // rc = simulated_annealing_RE(
                //     energies, states, J, h, ... other params ... ,
                //     .. temp args, sol, out.iterations
                // );
			}
			if( grb::Properties<>::isNonblockingExecution ) {
				rc = rc ? rc : wait();
			}
		}
		const double time_taken = timer.time();
		out.times.useful = time_taken / static_cast< double >( out.rep );
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for " << out.rep << " "
				<< "Simulated Annealing RE calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << grb::toString( rc ) << std::endl;
			std::cout << "\tnumber of IM-SB iterations: " << out.iterations << "\n";
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
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}



}


// --- Simple help / CLI parser for the new runner (no backward compatibility) ---
void printhelp( char *progname ) {
    std::cout << "Usage: " << progname << " [--use-default-data] [--j-matrix-fname STR] [--h-fname STR]\n"
              << "       [--n-replicas INT] [--nsweeps INT] [--seed INT] [--sweep STR]\n"
              << "       [--verify] [--ref-solution-fname STR] [--help]\n\n"
              << "Options:\n"
              << "  --use-default-data         Use embedded default test data\n"
              << "  --j-matrix-fname STR       Path to J matrix file (matrix-market or supported)\n"
              << "  --h-fname STR              Path to h (local fields) vector (whitespace separated)\n"
              << "  --n-replicas INT           Number of replicas (default: 3)\n"
              << "  --nsweeps INT              Number of sweeps (default: 2)\n"
              << "  --seed INT                 RNG seed (default: 8)\n"
              << "  --sweep STR                Sweep selector (default: sequential_sweep_immediate)\n"
              << "  --verify                   Verify output against reference solution\n"
              << "  --ref-solution-fname STR   Reference solution file (required with --verify unless using default data)\n"
              << "  --help, -h                 Print this help message\n";
}

bool parse_arguments( input &in, int argc, char ** argv ) {
    in.filename_Jmatrix.clear();
    in.filename_h.clear();
    in.filename_ref_solution.clear();
    in.direct = true;
    // map benchmarking configuration to the runner's fields
    in.rep = grb::config::BENCHMARKING::inner();
    in.outer = grb::config::BENCHMARKING::outer();
    // keep verify default (false) unless overridden via CLI
    in.verify = false;

    for ( int i = 1; i < argc; ++i ) {
        std::string a = argv[i];
        if ( a == "--use-default-data" ) {
            in.use_default_data = true;
        } else if ( a == "--j-matrix-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--j-matrix-fname requires an argument\n"; return false; }
            in.filename_Jmatrix = argv[++i];
        } else if ( a == "--h-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--h-fname requires an argument\n"; return false; }
            in.filename_h = argv[++i];
        } else if ( a == "--n-replicas" ) {
            if ( i+1 >= argc ) { std::cerr << "--n-replicas requires an argument\n"; return false; }
            in.n_replicas = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--nsweeps" ) {
            if ( i+1 >= argc ) { std::cerr << "--nsweeps requires an argument\n"; return false; }
            in.nsweeps = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--seed" ) {
            if ( i+1 >= argc ) { std::cerr << "--seed requires an argument\n"; return false; }
            in.seed = static_cast<unsigned>( std::stoul(argv[++i]) );
        } else if ( a == "--sweep" ) {
            if ( i+1 >= argc ) { std::cerr << "--sweep requires an argument\n"; return false; }
            in.sweep_name = argv[++i];
        } else if ( a == "--verify" ) {
            in.verify = true;
        } else if ( a == "--ref-solution-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--ref-solution-fname requires an argument\n"; return false; }
            in.filename_ref_solution = argv[++i];
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
        if ( in.filename_Jmatrix.empty() || in.filename_h.empty() ) {
            std::cerr << "Either --use-default-data or both --j-matrix-fname and --h-fname must be provided\n";
            return false;
        }
    }
    if ( in.verify && !in.use_default_data && in.filename_ref_solution.empty() ) {
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

    // seed RNGs (C and C++ engines) using requested seed (hardcoded default 8 if not provided)
    std::srand( static_cast<unsigned>( in.seed ) );
    static std::mt19937 global_rng( static_cast<unsigned>( in.seed ) );

    std::cout << "seed=" << in.seed << " n_replicas=" << in.n_replicas << " nsweeps=" << in.nsweeps << " sweep=" << in.sweep_name << "\n";

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

    std::cout << "Finished: error_code=" << out.error_code << " iterations=" << out.iterations << " best_energy=" << out.best_energy << "\n";
    return out.error_code;
}