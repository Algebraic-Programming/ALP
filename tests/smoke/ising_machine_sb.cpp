
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
#include <cmath>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/algorithms/ising_machine_sb.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

#include <utils/print_vec_mat.hpp>
#include <utils/output_verification.hpp>

using namespace grb;
using namespace algorithms;

using IOType = double;
using JType = int;

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
        std::vector<NonzeroT>,     // matrix data
        std::vector<JType>,        // h vector
        std::vector<IOType>,       // x vector
        std::vector<IOType>,       // y vector
        std::vector<JType>        // sol_ref vector
        // Add more types as needed
    >
> Storage;
// Access using std::get
// auto& n = std::get<0>(Storage::getData());
// auto& nz = std::get<1>(Storage::getData());
// auto& matrix_data = std::get<2>(Storage::getData());

namespace test_data {
    // test data from python implementation
	constexpr const std::size_t n = 10;
	
	const std::vector<JType> h_array_data = { 1, -1, 1, -1, 1, 1, -1, 1, 1, 1 };
	const std::vector<IOType> x_array_data = {
		-0.0996, -0.0315, 0.0572, 0.0630, 0.0087,
		-0.0143, -0.0170, -0.0411, 0.0433, -0.0298
	};
	const std::vector<IOType> y_array_data = {
		0.0373, 0.0540, 0.0486, -0.0877, -0.0418,
		-0.0261, 0.0018, -0.0710, 0.0507, -0.0483
	};
	const std::vector<JType> sol_ref_data = { 1, -1, 1, 1, 1, -1, -1, -1, 1, 1 };
	// matrix in format of list of nested pairs ((i, j), value)
	const std::vector<std::pair< std::pair< size_t, size_t >, JType > > j_matrix_data = {
		{{1, 1}, -1}, {{2, 2}, -1}, {{3, 1}, 1}, {{3, 3}, -1},
		{{4, 1}, 1}, {{4, 4}, 1}, {{5, 1}, -1}, {{5, 2}, -1},
		{{5, 3}, -1}, {{5, 4}, 1}, {{5, 5}, 1},
		{{6, 1}, -1	}, {{6, 2}, 1}, {{6, 4}, 1},
		{{6, 6}, -1}, {{7, 2}, 1}, {{7, 3}, -1},
		{{7, 4}, -1}, {{7, 6}, -1}, {{7, 7}, -1}, {{8, 2}, 1},
		{{8, 4}, -1}, {{8, 8}, -1}, {{9, 4}, 1}, {{9, 6}, -1},
		{{9, 8}, 1}, {{9, 9}, 1}, {{10, 2}, 1}, {{10, 3}, -1},
		{{10, 4}, 1}, {{10, 8}, -1}, {{10, 10}, 1},
		// since matrix is symmetric, we add the symmetric entries
		{{1, 3}, 1}, {{1, 4}, 1}, {{1, 5}, -1}, {{2, 5}, -1},
		{{3, 5}, -1}, {{4, 5}, 1}, {{1, 6}, -1	}, {{2, 6}, 1}, {{4, 6}, 1},
		{{2, 7}, 1}, {{3, 7}, -1}, {{4, 7}, -1}, {{6, 7}, -1},
		{{2, 8}, 1}, {{4, 8}, -1}, {{4, 9}, 1}, {{6, 9}, -1},
		{{8, 9}, 1}, {{2, 10}, 1}, {{3, 10}, -1},
		{{4, 10}, 1}, {{8, 10}, -1}
	};

    constexpr const std::size_t max_iters = 100;
    const std::size_t num_iters = 100;
    static const double energies_ref[ num_iters ] = { 
        -3,  -3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
        -1,  -9, -11, -11, -11, -11, -11, -11, -11, -11,  -9,  -9, -11,
        -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
        -9,  -9, -11, -11, -11,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
        -9,  -9,  -9,  -9,  -9,  -9,  -9, -11, -11, -11, -11, -11, -11,
        -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
        -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
        -11, -11, -11, -11, -11, -11, -11, -11, -11
    };

    const IOType p0  = 0.;
    const IOType p1  = 1.1;
    const IOType dt  = 0.25;

} // namespace test_data

struct input {
	// contains the command line arguments related data
	static bool use_default_data;
	std::string filename_Jmatrix;
	std::string filename_h;
	std::string filename_x;
	std::string filename_y;
	bool direct;
	size_t rep;
	size_t outer; // number of outer repetitions for benchmarking
	// number of iterations for Ising machine SB
	size_t num_iters;
	// p0, p1, dt parameters for Ising machine SB
	IOType p0;
	IOType p1;
	IOType dt;
	// whether to verify output against reference data
	bool verify;
	// filename of reference solution vector (not implemented)
	std::string filename_ref_solution;
};

bool input::use_default_data = false;

struct output {
	int error_code = 0;
	size_t rep;
	size_t iterations;
	grb::utils::TimerResults times;
    std::unique_ptr< PinnedVector< JType > > pinnedSolutionVector;
    std::unique_ptr< PinnedVector< JType > > pinnedRefSolutionVector;
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
				NonzeroT( entry.first.first-1, entry.first.second-1, entry.second )
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
	auto &Jdata = std::get<2>(Storage::getData());
    // Read and store h vector in singleton class
    auto &h = std::get<3>(Storage::getData());
    auto &x = std::get<4>(Storage::getData());
    auto &y = std::get<5>(Storage::getData());
	auto &sol = std::get<6>(Storage::getData());

    if(data_in.use_default_data){
        // if no file provided, use default data from file_content
		read_matrix_data_from_array<NonzeroT>( test_data::j_matrix_data, Jdata );
        read_vector_data_from_array<JType>( test_data::h_array_data, h );
        read_vector_data_from_array<IOType>( test_data::x_array_data, x );
        read_vector_data_from_array<IOType>( test_data::y_array_data, y );
        read_vector_data_from_array<JType>( test_data::sol_ref_data, sol );
    } else {
        // read from files if provided
        read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
        read_vector_data<JType>( data_in.filename_h, h );
        read_vector_data<IOType>( data_in.filename_x, x );
        read_vector_data<IOType>( data_in.filename_y, y );
		if(data_in.verify) {
			if(data_in.filename_ref_solution.empty()) {
				std::cerr << "Reference solution file not provided for verification\n";
				return;
			}
		}
		read_vector_data<JType>( data_in.filename_ref_solution, sol );
    }

	success = true;
}

void grbProgram(
    const struct input &data_in, 
    struct output &out
) {
    using namespace test_data;

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

    grb::utils::Timer timer;
	timer.reset();

    /* --- Problem setup --- */
    const size_t n = std::get<0>(Storage::getData());
	std::cout << "problem size n = " << n << "\n";
    grb::Vector<JType> h( n );
    grb::Vector<IOType> x0( n ), y0( n ); // initialy
    // ... populate J with test (random) values
    grb::RC rc = grb::SUCCESS;

    // load into GraphBLAS
    grb::Matrix<JType> J( n, n );
	{
		const auto &data = std::get<2>(Storage::getData());
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
        const auto &h_data = std::get<3>(Storage::getData());
		rc = rc ? rc : buildVector(
			h,
			h_data.cbegin(),
			h_data.cend(),
			SEQUENTIAL
		);
    }

    // build vector x with data from singleton
    {
        const auto &x_data = std::get<4>(Storage::getData());
        rc = rc ? rc : buildVector(
            x0, 
            x_data.cbegin(), 
            x_data.cend(), 
            SEQUENTIAL
        );
    }

    // build vector y with data from singleton
    {
        const auto &y_data = std::get<5>(Storage::getData());
        rc = rc ? rc : buildVector(
            y0, 
            y_data.cbegin(), 
            y_data.cend(), 
            SEQUENTIAL
        );
    }       

    if(rc != grb::SUCCESS) {
        std::cerr << "Vector build failed\n";
        return;
    }

    // energies is array of length num_iters, initialized to 0
    std::vector< IOType > energies( num_iters, 0 );

    grb::Vector< IOType > Jx( n );
    grb::Vector< IOType > temp( n );
    grb::Vector< JType > temp_int( n );
    grb::Vector< bool > mask( n );
    grb::Matrix< JType > J2( n, n );
    rc = rc ? rc : grb::resize( J2, grb::nnz(J) );
    if(rc != grb::SUCCESS) {
        std::cerr << "Matrix resize failed for J2\n";
        return;
    }
    grb::Vector< JType > sol( n );
    grb::Vector< JType > sol_ref( n );
	if(data_in.verify) {
		// build vector sol_ref with data from singleton
		const auto &sol_ref_data = std::get<6>(Storage::getData());
		rc = rc ? rc : buildVector(
			sol_ref,
			sol_ref_data.cbegin(),
			sol_ref_data.cend(),
			grb::SEQUENTIAL
		);
	} 
    if(rc != grb::SUCCESS) {
        std::cerr << "Vector build failed\n";
        return;
    }

	rc = rc ? rc : wait();
	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		rc = bSB(
            energies, x0, y0, J, h, p0, p1, num_iters, dt,
            J2, Jx, temp, temp_int, mask, sol, out.iterations
        );

		rc = rc ? rc : wait();
		double single_time = timer.time();
		if( !(rc == SUCCESS || rc == FAILED) ) {
			std::cerr << "Failure: call to Ising Machine SB did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == FAILED ) {
			std::cout << "Warning: call to Ising Machine SB did not converge\n";
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
					std::cout << "Info: cold Ising Machine SB did not converge within ";
				} else {
					std::cout << "Info: cold Ising Machine SB completed within ";
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
				rc = bSB(
                    energies, x0, y0, J, h, p0, p1, num_iters, dt,
                    J2, Jx, temp, temp_int, mask, sol, out.iterations
                );
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
				<< "Ising Machine SB calls (hot start): " << out.times.useful << ". "
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

	out.pinnedRefSolutionVector = std::unique_ptr< PinnedVector< JType > >(
		new PinnedVector< JType >( sol_ref, SEQUENTIAL ) );

	// output
	out.pinnedSolutionVector = std::unique_ptr< PinnedVector< JType > >(
		new PinnedVector< JType >( sol, SEQUENTIAL ) );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;



    if( rc != grb::SUCCESS ) {
        std::cerr << "bSB returned error code " << rc << '\n';
    } else {
		if (data_in.verify) {
			// print all energies
			for (std::size_t i = 0; i < num_iters; ++i) {
	#ifdef DEBUG_IMSB
			std::cout << "Energy at iteration " << i << " = " << energies[i] << '\n';
	#endif
			if( energies[i] != energies_ref[i]) {
	#ifdef DEBUG_IMSB
				std::cerr << "Error: Energy at iteration " << i << " does not match reference value.\n";
				std::cerr << "Expected: " << energies_ref[i] << ", got: " << energies[i] << '\n';
	#endif
				out.error_code = 40;
				return ;
			}
			}
			std::cout << "All energies match reference values.\n";
			std::cout << "TEST OK\n"; 
		} else {
			std::cout << "No verification performed (verification disabled).\n";
	}
}

    	// set error code
	out.error_code = rc;

}


// supported command line arguments
void printhelp( char *progname ) {
	std::cout << " Use: \n";
	std::cout << progname << " [--use-default-data] [--j-matrix-fname STR] [--h-fname STR] [--x-fname STR] [--y-fname STR] [--no-direct] [--num-iters INT] [--p0 FLOAT] [--p1 FLOAT] [--dt FLOAT] [--test-rep INT] [--test-outer-rep INT] [--verify] [--ref-solution-fname STR]\n";
	std::cout << "\n";
	std::cout << " --use-default-data (no argument): use hardcoded default data from the test_data namespace for internal tests\n";
	// input data parameters (mandatory if --use-default-data is not used)
	std::cout << "\n";
	std::cout << "input data parameters (mandatory if --use-default-data is not used):\n";
	std::cout << " --j-matrix-fname STR: filename of J matrix in matrix market format\n";
	std::cout << " --h-fname STR: filename of h vector, where vector elements are stored line-by-line\n";
	std::cout << " --x-fname STR: filename of x vector, where vector elements are stored line-by-line\n";
	std::cout << " --y-fname STR: filename of y vector, where vector elements are stored line-by-line\n";
	std::cout << " --ref-solution-fname STR: mandatory if --verify is used and --use-default-data is not used. (not implemented) filename of reference solution vector, where vector elements are stored line-by-line\n";
	std::cout << " --no-direct (no argument): disable direct addressing\n";
	// either --use-default-data or input data parameters must be provided
	std::cout << "\n";
	std::cout << "either --use-default-data or input data parameters must be provided\n";
	// solver parameters can be set via command-line arguments or will use defaults if not provided
	std::cout << "\n";
	std::cout << "solver parameters (can be set via command-line arguments or will use defaults):\n";
	std::cout << " --p0 FLOAT: p0 parameter, default " << test_data::p0 << "\n";
	std::cout << " --p1 FLOAT: p1 parameter, default " << test_data::p1 << "\n";
	std::cout << " --dt FLOAT: dt parameter, default " << test_data::dt << "\n";
	std::cout << " --num-iters INT: number of iterations, default " << test_data::num_iters << " \n";
	// performance testing parameters (optional)
	std::cout << "\n";
	std::cout << "performance testing parameters (optional):\n";
	std::cout << " --test-rep INT: consecutive test inner algorithm repetitions, default 1\n";
	std::cout << " --test-outer-rep INT: consecutive test outer (including IO) algorithm repetitions, default 1\n";
	// numerical verification parameters (optional)
	std::cout << "\n";
	std::cout << "numerical verification parameters (optional):\n";
	std::cout << " --verify (no argument): verify output against reference solution\n";
	// other parameters
	std::cout << "\n";
	std::cout << "other parameters:\n";
	std::cout << " --help, -h, -? (no argument): print this help message\n";
}

bool parse_arguments(
	input &in,
	int argc,
	char **argv
) {
	in.filename_Jmatrix.clear();
	in.filename_h.clear();
	in.filename_x.clear();
	in.filename_y.clear();
	in.filename_ref_solution.clear();
	in.direct = true;
	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	// get outer number of iterations
	in.outer = grb::config::BENCHMARKING::outer();
	in.p0 = 0.0;
	in.p1 = 0.0;
	in.dt = 0.0;
	in.num_iters = 0;
	in.verify = false;

	bool jmatrix_set = false, h_set = false, x_set = false, y_set = false, refsol_set = false;

	for( int i = 1; i < argc; ++i ) {
		std::string arg( argv[ i ] );
		if( arg == "--use-default-data" ) {
			in.use_default_data = true;
		} else if( arg == "--j-matrix-fname" ) {
			if( i + 1 < argc ) {
				in.filename_Jmatrix = std::string( argv[ ++i ] );
				in.use_default_data = false;
				jmatrix_set = true;
			} else {
				std::cerr << "--j-matrix-fname requires an argument\n";
				return false;
			}
		} else if( arg == "--h-fname" ) {
			if( i + 1 < argc ) {
				in.filename_h = std::string( argv[ ++i ] );
				in.use_default_data = false;
				h_set = true;
			} else {
				std::cerr << "--h-fname requires an argument\n";
				return false;
			}
		} else if( arg == "--x-fname" ) {
			if( i + 1 < argc ) {
				in.filename_x = std::string( argv[ ++i ] );
				in.use_default_data = false;
				x_set = true;
			} else {
				std::cerr << "--x-fname requires an argument\n";
				return false;
			}
		} else if( arg == "--y-fname" ) {
			if( i + 1 < argc ) {
				in.filename_y = std::string( argv[ ++i ] );
				in.use_default_data = false;
				y_set = true;
			} else {
				std::cerr << "--y-fname requires an argument\n";
				return false;
			}
		} else if( arg == "--no-direct" ) {
			in.direct = false;
		} else if( arg == "--test-rep" ) {
			if( i + 1 < argc ) {
				int r = atoi( argv[ ++i ] );
				if( r < 0 ) {
					std::cerr << "--test-rep requires a non-negative integer argument\n";
					return false;
				}
				in.rep = static_cast< size_t >( r );
			} else {
				std::cerr << "--test-rep requires an argument\n";
				return false;
			}
		} else if( arg == "--test-outer-rep" ) {
			if( i + 1 < argc ) {
				int r = atoi( argv[ ++i ] );
				if( r < 1 ) {
					std::cerr << "--test-outer-rep requires a positive integer argument\n";
					return false;
				}
				in.outer = static_cast< size_t >( r );
			} else {
				std::cerr << "--test-outer-rep requires an argument\n";
				return false;
			}
		} else if( arg == "--p0" ) {
			if( i + 1 < argc ) {
				in.p0 = atof( argv[ ++i ] );
			} else {
				std::cerr << "--p0 requires a FLOAT argument\n";
				return false;
			}
		} else if( arg == "--p1" ) {
			if( i + 1 < argc ) {
				in.p1 = atof( argv[ ++i ] );
			} else {
				std::cerr << "--p1 requires a FLOAT argument\n";
				return false;
			}
		} else if( arg == "--dt" ) {
			if( i + 1 < argc ) {
				in.dt = atof( argv[ ++i ] );
			} else {
				std::cerr << "--dt requires a FLOAT argument\n";
				return false;
			}
		} else if( arg == "--num-iters" ) {
			if( i + 1 < argc ) {
				int niter = atoi( argv[ ++i ] );
				if( niter < 1 ) {
					std::cerr << "--num-iters requires a positive integer argument\n";
					return false;
				}
				in.num_iters = static_cast< size_t >( niter );
			} else {
				std::cerr << "--num-iters requires an argument\n";
				return false;
			}
		} else if( arg == "--verify" ) {
			in.verify = true;
		} else if( arg == "--ref-solution-fname" ) {
			if( i + 1 < argc ) {
				in.filename_ref_solution = std::string( argv[ ++i ] );
				refsol_set = true;
			} else {
				std::cerr << "--ref-solution-fname requires an argument\n";
				return false;
			}
		} else if( arg == "--help" || arg == "-h" || arg == "-?" ) {
			printhelp( argv[ 0 ] );
			return false;
		} else {
			std::cerr << "unknown command line argument \"" << arg << "\"\n";
			return false;
		}
	}

	// Check that either --use-default-data or all input data parameters are provided
	if( !in.use_default_data ) {
		if( !(jmatrix_set && h_set && x_set && y_set) ) {
			std::cerr << "Error: Either --use-default-data or all input data parameters (--j-matrix-fname, --h-fname, --x-fname, --y-fname) must be provided.\n";
			return false;
		}
	}

	// Ensure reference solution filename is provided if verification is requested and not using default data
	if( in.verify && !in.use_default_data ) {
		if( !refsol_set || in.filename_ref_solution.empty() ) {
			std::cerr << "Error: --ref-solution-fname STR is mandatory if --verify is used and --use-default-data is not used.\n";
			return false;
		}
	}

	// set defaults for solver parameters if not set
	if( in.p0 == 0.0 && in.p1 == 0.0 && in.dt == 0.0 ) {
		in.p0 = test_data::p0;
		in.p1 = test_data::p1;
		in.dt = test_data::dt;
	}
	if( in.num_iters == 0 ) {
		in.num_iters = test_data::num_iters;
	}
	return true;
}


void print_cmd_paramaters( const input &in) {
	// print paramerters
	std::cout << "Parameters:\n";
	std::cout << " --use-default-data: " << (in.use_default_data ? "true" : "false") << "\n";
	if( !in.use_default_data ) {
		std::cout << " --j-matrix-fname: " << in.filename_Jmatrix << "\n";
		std::cout << " --h-fname: " << in.filename_h << "\n";
		std::cout << " --x-fname: " << in.filename_x << "\n";
		std::cout << " --y-fname: " << in.filename_y << "\n";
	}
	std::cout << " --no-direct: " << (in.direct ? "false" : "true") << "\n";
	std::cout << " --p0: " << in.p0 << "\n";
	std::cout << " --p1: " << in.p1 << "\n";
	std::cout << " --dt: " << in.dt << "\n";
	std::cout << " --num-iters: " << in.num_iters << "\n";
	std::cout << " --test-rep: " << in.rep << "\n";
	std::cout << " --test-outer-rep: " << in.outer << "\n";
	std::cout << " --verify: " << (in.verify ? "true" : "false") << "\n";
	if( in.verify ) {
		std::cout << " --ref-solution-fname: " << in.filename_ref_solution << "\n";
	}
	std::cout << "\n";
}

int main( int argc, char ** argv ) {
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	input  in;
	output out;

	if( !parse_arguments( in, argc, argv ) ) {
		std::cerr << "error parsing command line arguments\n";
		printhelp( argv[0] );
		return 1;
	}

#ifdef DEBUG_IMSB
	// print paramerters
	print_cmd_paramaters( in );
#endif

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch I/O
	{
		grb::utils::Timer timer;
		timer.reset();

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
		// I/O done
		out.times.io = timer.time();
		timer.reset();
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
		rc = benchmarker.exec( &grbProgram, in, out, 1, in.outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code "
			<< grb::toString( rc ) << std::endl;
		return 90;
	} else if( out.error_code == 0 ) {
		std::cout << "Benchmark completed successfully and took " << out.iterations
			 << ".\n";
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	// inspect output vector
	if( !(out.pinnedSolutionVector) ) {
		std::cerr << "no output vector to inspect" << std::endl;
	} else {
		const PinnedVector< JType > &solution = *(out.pinnedSolutionVector);
        const PinnedVector< JType > &solution_ref = *(out.pinnedRefSolutionVector);
		std::cout << "Size of x is " << solution.size() << std::endl;
		if( solution.size() > 0 ) {
			print_vector( solution, 30, "SOLUTION" );
		} else {
			std::cerr << "ERROR: solution contains no values" << std::endl;
		}
		if( in.verify && (solution_ref.size() > 0) ) {
			print_vector( solution_ref, 30, "REFERENCE SOLUTION" );
		}
	}

	// verify output vector if requested
	if( in.verify ) {
		const PinnedVector< JType > &solution = *(out.pinnedSolutionVector);
        const PinnedVector< JType > &solution_ref = *(out.pinnedRefSolutionVector);
		assert(solution.size() == solution_ref.size());
		assert(solution.size() == std::get<0>(Storage::getData()));
		JType norm2 = 0;
		for( size_t i = 0; i < solution.size(); i++ ) {
			JType diff = solution.getNonzeroValue(i) - solution_ref.getNonzeroValue(i);
			norm2 += diff * diff;
		}
		out.error_code = ( norm2 == 0 ) ? 0 : 50;
		if( out.error_code == 0 ) {
			std::cout << "Output vector verificaton was successful!\n";
			std::cout << "Test OK\n";
		} else {
			std::cerr << std::flush;
			std::cout << "Verification FAILED\n";
			std::cout << "Test FAILED\n";
		}
	} else {
		std::cout << "Output vector verificaton was not requested!\n";
		std::cout << "Test OK\n";
	}	

	// done
	return out.error_code;
}
