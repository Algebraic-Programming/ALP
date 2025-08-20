
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
        std::vector<IOType>        // y vector
        // Add more types as needed
    >
> Storage;

// Access using std::get
// auto& n = std::get<0>(Storage::getData());
// auto& nz = std::get<1>(Storage::getData());
// auto& matrix_data = std::get<2>(Storage::getData());

namespace test_data {
    // test data from python implementation
    constexpr const std::size_t max_iters = 100;

    constexpr std::size_t N = 10;
    // constexpr std::size_t Nz = 54;
    // static const size_t i_arr[ Nz ] = {
    //     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 
    //     4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9
    // };
    // static const size_t j_arr[ Nz ] = { 
    //     0, 2, 3, 4, 5, 1, 4, 5, 6, 7, 9, 0, 2, 4, 6, 9, 0, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 
    //     3, 4, 0, 1, 3, 5, 6, 8, 1, 2, 3, 5, 6, 1, 3, 7, 8, 9, 3, 5, 7, 8, 1, 2, 3, 7, 9
    // };
    // static const int v_arr[ Nz ] = { 
    //     -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  
    //     1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1
    // };

    static const int h_arr[ N ] = { 1, -1,  1, -1,  1,  1, -1,  1,  1,  1 };
    static const double x_arr[ N ] = { -0.0996, -0.0315,  0.0572,  0.0630,  0.0087, -0.0143, -0.0170, -0.0411, 0.0433, -0.0298 };
    static const double y_arr[ N ] = {  0.0373,  0.0540,  0.0486, -0.0877, -0.0418, -0.0261,  0.0018, -0.0710, 0.0507, -0.0483 };

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

    static const int sol_ref_data[ N ] = { 
        1, -1,  1,  1,  1, -1, -1, -1,  1,  1
    };

    const IOType p0  = 0.;
    const IOType p1  = 1.1;
    const IOType dt  = 0.25;

} // namespace test_data

struct input {
    // contains the command line arguments
    std::string filename;
	bool direct;
	size_t rep;
};

struct output {
	int error_code = 0;
	size_t rep;
	size_t iterations;
	grb::utils::TimerResults times;
    std::unique_ptr< PinnedVector< JType > > pinnedSolutionVector;
    std::unique_ptr< PinnedVector< JType > > pinnedRefSolutionVector;
};


void ioProgram( const struct input &data_in, bool &success ) {
	success = false;
	// Parse and store matrix in singleton class
	auto &data = std::get<2>(Storage::getData());
	try {
		Parser parser( data_in.filename, data_in.direct );
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
			data.push_back( NonzeroT( *it ) );
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
    using namespace test_data;

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

    grb::utils::Timer timer;
	timer.reset();
    // TODO: IO goes here

	// I/O done
	out.times.io = timer.time();
	timer.reset();

    /* --- Problem setup --- */
    grb::Vector<JType> h( N );
    grb::Vector<IOType> x0( N ), y0( N ); // initialy 
    // ... populate J with test (random) values
    grb::RC rc = grb::SUCCESS;
    // rc = rc ? rc : buildMatrixUnique( J, &( i_arr[ 0 ] ), &( j_arr[ 0 ] ), &( v_arr[ 0 ] ), Nz, grb::SEQUENTIAL );
    // if(rc != grb::SUCCESS) {
    //     std::cerr << "matrix build failed\n";
    //     return;
    // }


	// load into GraphBLAS
	const size_t n = std::get<0>(Storage::getData());
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
	}

    // Fill h, x0, y0 with random values using buildVector
    rc = rc ? rc : buildVector(h, h_arr, h_arr + N, grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(x0, x_arr, x_arr + N, grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(y0, y_arr, y_arr + N, grb::SEQUENTIAL);
    if(rc != grb::SUCCESS) {
        std::cerr << "Vector build failed\n";
        return;
    }

    // energies is array of length num_iters, initialized to 0
    std::vector< IOType > energies( num_iters, 0 );

    grb::Vector< IOType > Jx( N );
    grb::Vector< IOType > temp( N );
    grb::Vector< JType > temp_int( N );
    grb::Vector< bool > mask( N );
    grb::Matrix< JType > J2( N, N );
    rc = rc ? rc : grb::resize( J2, grb::nnz(J) );
    if(rc != grb::SUCCESS) {
        std::cerr << "Matrix resize failed for J2\n";
        return;
    }
    grb::Vector< JType > sol( N );
    grb::Vector< JType > sol_ref( N );
    rc = rc ? rc : buildVector(sol_ref, sol_ref_data, sol_ref_data + N, grb::SEQUENTIAL);

    out.times.preamble = timer.time();

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
			if( Properties<>::isNonblockingExecution ) {
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
    }

    	// set error code
	out.error_code = rc;

}


int main( int argc, char ** argv ) {
    // TODO: add argument parsing for input file, direct/indirect addressing, etc.
    // for now, just print the executable name
    (void) argc; // unused
    (void) argv; // unused
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
    in.filename = "/home/d/Scratch/SA/ising_machine.mtx";

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();

	// check for verification of the output
	// bool verification = false;

	std::cout << "Executable called with parameters "
		<< "inner repititions = " << in.rep << ", "
		<< "outer reptitions = " << outer << ", "
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
			 << ".\n";
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	if( !(out.pinnedSolutionVector) ) {
		std::cerr << "no output vector to inspect" << std::endl;
	} else {
		const PinnedVector< JType > &solution = *(out.pinnedSolutionVector);
        const PinnedVector< JType > &solution_ref = *(out.pinnedRefSolutionVector);
		std::cout << "Size of x is " << solution.size() << std::endl;
		if( solution.size() > 0 ) {
			print_vector( solution, 30, "SOLUTION" );
            // expected solution from sol_ref_data
            print_vector( solution_ref, 30, "EXPECTED SOLUTION" );
		} else {
			std::cerr << "ERROR: solution contains no values" << std::endl;
		}
	}


	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} 
    
    std::cout << "Test OK\n";

	// done
	return out.error_code;
}
