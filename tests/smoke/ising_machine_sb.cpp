
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
	const std::vector<std::pair< std::pair< size_t, size_t >, JType > > j_matrix_data = {
		{{1, 1}, -1}, {{2, 2}, -1}, {{3, 1}, 1}, {{3, 3}, -1},
		{{4, 1}, 1}, {{4, 4}, 1}, {{5, 1}, -1}, {{5, 2}, -1},
		{{5, 3}, -1}, {{5, 4}, 1}, {{5, 5}, 1},
		{{6, 1}, -1	}, {{6, 2}, 1}, {{6, 4}, 1},
		{{6, 6}, -1}, {{7, 2}, 1}, {{7, 3}, -1},
		{{7, 4}, -1}, {{7, 6}, -1},
		{{7, 7}, -1}, {{8, 2}, 1},
		{{8, 4}, -1}, {{8, 8}, -1},
		{{9, 4}, 1}, {{9, 6}, -1},
		{{9, 8}, 1}, {{9, 9}, 1},
		{{10, 2}, 1}, {{10, 3}, -1},
		{{10, 4}, 1}, {{10, 8}, -1},
		{{10, 10}, 1}
		,
		// since matrix is symmetric, we can add the symmetric entries
		{{1, 3}, 1}, 
		{{1, 4}, 1}, {{1, 5}, -1}, {{2, 5}, -1},
		{{3, 5}, -1}, {{4, 5}, 1},
		{{1, 6}, -1	}, {{2, 6}, 1}, {{4, 6}, 1},
		{{2, 7}, 1}, {{3, 7}, -1},
		{{4, 7}, -1}, {{6, 7}, -1},
		{{2, 8}, 1},
		{{4, 8}, -1},
		{{4, 9}, 1}, {{6, 9}, -1},
		{{8, 9}, 1},
		{{2, 10}, 1}, {{3, 10}, -1},
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
			// print last data element
			std::cout << "read_matrix_data_from_file: " << data.back().first.first << ", "
				<< data.back().first.second << ", " << data.back().second << "\n";
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
			//std::cout << "read_matrix_data_from_array: " << entry.first.first << ", " << entry.first.second << ", " << entry.second << "\n";
			// since 			
			// data.push_back( Dtype( *it ) );
			// print last data element
			// std::cout << "read_matrix_data: " << it->first.first << ", " << it->first.second << ", " << it->second << "\n";
			// we want the same here
			data.emplace_back(
				NonzeroT( entry.first.first-1, entry.first.second-1, entry.second )
			);
			// print last data element from std::vector<NonzeroT> data
			std::cout << "read_matrix_data_from_array: " << data.back().first.first << ", "
				<< data.back().first.second << ", " << data.back().second << "\n";
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

    if(data_in.use_default_data){
        // if no file provided, use default data from file_content
		// TODO: for now use matrix data file
		//read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
		read_matrix_data_from_array<NonzeroT>( test_data::j_matrix_data, Jdata );
        read_vector_data_from_array<JType>( test_data::h_array_data, h );
        read_vector_data_from_array<IOType>( test_data::x_array_data, x );
        read_vector_data_from_array<IOType>( test_data::y_array_data, y );

    } else {
        // read from files if provided
        read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
        read_vector_data<JType>( data_in.filename_h, h );
        read_vector_data<IOType>( data_in.filename_x, x );
        read_vector_data<IOType>( data_in.filename_y, y );
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
    const size_t n = std::get<0>(Storage::getData());
	std::cout << "n = " << n << std::endl;
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

		// print matrix
		if( s == 0 ) {
			std::cout << "Matrix J:\n";
			print_matrix( J);
		}
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
    // TODO: enable sol_ref
    // grb::Vector< JType > sol_ref( N );
    // rc = rc ? rc : buildVector(sol_ref, sol_ref_data, sol_ref_data + N, grb::SEQUENTIAL);

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

    // TODO: enable sol_ref
	// out.pinnedRefSolutionVector = std::unique_ptr< PinnedVector< JType > >(
	// 	new PinnedVector< JType >( sol_ref, SEQUENTIAL ) );

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
//#ifdef DEBUG_IMSB
           std::cout << "Energy at iteration " << i << " = " << energies[i] << '\n';
//#endif
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
    in.filename_Jmatrix = "/home/d/Scratch/SA/ising_machine.mtx";
    in.filename_h = "/home/d/Scratch/SA/h_vector.dat";
    in.filename_x = "/home/d/Scratch/SA/x_vector.dat";
    in.filename_y = "/home/d/Scratch/SA/y_vector.dat";
    in.use_default_data = true; // use default data from files

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
            // TODO: enable sol_ref
            // print_vector( solution_ref, 30, "EXPECTED SOLUTION" );
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
