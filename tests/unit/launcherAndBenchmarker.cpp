
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
 * @file
 *
 * Tests the grb::Launcher abstraction.
 *
 * @author Alberto Scolari
 * @date August 2023
 */


#include <iostream>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <limits>

#include <string.h>
#include <stdio.h>
#ifdef DISTRIBUTED_EXECUTION
	#include <mpi.h>
#endif

#include <graphblas.hpp>
#include <graphblas/utils/ranges.hpp>


#ifdef NO_LPF_AUTO_INIT
 const int LPF_MPI_AUTO_INITIALIZE = 0;
#endif

constexpr size_t STR_LEN = 1024;

static const char truth[ STR_LEN + 1 ] = "Night is worn,\n"
	"and the morn\n"
	"rises from the slumberous mass.";

struct input {
	char str[ STR_LEN + 1 ];
};

bool operator==( const struct input &obj, const char * ext ) {
	return strnlen( obj.str, STR_LEN + 1 ) == strnlen( ext, STR_LEN + 1 ) &&
		strncmp( obj.str, ext, STR_LEN + 1 ) == 0;
}

bool operator==( const char * ext, const struct input &obj ) {
	return obj == ext;
}

struct output {
	int exit_code;
	grb::utils::TimerResults times;
};

void grbProgram( const struct input &in, struct output &out ) {
	out.times.preamble = 2.0;
	out.times.useful = 2.0;
	out.times.io = out.times.postamble = 2.0;
	out.times.postamble = 2.0;

	const size_t P = grb::spmd<>::nprocs();
	const size_t s = grb::spmd<>::pid();
	out.exit_code = in == truth ? 0 : 1;
	if( out.exit_code == 0 ) {
		std::cout << "PID " << s << " of " << P << ": match, string is\n"
			<< "\"" << in.str << "\"\n";
	} else {
		std::cout << "PID " << s << " of " << P << ": ERROR!\n"
			<< "\"" << in.str << "\"\n"
			<< "!=\n"
			<< "\"" << truth << "\"\n";
	}
}

void vgrbProgram( const void* __in, const size_t, struct output &out ) {
	const struct input &in = *reinterpret_cast< const struct input *>( __in );
	return grbProgram( in, out );
}

class Runner {

	public:

	virtual grb::RC launch_typed(
		grb::AlpTypedFunc< input, output >,
		const input &, output &,
		bool
	) = 0;

	virtual grb::RC launch_untyped(
		grb::AlpUntypedFunc< void, output >,
		const void *, size_t,
		output &,
		bool
	) = 0;

	virtual grb::RC finalize() = 0;

	virtual ~Runner() = default;

};

template< grb::EXEC_MODE mode >
class bsp_launcher :
	public grb::Launcher< mode >, public Runner
{
	public:

		using grb::Launcher< mode >::Launcher;

		grb::RC launch_typed(
			grb::AlpTypedFunc< input, output > grbProgram,
			const input &in, output &out, bool bc
		) override {
			return this->exec( grbProgram, in, out, bc );
		}

		grb::RC launch_untyped(
			grb::AlpUntypedFunc< void, output > grbProgram,
			const void * in, size_t in_size,
			output &out, bool bc
		) override {
			return this->exec( grbProgram, in, in_size, out, bc );
		}

		virtual grb::RC finalize() override {
			return grb::Launcher< mode >::finalize();
		}

};

template< grb::EXEC_MODE mode >
class bsp_benchmarker :
	public grb::Benchmarker< mode >, public Runner
{

	private:

		size_t inner = 2;
		size_t outer = 2;

	public:

		using grb::Benchmarker< mode >::Benchmarker;

		grb::RC launch_typed(
			grb::AlpTypedFunc< input, output > grbProgram,
			const input &in, output &out,
			bool bc
		) override {
			return this->exec( grbProgram, in, out, bc, inner, outer );
		}

		grb::RC launch_untyped(
			grb::AlpUntypedFunc< void, output > grbProgram,
			const void * in, size_t in_size,
			output &out, bool bc
		) override {
			return this->exec( grbProgram, in, in_size, out, bc, inner, outer );
		}

		virtual grb::RC finalize() override {
			return grb::Benchmarker< mode >::finalize();
		}

};


enum RunnerType { Launch, Benchmark };

std::unique_ptr< Runner > make_runner(
	grb::EXEC_MODE mode, RunnerType type,
	size_t s, size_t P,
	const std::string &host, const std::string &port,
	bool mpi_inited
) {
	Runner *ret = nullptr;
#ifndef DISTRIBUTED_EXECUTION
	( void ) mpi_inited;
#endif

	switch (type) {
		case Launch:
			switch (mode) {
				case grb::AUTOMATIC:
					ret = new bsp_launcher< grb::AUTOMATIC >;
					break;
#ifdef DISTRIBUTED_EXECUTION
				case grb::FROM_MPI:
					ret = new bsp_launcher< grb::FROM_MPI >( MPI_COMM_WORLD );
					break;

				case grb::MANUAL:
					ret = new bsp_launcher< grb::MANUAL >( s, P, host, port, mpi_inited );
					break;
#else
				case grb::MANUAL:
					ret = new bsp_launcher< grb::MANUAL >( s, P, host, port );
					break;
#endif
				default:
					break;
			}
			break;

		case Benchmark:
			switch (mode) {
				case grb::AUTOMATIC:
					ret = new bsp_benchmarker< grb::AUTOMATIC >;
					break;
#ifdef DISTRIBUTED_EXECUTION
				case grb::FROM_MPI:
					ret = new bsp_benchmarker< grb::FROM_MPI >( MPI_COMM_WORLD );
					break;

				case grb::MANUAL:
					ret = new bsp_benchmarker< grb::MANUAL >( s, P, host, port, mpi_inited );
					break;
#else
				case grb::MANUAL:
					ret = new bsp_benchmarker< grb::MANUAL >( s, P, host, port );
					break;
				case grb::FROM_MPI:
#endif

				default:
					break;
			}
			break;

		default:
			break;
	}

	if( ret == nullptr ) {
		throw std::runtime_error( "something went wrong while creating runner" );
	}
	return std::unique_ptr< Runner >( ret );
}

#define ERROR_ON( cond, str ) if( cond ) {                                          \
		std::cerr << __FILE__ ", " << __LINE__ << ": " << str << std::endl; \
		std::cout << "Test FAILED\n" << std::endl;                          \
		return EXIT_FAILURE;                                                \
	}

int main( int argc, char ** argv ) {

	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

#ifdef DISTRIBUTED_EXECUTION
	int lpf_mpi_inited = 0;
	int success = MPI_Initialized( &lpf_mpi_inited );
	ERROR_ON( success != MPI_SUCCESS, "cannot determine initalization info" );
#endif
	const char * host = nullptr;
	const char * port = nullptr;
#ifdef DISTRIBUTED_EXECUTION
	typedef lpf_pid_t test_pid_t;
#else
	typedef size_t test_pid_t;
#endif
	test_pid_t P = std::numeric_limits< test_pid_t >::max();
	test_pid_t s = std::numeric_limits< test_pid_t >::max();
	grb::EXEC_MODE mode = grb::FROM_MPI;

#ifdef DISTRIBUTED_EXECUTION
	if( lpf_mpi_inited != 0 ) {
		mode = grb::AUTOMATIC;
		ERROR_ON( argc != 1, "no argument needed" );
	} else {
		if( argc == 1 ) {
			mode = grb::FROM_MPI;
		} else if( argc == 5 ) {
			mode = grb::MANUAL;
		} else {
			ERROR_ON( true, "either no arguments or four arguments expected.\n"
			       "For the four-argument variant, the following are expected:\n"
				" - hostname\n"
				" - portname\n"
				" - total number of processes\n"
				" - unique ID of this process\n"
			);
		}
	}
#else
	if( argc == 1 ) {
		mode = grb::AUTOMATIC;
	} else if( argc == 5 ) {
		mode = grb::MANUAL;
	} else {
		ERROR_ON( true, "either no arguments or four arguments expected.\n"
				"For the four-argument variant, the following are expected:\n"
			" - hostname\n"
			" - portname\n"
			" - total number of processes\n"
			" - unique ID of this process\n"
		);
	}
#endif
	const char *mode_str = nullptr;

	switch ( mode )
	{
	case grb::AUTOMATIC:
		mode_str = "AUTOMATIC";
		break;
#ifdef DISTRIBUTED_EXECUTION
	case grb::FROM_MPI:
		mode_str = "FROM_MPI";
		break;
#endif
	case grb::MANUAL:
		mode_str = "MANUAL";
		break;
	default:
		ERROR_ON( true, "unrecognized option: " << mode );
		break;
	}

	std::cout << "\n===> chosen initialization method: " << mode_str << " <==="
		<< std::endl;

	if( mode == grb::MANUAL ) {
		// read command-line args
		host = argv[ 1 ];
		port = argv[ 2 ];
		try {
			P = static_cast< test_pid_t >( std::stoi( argv[ 3 ] ) );
			s = static_cast< test_pid_t >( std::stoi( argv[ 4 ] ) );
		} catch( std::exception &e ) {
			std::cerr << "Caught exception: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return EXIT_FAILURE;
		}

		// input sanity checks
		ERROR_ON( host == nullptr || strlen( host ) == 0,
			"Invalid hostname: " << argv[ 1 ] );
		ERROR_ON( port == nullptr || strlen( port ) == 0,
			"value for port name or number: " << argv[ 2 ] );
		ERROR_ON( !grb::utils::is_in_normalized_range( s, P ),
			"Invalid value for PID: " << argv[ 4 ] );
	}
#ifdef DISTRIBUTED_EXECUTION
	if( mode == grb::FROM_MPI || mode == grb::MANUAL ) {
		success = MPI_Init( NULL, NULL );
		ERROR_ON( success != MPI_SUCCESS, "Call to MPI_Init failed" );
	}
#endif

	struct input in;
	struct output out;
	out.exit_code = 0;
	
	for( const bool broadcast : { true, false } ) {
		for( const RunnerType rt : {Launch, Benchmark } ) {
			// const bool broadcast = true;
			const char * const runner_name = rt == Launch ? "Launch" : "Benchmark";
			const char * const bc_str = broadcast ? "true" : "false";
			std::cout << "\n ==> runner type: " << runner_name << ", broadcast: " << bc_str << std::endl;
			std::unique_ptr< Runner > runner;
			try {
				runner = make_runner(
					mode, rt, s, P,
					std::string( (host != nullptr ? host : "" ) ),
					std::string( (port != nullptr ? port : "" ) ),
					true
				);
			} catch( std::runtime_error &e ) {
				std::cerr << "got a runtime exception: " << e.what() << std::endl;
				std::cout << "Test FAILED\n" << std::endl;
				return EXIT_FAILURE;
			} catch( std::exception &e ) {
				std::cerr << "got an exception: " << e.what() << std::endl;
				std::cout << "Test FAILED\n" << std::endl;
				return EXIT_FAILURE;
			} catch( ... ) {
				std::cerr << "got an unknown exception" << std::endl;
				std::cout << "Test FAILED\n" << std::endl;
				return EXIT_FAILURE;
			}

			std::cout << "  => untyped call\n" << std::endl;
			(void) strncpy( in.str, truth, STR_LEN + 1 );
			grb::RC ret = runner->launch_untyped(
				&vgrbProgram,
				reinterpret_cast< void * >( &in ), sizeof( struct input ),
				out, broadcast
			);
			ERROR_ON( ret != grb::SUCCESS,
				"untyped test with broadcast FAILED with code: " << grb::toString( ret ) );
			ERROR_ON( out.exit_code != 0,
				"untyped test with broadcast FAILED with exit code " << out.exit_code );

			std::cout << "\n  => typed call\n" << std::endl;
			ret = runner->launch_typed( &grbProgram, in, out, broadcast );
			ERROR_ON( ret != grb::SUCCESS,
				"typed test with broadcast FAILED with code: " << grb::toString( ret ) );
			ERROR_ON( out.exit_code != 0,
				"typed test with broadcast FAILED with exit code " << out.exit_code );

			ret = runner->finalize();
			
			ERROR_ON( ret != grb::SUCCESS,
				"finalization FAILED with code: " << grb::toString( ret ) );
			std::cout << "  => OK" << std::endl;
		}
	}
#ifdef DISTRIBUTED_EXECUTION
	if( mode == grb::FROM_MPI || mode == grb::MANUAL ) {
		success = MPI_Finalize();
		ERROR_ON( success != MPI_SUCCESS, "Call to MPI_Finalize failed" );
	}
#endif

	std::cout << "\nTest OK\n" << std::endl;
	return EXIT_SUCCESS;
}

