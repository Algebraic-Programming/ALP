
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

static const char prelude[ STR_LEN + 1 ] = "O Earth O Earth return!\n"
	"Arise from out the dewy grass;";

static const char truth[ STR_LEN + 1 ] = "Night is worn,\n"
	"and the morn\n"
	"rises from the slumberous mass.";

static const char default_str[ STR_LEN + 1 ] = "Hear the voice of the Bard!\n"
	"Who Present, Past, and Future, sees;";

struct input {
	char str[ STR_LEN + 1 ];

	input() {
		(void) strncpy( str, default_str, STR_LEN + 1 );
	}
};

// same as input, just not default-constructible for a testing scenarion
struct nd_input : input {

	nd_input() = delete; // make this non default-constructible

	nd_input( const char * _str ) {
		(void) strncpy( this->str, _str, STR_LEN + 1 );
	}
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
	size_t P;
	grb::utils::TimerResults times;
};

template< grb::EXEC_MODE mode, bool broadcasted, typename InputT >
void grbProgram( const InputT &in, struct output &out ) {
	static_assert( std::is_base_of< input, InputT >::value );
	out.times.preamble = 2.0;
	out.times.useful = 2.0;
	out.times.io = out.times.postamble = 2.0;
	out.times.postamble = 2.0;

	const size_t P = grb::spmd<>::nprocs();
	const size_t s = grb::spmd<>::pid();
	out.P = P;

	const char * expected = nullptr;

	if( broadcasted ) {
		// independently from mode is or process id, every process must have the same
		// string
		expected = truth;
	} else {
		// in non-broadcasting mode, what a process has depends on its rank and the
		// launcher mode.
		switch (mode) {
			case grb::AUTOMATIC:
				// here, only the master process can have the "new" string
				// while the other processes have the "default" string
				expected = s == 0 ? truth : default_str;
				break;
			case grb::FROM_MPI:
			case grb::MANUAL:
				// the master must have the new string, while other processes the prelude
				expected = s == 0 ? truth : prelude;
				break;
			default:
				out.exit_code = 1;
				printf( "- ERROR: unknown mode %d\n", mode );
				return;
				break;
		}
	}
	out.exit_code = in == expected ? 0 : 1;

	std::cout << "--- PID " << s << " of " << P << ": ";
	if( out.exit_code == 0 ) {
		std::cout << "MATCH\n";
	} else {
		std::cout << "ERROR! Input string\n\"" << in.str
			<< "\"\n!= Expected string\n\"" << expected << "\"\n";
	}
}

template< grb::EXEC_MODE mode, bool broadcasted, typename InputT >
void vgrbProgram(
	const void * const __in, const size_t size,
	struct output &out
) {
	if( size != STR_LEN + 1 ) {
		const size_t P = grb::spmd<>::nprocs();
		const size_t s = grb::spmd<>::pid();
		out.P = P;
		std::cout << "--- PID " << s << " of " << P << ": "
			<< "ERROR! Input size " << size << " !- expected " << (STR_LEN+1) << "\n";
		return;
	}
	const struct input &in = *reinterpret_cast< const struct input *>( __in );
	grbProgram< mode, broadcasted, InputT >( in, out );
}

void autoVgrbProgram(
	const void * const __in, const size_t size,
	struct output &out
) {
	const size_t P = grb::spmd<>::nprocs();
	const size_t s = grb::spmd<>::pid();
	out.P = P;
	if( s == 0 ) {
		const input &in = *static_cast< const input * >( __in );
		out.exit_code = size == sizeof( input ) &&
			in == truth ? 0 : 1;
		std::cout << "--- PID " << s << " of " << P << ": ";
		if( out.exit_code == 0 ) {
			std::cout << "MATCH\n";
		} else {
			std::cout << "ERROR! Input size is " << size << ", "
				<< "string\n\"" << in.str << "\"\n!= "
				<< "expected\n\"" << truth << "\"\n";
		}
	} else {
		out.exit_code = __in == nullptr && size == 0 ? 0 : 1;
		std::cout << "--- PID " << s << " of " << P << ": ";
		if( out.exit_code == 0 ) {
			std::cout << "MATCH, got expected values (nullptr and 0)\n";
		} else {
			std::cout << "ERROR! Got " << __in << " != nullptr and " << size
				<< " != 0\n";
		}
	}
}

template< grb::EXEC_MODE mode, bool broadcasted, typename InputT >
struct caller {
	static constexpr grb::AlpTypedFunc< InputT, output > fun =
		grbProgram< mode, broadcasted, InputT >;
};

template< grb::EXEC_MODE mode, bool broadcasted, typename InputT >
struct vcaller {
	static constexpr grb::AlpUntypedFunc< output > fun =
		vgrbProgram< mode, broadcasted, input >;
};

template< typename InputT >
struct vcaller< grb::AUTOMATIC, false, InputT > {
	static constexpr grb::AlpUntypedFunc< output > fun = autoVgrbProgram;
};

template< typename InputT >
class Runner {

	public:

		virtual grb::RC launch_typed(
			grb::AlpTypedFunc< InputT, output >,
			const InputT &, output &,
			bool
		) = 0;

		virtual grb::RC launch_untyped(
			grb::AlpUntypedFunc< output >,
			const void *, size_t,
			output &,
			bool
		) = 0;

		virtual grb::RC finalize() = 0;

		virtual ~Runner() = default;

};

template< grb::EXEC_MODE mode, typename InputT >
class bsp_launcher :
	public grb::Launcher< mode >, public Runner< InputT >
{

	public:

		using grb::Launcher< mode >::Launcher;

		grb::RC launch_typed(
			grb::AlpTypedFunc< InputT, output > grbProgram,
			const InputT &in, output &out, bool bc
		) override {
			return this->exec( grbProgram, in, out, bc );
		}

		grb::RC launch_untyped(
			grb::AlpUntypedFunc< output > grbProgram,
			const void * in, size_t in_size,
			output &out, bool bc
		) override {
			return this->exec( grbProgram, in, in_size, out, bc );
		}

		virtual grb::RC finalize() override {
			return grb::Launcher< mode >::finalize();
		}

};

template< grb::EXEC_MODE mode, typename InputT >
class bsp_benchmarker :
	public grb::Benchmarker< mode >, public Runner< InputT >
{

	private:

		size_t inner = 2;
		size_t outer = 2;


	public:

		using grb::Benchmarker< mode >::Benchmarker;

		grb::RC launch_typed(
			grb::AlpTypedFunc< InputT, output > grbProgram,
			const InputT &in, output &out,
			bool bc
		) override {
			return this->exec( grbProgram, in, out, inner, outer, bc );
		}

		grb::RC launch_untyped(
			const grb::AlpUntypedFunc< output > grbProgram,
			const void * const in, const size_t in_size,
			output &out, const bool bc
		) override {
			return this->exec( grbProgram, in, in_size, out, inner, outer, bc );
		}

		virtual grb::RC finalize() override {
			return grb::Benchmarker< mode >::finalize();
		}

};


enum RunnerType { Launch, Benchmark };

template< typename InputT >
std::unique_ptr< Runner< InputT > > make_runner(
	grb::EXEC_MODE mode, RunnerType type,
	size_t s, size_t P,
	const std::string &host, const std::string &port,
	const bool mpi_inited
) {
	Runner< InputT > *ret = nullptr;
#ifndef DISTRIBUTED_EXECUTION
	(void) mpi_inited;
#endif

	switch (type) {

		case Launch:

			switch (mode) {
				case grb::AUTOMATIC:
					ret = new bsp_launcher< grb::AUTOMATIC, InputT >;
					break;
#ifdef DISTRIBUTED_EXECUTION
				case grb::FROM_MPI:
					ret = new bsp_launcher< grb::FROM_MPI, InputT >( MPI_COMM_WORLD );
					break;

				case grb::MANUAL:
					ret = new bsp_launcher< grb::MANUAL, InputT >( s, P, host, port,
						mpi_inited );
					break;
#else
				case grb::MANUAL:
					ret = new bsp_launcher< grb::MANUAL, InputT >( s, P, host, port );
					break;
#endif
				default:
					break;
			}
			break;

		case Benchmark:
			switch (mode) {
				case grb::AUTOMATIC:
					ret = new bsp_benchmarker< grb::AUTOMATIC, InputT >;
					break;
#ifdef DISTRIBUTED_EXECUTION
				case grb::FROM_MPI:
					ret = new bsp_benchmarker< grb::FROM_MPI, InputT >( MPI_COMM_WORLD );
					break;

				case grb::MANUAL:
					ret = new bsp_benchmarker< grb::MANUAL, InputT >( s, P, host, port,
						mpi_inited );
					break;
#else
				case grb::MANUAL:
					ret = new bsp_benchmarker< grb::MANUAL, InputT >( s, P, host, port );
					break;

				case grb::FROM_MPI:
#endif

				default:
					break;
			}
			break;

		default:
			// error is caught later
			break;

	}

	if( ret == nullptr ) {
		throw std::runtime_error( "Error while creating runner" );
	}
	return std::unique_ptr< Runner< InputT > >( ret );
}

#define ERROR_ON( cond, str ) if( cond ) {                                  \
		std::cerr << __FILE__ ", " << __LINE__ << ": " << str << std::endl; \
		std::cout << "Test FAILED\n" << std::endl;                          \
		throw std::runtime_error( "check failed" );                         \
	}


template<
	template< grb::EXEC_MODE, bool, typename InputT > class FunT,
	grb::EXEC_MODE mode, typename RetT, typename InputT
>
RetT getFun( bool broadcast ) {
	return broadcast 
		? FunT< mode, true, InputT >::fun
		: FunT< mode, false, InputT >::fun;
}

template<
	template< grb::EXEC_MODE, bool, typename InputT > class CallerT,
	typename RetT, typename InputT
>
RetT getALPFun( grb::EXEC_MODE mode, bool broadcast ) {
	switch (mode) {
		case grb::AUTOMATIC:
			return getFun< CallerT, grb::AUTOMATIC, RetT, InputT >( broadcast );
			break;
		case grb::FROM_MPI:
			return getFun< CallerT, grb::FROM_MPI, RetT, InputT >( broadcast );
			break;
		case grb::MANUAL:
			return getFun< CallerT, grb::MANUAL, RetT, InputT >( broadcast );
			break;
		default:
			std::cerr << __FILE__ ", " << __LINE__ << ": " << "unknown mode " << mode
				<< std::endl;
			throw std::runtime_error( "unknown mode" );
			break;
	}
}

template< typename InputT >
std::unique_ptr< Runner< InputT > > create_runner(
	grb::EXEC_MODE mode, RunnerType rt,
	size_t s, size_t P,
	const std::string &host, const std::string &port,
	bool mpi_inited
) {
	try {
		return make_runner< InputT >(
			mode, rt, s, P,
			host,
			port,
			mpi_inited
		);
	} catch( std::runtime_error &e ) {
		std::cerr << "got a runtime exception: " << e.what() << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		throw e;
	} catch( std::exception &e ) {
		std::cerr << "got an exception: " << e.what() << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		throw e;
	} catch( ... ) {
		std::cerr << "got an unknown exception" << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		throw std::runtime_error( "unknown exception" );
	}
	return std::unique_ptr< Runner< InputT > >();
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
	// default values for shared-memory execution
	test_pid_t P = 1;
	test_pid_t s = 0;
	grb::EXEC_MODE mode = grb::AUTOMATIC;

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

	switch( mode ) {
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
			ERROR_ON( true, "unrecognised or invalid option: " << mode );
			break;
	}

	std::cout << "\n===> chosen initialisation method: " << mode_str << " <==="
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
	if( mode == grb::FROM_MPI ) {
		int rank;
		success = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
		ERROR_ON( success != MPI_SUCCESS, "Call to MPI_Comm_rank failed" );
		s = static_cast< test_pid_t >( rank );
	}
#endif

	const char * input_str = ( mode == grb::AUTOMATIC ) ? truth :
		( s == 0 ) ? truth : prelude;

	struct input in;
	struct output out;
	for( const bool broadcast : { true, false } ) {
		for( const RunnerType rt : { Launch, Benchmark } ) {
			const char * const runner_name = rt == Launch ? "Launch" : "Benchmark";
			const char * const bc_str = broadcast ? "true" : "false";
			std::cout << "\n ==> runner type: " << runner_name << ", "
				<< "broadcast: " << bc_str << std::endl;
			std::unique_ptr< Runner< input > > runner = create_runner< input >(
				mode, rt, s, P,
				std::string( (host != nullptr ? host : "" ) ),
				std::string( (port != nullptr ? port : "" ) ),
				true
			);
			std::cout << "  => untyped call\n" << std::endl;
			(void) strncpy( in.str, input_str, STR_LEN + 1 );
			grb::AlpUntypedFunc< output > vfun =
				getALPFun< vcaller, grb::AlpUntypedFunc< output >, input >(
					mode, broadcast
				);
			out.exit_code = 256; // the ALP function MUST set to 0
			grb::RC ret = runner->launch_untyped(
				vfun,
				reinterpret_cast< void * >( &in ), sizeof( input ),
				out, broadcast
			);
			ERROR_ON( ret != grb::SUCCESS,
				"untyped test FAILED with code: " << grb::toString( ret ) );
			ERROR_ON( out.exit_code != 0,
				"untyped test FAILED with exit code " << out.exit_code );

			std::cout << "\n  => typed call\n" << std::endl;
			grb::AlpTypedFunc< input, output > fun =
				getALPFun< caller, grb::AlpTypedFunc< input, output >, input >(
					mode, broadcast
				);
			out.exit_code = 256;
			ret = runner->launch_typed( fun, in, out, broadcast );
			ERROR_ON( ret != grb::SUCCESS,
				"typed test FAILED with code: " << grb::toString( ret ) );
			ERROR_ON( out.exit_code != 0,
				"typed test FAILED with exit code " << out.exit_code );

			ret = runner->finalize();

			ERROR_ON( ret != grb::SUCCESS,
				"finalisation FAILED with code: " << grb::toString( ret ) );
			std::cout << "  => OK" << std::endl;

			if( mode == grb::AUTOMATIC ) {
				// AUTOMTIC mode must implement a specific behaviour for
				// non-default-constructible input types like nd_input, here tested

				std::unique_ptr< Runner< nd_input > > nd_runner = create_runner< nd_input >(
					mode, rt, s, P,
					std::string( (host != nullptr ? host : "" ) ),
					std::string( (port != nullptr ? port : "" ) ),
					true
				);

				std::cout << "\n  => untyped call, non-default-constructible input\n"
					<< std::endl;
				out.exit_code = 256;
				nd_input ndin( input_str );
				ret = nd_runner->launch_untyped(
					vfun,
					reinterpret_cast< void * >( &ndin ), sizeof( nd_input ),
					out, broadcast
				);
				// untyped calls must succeed even with a non-default-constructible input
				ERROR_ON( ret != grb::SUCCESS,
					"untyped test FAILED with code: " << grb::toString( ret ) );
				ERROR_ON( out.exit_code != 0,
					"untyped test FAILED with exit code " << out.exit_code );

				std::cout << "\n  => typed call, non-default-constructible input\n"
					<< std::endl;
				out.exit_code = 256;
				grb::AlpTypedFunc< nd_input, output > ndfun =
					getALPFun< caller, grb::AlpTypedFunc< nd_input, output >, nd_input >(
						mode, broadcast
					);
				ret = nd_runner->launch_typed( ndfun, ndin, out, broadcast );
				// get P from process, as it may not be known outside of the
				// launcher (e.g., for AUTOMATIC mode)
				const bool should_fail = ( !broadcast ) && out.P > 1;
				int expected_retval = should_fail ? 256 : 0;
				// typed call should fail if ALL of the following conditions are met:
				// - AUTOMATIC mode
				// - non-default-constructible input
				// - no broadcast requested
				// - more than one process to run.
				// The idea is that process 0 receives the "original" input via
				// the launcher, but other processes cannot create a meaningful
				// one, because the input is non-default-constructible and
				// because broadcast has not been requested (note: broadcast
				// occurs ONLY on user's request): in such a case, the call
				// cannot proceed and is aborted
				ERROR_ON( should_fail && ret == grb::SUCCESS,
					"run is successful, but should have failed" );
				ERROR_ON( out.exit_code != expected_retval,
					"typed test FAILED with exit code " << out.exit_code );
			}
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

