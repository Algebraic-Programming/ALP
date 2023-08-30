
#include <iostream>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <limits>

#include <string.h>
#include <stdio.h>
#include <mpi.h>

#include <graphblas.hpp>
#include <graphblas/utils/ranges.hpp>

#ifdef NO_LPF_AUTO_INIT
const int LPF_MPI_AUTO_INITIALIZE = 0;
#endif

constexpr size_t STR_LEN = 1024;

static const char truth[ STR_LEN + 1 ] = "Night is worn,\nand the morn\nrises from the slumberous mass.";


struct input {
	char str[ STR_LEN + 1 ];
};

bool operator==( const struct input & obj, const char * ext ) {
	return strnlen( obj.str, STR_LEN + 1 ) == strnlen( ext, STR_LEN + 1 ) &&
		strncmp( obj.str, ext, STR_LEN + 1 ) == 0;
}

bool operator==( const char * ext, const struct input & obj ) {
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
		printf( "PID %lu of %lu: match, string is\n\"%s\"\n", s, P, in.str );
	} else {
		printf( "PID %lu of %lu: ERROR! \n\"%s\"\n!=\n\"%s\"\n", s, P, in.str, truth );
	}
}

void vgrbProgram( const void* __in, const size_t, struct output &out ) {
	const struct input &in = *reinterpret_cast< const struct input *>( __in );
	return grbProgram( in, out );
}

class runner_t {
public:
	virtual grb::RC launch_typed( grb::AlpTypedFunc< input, output >, const input &, output &, bool  ) = 0;
	virtual grb::RC launch_untyped( grb::AlpUntypedFunc< void, output >, const void *, size_t, output &, bool ) = 0;
	virtual grb::RC finalize() = 0;

	virtual ~runner_t() = default;
};

template< grb::EXEC_MODE mode >
class bsp_launcher : public grb::Launcher< mode, grb::Backend::BSP1D >, public runner_t {
public:
	using grb::Launcher< mode, grb::Backend::BSP1D >::Launcher;

	grb::RC launch_typed( grb::AlpTypedFunc< input, output > grbProgram, const input &in, output &out, bool bc ) override {
		return this->exec( grbProgram, in, out, bc );
	}
	grb::RC launch_untyped( grb::AlpUntypedFunc< void, output > grbProgram, const void * in, size_t in_size, output &out, bool bc ) override {
		return this->exec( grbProgram, in, in_size, out, bc );
	}
	virtual grb::RC finalize() override {
		return grb::Launcher< mode, grb::Backend::BSP1D >::finalize();
	}
};

template< grb::EXEC_MODE mode >
class bsp_benchmarker : public grb::Benchmarker< mode, grb::Backend::BSP1D >, public runner_t {
	size_t inner = 2;
	size_t outer = 2;

public:
	using grb::Benchmarker< mode, grb::Backend::BSP1D >::Benchmarker;

	grb::RC launch_typed( grb::AlpTypedFunc< input, output > grbProgram, const input &in, output &out, bool bc ) override {
		return this->exec( grbProgram, in, out, bc, inner, outer );
	}
	grb::RC launch_untyped( grb::AlpUntypedFunc< void, output > grbProgram, const void * in, size_t in_size, output &out, bool bc ) override {
		return this->exec( grbProgram, in, in_size, out, bc, inner, outer );
	}
	virtual grb::RC finalize() override {
		return grb::Benchmarker< mode, grb::Backend::BSP1D >::finalize();
	}
};


enum runner_type { Launch, Benchmark };

std::unique_ptr< runner_t > make_runner( grb::EXEC_MODE mode, runner_type type,
	size_t s, size_t P, const std::string &host, const std::string &port, bool mpi_inited ) {

	runner_t *ret = nullptr;

	switch (type)
	{
	case Launch:
		switch (mode)
		{
		case grb::AUTOMATIC:
			ret = new bsp_launcher< grb::AUTOMATIC >;
			break;

		case grb::FROM_MPI:
			ret = new bsp_launcher< grb::FROM_MPI >( MPI_COMM_WORLD );
			break;

		case grb::MANUAL:
			ret = new bsp_launcher< grb::MANUAL >( s, P, host, port, mpi_inited );
			break;
		default:
			break;
		}
		break;

	case Benchmark:
		switch (mode)
		{
		case grb::AUTOMATIC:
			ret = new bsp_benchmarker< grb::AUTOMATIC >;
			break;

		case grb::FROM_MPI:
			ret = new bsp_benchmarker< grb::FROM_MPI >( MPI_COMM_WORLD );
			break;

		case grb::MANUAL:
			ret = new bsp_benchmarker< grb::MANUAL >( s, P, host, port, mpi_inited );
			break;
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
	return std::unique_ptr< runner_t >( ret );
}

#define ERROR_ON( cond, str ) if( cond ) {									\
		std::cerr << __FILE__ ", " << __LINE__ << ": " << str << std::endl;	\
		return EXIT_FAILURE;												\
	}

int main( int argc, char ** argv ) {

	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";


	int lpf_mpi_inited = 0;
	int success = MPI_Initialized( &lpf_mpi_inited );
	ERROR_ON( success != MPI_SUCCESS, "cannot determine initalization info" );
	const char * host = nullptr;
	const char * port = nullptr;
	lpf_pid_t P = std::numeric_limits< lpf_pid_t >::max();
	lpf_pid_t s = std::numeric_limits< lpf_pid_t >::max();
	grb::EXEC_MODE mode = grb::FROM_MPI;

	if( lpf_mpi_inited != 0 ) {
		mode = grb::AUTOMATIC;
		ERROR_ON( argc != 1, "no argument needed" );
	} else {
		if( argc == 1 ) {
			mode = grb::FROM_MPI;
		} else if( argc == 5 ) {
			mode = grb::MANUAL;
		} else {
			ERROR_ON( true, "no argument needed" );
		}
	}
	std::cout << "\n===> chosen initialization method: " << mode << " <===" << std::endl;

	if( mode == grb::MANUAL ) {
		// read command-line args
		host = argv[ 1 ];
		port = argv[ 2 ];
		P = static_cast< lpf_pid_t >( std::stoi( argv[ 3 ] ) );
		s = static_cast< lpf_pid_t >( std::stoi( argv[ 4 ] ) );

		// input sanity checks
		ERROR_ON( host == NULL || strlen( host ) == 0, "Invalid hostname: " << argv[ 1 ] );
		ERROR_ON( port == NULL || strlen( port ) == 0, "value for port name or number: " << argv[ 2 ] );
		ERROR_ON( !grb::utils::is_in_normalized_range( s, P ), "Invalid value for PID: " << argv[ 4 ] );
	}
	if( mode == grb::FROM_MPI || mode == grb::MANUAL ) {
		success = MPI_Init( NULL, NULL );
		ERROR_ON( success != MPI_SUCCESS, "Call to MPI_Init failed" );
	}

	struct input in;
	struct output out;
	out.exit_code = 0;

	for( runner_type rt : {Launch, Benchmark } ) {
		const char * const runner_name = rt == Launch ? "Launch" : "Benchmark";
		std::cout << "\n ==> runner type: " << runner_name << std::endl;
		std::unique_ptr< runner_t > runner;
		try {
			runner = make_runner( mode, rt, s, P,
				std::string( (host != nullptr ? host : "" ) ),
				std::string( (port != nullptr ? port : "" ) ),
				true );
		} catch( std::runtime_error &e ) {
			std::cerr << "got a runtime exception: " << e.what() << std::endl;
			return EXIT_FAILURE;
		} catch( std::exception &e ) {
			std::cerr << "got an exception: " << e.what() << std::endl;
			return EXIT_FAILURE;
		} catch( ... ) {
			std::cerr << "got an unknown exception" << std::endl;
			return EXIT_FAILURE;
		}

		std::cout << "  => untyped call\n" << std::endl;
		(void)strncpy( in.str, truth, STR_LEN + 1 );
		grb::RC ret = runner->launch_untyped( &vgrbProgram, reinterpret_cast< void * >( &in ), sizeof( struct input ), out, true );
		ERROR_ON( ret != grb::SUCCESS, "untyped test FAILED with code: " << grb::toString( ret ) );
		ERROR_ON( out.exit_code != 0, "untyped test FAILED with exit code " << out.exit_code );

		std::cout << "\n  => typed call\n" << std::endl;
		ret = runner->launch_typed( &grbProgram, in, out, true );
		ERROR_ON( ret != grb::SUCCESS, "typed test FAILED with code: " << grb::toString( ret ) );
		ERROR_ON( out.exit_code != 0, "typed test FAILED with exit code " << out.exit_code );

		runner->finalize();
	}
	if( mode == grb::FROM_MPI || mode == grb::MANUAL ) {
		success = MPI_Finalize();
		ERROR_ON( success != MPI_SUCCESS, "Call to MPI_Finalize failed" );
	}

	std::cout << "\nTest OK\n" << std::endl;
	return EXIT_SUCCESS;
}
