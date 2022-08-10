
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

/*
 * @author A. N. Yzelman
 * @date 17th of April, 2017
 */

#ifndef _H_GRB_BSP1D_EXEC
#define _H_GRB_BSP1D_EXEC

#include <lpf/collectives.h>
#include <lpf/core.h>
#include <lpf/mpi.h>
#include <mpi.h>
#include <string.h> //for memcpy

#include <graphblas/backends.hpp>
#include <graphblas/base/exec.hpp>
#include <graphblas/collectives.hpp>
#include <graphblas/rc.hpp>

#include "init.hpp"

#ifndef _GRB_NO_STDIO
 #include <iostream> //for std::cerr
#endif


/** Global internal singleton to track whether MPI was initialized. */
extern bool _grb_mpi_initialized;

/** Global internal function used to call lpf_hook or lpf_exec with. */
template< typename T, typename U, bool broadcast = true >
void _grb_exec_spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	assert( P > 0 );
	assert( s < P );

#ifdef _DEBUG
	if( s == 0 ) {
		std::cout << "Info: launcher spawned or hooked " << P << " ALP/GraphBLAS "
			<< "user processes.\n";
	}
#endif

	T data_in_local; // construct default input type

	// get input data from PID 0
	if( broadcast && P > 1 ) {

		// init collectives
		lpf_coll_t coll;
		lpf_err_t brc = lpf_collectives_init( ctx, s, P, 0, 0, 0, &coll );
		assert( brc == LPF_SUCCESS );

		// we need input fields from root, prepare for broadcast
		brc = lpf_resize_message_queue( ctx, 2*(P-1) ); // two-phase broadcast may
		                                                // get up to P-1 messages and
								// send up to P-1 messages
								// per process
		assert( brc == LPF_SUCCESS );
		brc = lpf_resize_memory_register( ctx, 2 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		lpf_memslot_t global;
		if( s == 0 ) {
			assert( args.input_size == sizeof( T ) );
			brc = lpf_register_global( ctx,
				const_cast< void * >( args.input ),
				args.input_size, &global
			);
		} else {
			assert( args.input_size == 0 );
			brc = lpf_register_global( ctx, &data_in_local, sizeof( T ), &global );
		}
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_broadcast( coll, global, global, sizeof( T ), 0 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_deregister( ctx, global );
		assert( brc == LPF_SUCCESS );

#ifdef NDEBUG
		(void)brc;
#endif
	}

	// sanity check
	if( !broadcast ) {
		// if we do not broadcast then everyone should have their own local input
		assert( args.input_size == sizeof( T ) );
	}

	// get input data
	const T &data_in = broadcast ?
		// then get unified view of input data after broadcast
		( s == 0 ? *static_cast< const T * >( args.input ) : data_in_local ) :
		// otherwise just copy from args_in if there is one (to catch automatic mode)
		*static_cast< const T * >( args.input );

	// we need an output field
	U data_out_local = U();
	U &data_out = args.output_size == sizeof( U ) ?
		*static_cast< U * >( args.output ) : // if we were passed output area, use it
		data_out_local;                      // otherwise use local empy output area

	// initialise ALP/GraphBLAS
	grb::RC grb_rc = grb::init( s, P, ctx );
	if( grb_rc != grb::SUCCESS ) {
		std::cerr << "Error: could not initialise ALP/GraphBLAS" << std::endl;
		assert( false );
		return;
	}

	// retrieve and run the function to be executed
	if( args.f_size == 1 ) {
		typedef void ( *grb_func_t )( const T &, U & );
		grb_func_t grb_program =
			reinterpret_cast< grb_func_t >( args.f_symbols[ 0 ] );
		( *grb_program )( data_in, data_out );
	} else {
		// assume we are performning benchmarks
		typedef void ( *grb_func_t )( const T &, U & );
		typedef void ( *bench_func_t )( void ( *grb_program )( const T &, U & ),
			const T &, U &, lpf_pid_t );
		bench_func_t bench_program = reinterpret_cast< bench_func_t >( args.f_symbols[ 0 ] );
		grb_func_t grb_program = reinterpret_cast< grb_func_t >( args.f_symbols[ 1 ] );
		( *bench_program )( grb_program, data_in, data_out, s );
	}

	// finalise ALP/GraphBLAS
	grb_rc = grb::finalize();
	if( grb_rc != grb::SUCCESS ) {
		std::cerr << "Error: could not finalise ALP/GraphBLAS" << std::endl;
		assert( false );
	}
}

/** Global internal function used to call lpf_hook or lpf_exec with. */
template< typename U, bool broadcast = true >
void _grb_exec_varin_spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	assert( P > 0 );
	assert( s < P );

#ifdef _DEBUG
	// info to stdout
	if( s == 0 ) {
		std::cout << "Info: launcher spawned " << P << " processes.\n";
	}
#endif
	// input data to grbProgram
	void * data_in = NULL;

	// size of the data_in block
	size_t size;

	// we need input fields from root. First synchronise on input size
	if( broadcast && P > 1 ) {

		// init collectives
		lpf_coll_t coll;
		lpf_err_t brc = lpf_resize_message_queue( ctx, P - 1 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_resize_memory_register( ctx, 2 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_collectives_init( ctx, s, P, 1, 0, sizeof( size_t ), &coll );
		assert( brc == LPF_SUCCESS );

		// broadcast the size of data
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		lpf_memslot_t global;
		if( s == 0 ) {
			size = args.input_size;
		}
		brc = lpf_register_global( ctx, &size, sizeof( size_t ), &global );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_broadcast( coll, global, global, sizeof( size_t ), 0 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_deregister( ctx, global );
		assert( brc == LPF_SUCCESS );

		// now that the input size is known, retrieve the input data
		if( s > 0 ) {
			data_in = new char[ size ];
		} else {
			data_in = const_cast< void * >( args.input );
		}
		brc = lpf_register_global( ctx, data_in, size, &global );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_broadcast( coll, global, global, size, 0 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_deregister( ctx, global );
		assert( brc == LPF_SUCCESS );

#ifdef NDEBUG
		(void)brc;
#endif
	} else {
		data_in = const_cast< void * >( args.input );
		size = args.input_size;
	}

	// we need an output field
	U data_out_local = U();
	U &data_out = args.output_size == sizeof( U ) ?
		*static_cast< U * >( args.output ) :
		data_out_local;
	// note: the above switch handily catches automatic mode

	// initialise ALP/GraphBLAS
	grb::RC grb_rc = grb::init( s, P, ctx );
	if( grb_rc != grb::SUCCESS ) {
		std::cerr << "Error: could not initialise ALP/GraphBLAS" << std::endl;
		assert( false );
		return;
	}

	// retrieve and run the function to be executed
	if( args.f_size == 1 ) {
		typedef void ( *grb_func_t )( void *, size_t, U & );
		grb_func_t grb_program =
			reinterpret_cast< grb_func_t >( args.f_symbols[ 0 ] );
		( *grb_program )( (void *)data_in, size, data_out );
	} else {
		// assume we are performning benchmarks
		typedef void ( *grb_func_t )( void *, size_t, U & );
		typedef void ( *bench_func_t )( void ( *grb_program )( void *, size_t, U & ),
				void *, size_t,
				U &, lpf_pid_t
		);
		bench_func_t bench_program = reinterpret_cast< bench_func_t >( args.f_symbols[ 0 ] );
		grb_func_t grb_program = reinterpret_cast< grb_func_t >( args.f_symbols[ 1 ] );
		( *bench_program )( grb_program, (void *)data_in, size, data_out, s );
	}

	// finalise ALP/GraphBLAS
	grb_rc = grb::finalize();
	if( grb_rc != grb::SUCCESS ) {
		std::cerr << "Error: could not finalise ALP/GraphBLAS" << std::endl;
		assert( false );
	}
}

namespace grb {

	/**
	 * No implementation notes.
	 */
	template<>
	class Launcher< FROM_MPI, BSP1D > {


		protected:

			/** The LPF init struct. Will be initialised during construction. */
			lpf_init_t init;


		public:

			/**
			 * No implementation notes.
			 *
			 * @param[in] MPI communicator to hook into.
			 *
			 * @throws runtime_error When a standard MPI call fails.
			 */
			Launcher( const MPI_Comm comm = MPI_COMM_WORLD ) {
				// run-time sanity check when using MPI:
				// we (as in LPF) should NOT be managing MPI
				if( LPF_MPI_AUTO_INITIALIZE ) {
					throw std::runtime_error( "Program was not linked with the symbol "
						"LPF_MPI_AUTO_INITIALIZE set to 0 while an instance of "
						"Launcher<Manual> or Launcher<FROM_MPI> is being requested." );
				}

				// init from communicator
				const lpf_err_t initrc = lpf_mpi_initialize_with_mpicomm( comm, &init );

				// check for success
				if( initrc != LPF_SUCCESS ) {
					throw std::runtime_error(
						"LPF could not connect launcher group over TCP/IP."
					);
				}

				// done!
			}

			/** Disable copy constructor. */
			Launcher( const Launcher & ) = delete;

			/** Disable copy constructor. */
			Launcher & operator=( const Launcher & ) = delete;

			/**
			 * Implementation note: this Launcher will clear a field of
			 * type \a lpf_init_t.
			 */
			~Launcher() {
				assert( init != LPF_INIT_NONE );
				const lpf_err_t finrc = lpf_mpi_finalize( init );
				if( finrc != LPF_SUCCESS ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Warning: could not destroy launcher::init from ~Launcher.\n";
#endif
				}
				init = LPF_INIT_NONE;
			}

			/** No implementation notes. */
			template< typename U >
			RC exec( void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) const {
				// prepare args
				lpf_func_t fargs[ 2 ];
				lpf_args_t args;
				fargs[ 0 ] = reinterpret_cast< lpf_func_t >( grb_program );
				args = { data_in, in_size, &data_out, sizeof( U ), fargs, 1 };

				// do hook
				const lpf_err_t spmdrc = broadcast ?
					lpf_hook( init, &(_grb_exec_varin_spmd< U, true >), args ) :
					lpf_hook( init, &(_grb_exec_varin_spmd< U, false >), args );

				// check error code
				if( spmdrc != LPF_SUCCESS ) {
					return PANIC;
				}

				// done
				return SUCCESS;
			}

			/** No implementation notes. */
			template< typename T, typename U >
			RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
				const T &data_in, U &data_out,            // input & output data
				const bool broadcast = false
			) {
				// prepare args
				lpf_func_t fargs[ 2 ];
				lpf_args_t args;
				fargs[ 0 ] = reinterpret_cast< lpf_func_t >( grb_program );
				args = { &data_in, sizeof( T ), &data_out, sizeof( U ), fargs, 1 };

				// do hook
				const lpf_err_t spmdrc = broadcast ?
					lpf_hook( init, &(_grb_exec_spmd< T, U, true >), args ) :
					lpf_hook( init, &(_grb_exec_spmd< T, U, false >), args );

				// check error code
				if( spmdrc != LPF_SUCCESS ) {
					return PANIC;
				}

				// done
				return SUCCESS;
			}

			/**
			 * Since the user is using ALP/GraphBLAS directly from MPI, the user codes
			 * should call MPI_Finalize. This function thus is a no-op in this particular
			 * specialisation.
			 */
			static grb::RC finalize() {
				return SUCCESS;
			}

	};

	/**
	 * No implementation notes.
	 */
	template< enum EXEC_MODE mode >
	class Launcher< mode, BSP1D > {


		private:

			// we should never be called for FROM_MPI mode-- the above
			// specialisation should be used instead
			static_assert( mode != FROM_MPI,
				"EXEC_MODE::FROM_MPI for BSP1D is implemented in specialised class" );

			/** The user process ID in this launcher group. */
			const size_t _s;

			/** The total number of user processes in this launcher group. */
			const size_t _P;

			/** The connection broker in this launcher group. */
			const std::string _hostname;

			/** The port at #_hostname used for brokering connections. */
			const std::string _port;


		protected:

			/** The LPF init struct. Will be initialised during construction. */
			lpf_init_t init;


		public:

			/**
			 * When \a mode is #AUTOMATIC, this implementation adheres to
			 * the base specification. When \a mode is #MANUAL, this
			 * implementation specifies additionally the following:
			 *
			 * The time-out of this constructor is thirty seconds.
			 *
			 * @param[in] hostname May not be empty. Must resolve to an IP.
			 * @param[in] port     May not be empty. Must be either a port
			 *                     number of a registered service name.
			 *
			 * In addition to the standard-defined exceptions, the following
			 * may additionally be thrown:
			 * @throws invalid_argument When hostname or port are empty.
			 * @throws runtime_error    When the requested launcher group
			 *                          could not be created.
			 */
			Launcher( const size_t process_id = 0,            // user process ID
				const size_t nprocs = 1,                  // total number of user processes
				const std::string hostname = "localhost", // one of the process' hostnames
				const std::string port = "0",             // a free port at hostname
				const bool is_mpi_inited = false
			) : _s( process_id ),
				_P( nprocs ), _hostname( hostname ), _port( port )
			{
				// sanity check
				if( nprocs == 0 ) {
					throw std::invalid_argument( "Total number of user processes must be "
						"strictly larger than zero." );
				}
				if( process_id >= nprocs ) {
					throw std::invalid_argument( "Process ID must be strictly smaller than "
						"total number of user processes." );
				}

				// when using MPI in hook mode
				if( mode == MANUAL ) {
					// run-time sanity check when using MPI:
					// we (as in LPF) should NOT be managing MPI
					if( LPF_MPI_AUTO_INITIALIZE ) {
						throw std::runtime_error( "Program was not linked with the symbol "
							"LPF_MPI_AUTO_INITIALIZE set to 0 while an instance of "
							"Launcher<Manual> or Launcher<FROM_MPI> is being requested." );
					}
					// initialise MPI if not already done
					if( !is_mpi_inited && !_grb_mpi_initialized ) {
						if( MPI_Init( NULL, NULL ) != MPI_SUCCESS ) {
							throw std::runtime_error( "Call to MPI_Init failed." );
						} else {
							_grb_mpi_initialized = true;
						}
					}
				}

				// handle each mode's specifics
				if( mode == MANUAL ) {
					// additional sanity check
					if( hostname.compare( "" ) == 0 || port.compare( "" ) == 0 ) {
						throw std::invalid_argument(
							"Hostname and/or port name cannot be empty."
						);
					}

					// try and create a lpf_init_t
					const lpf_err_t initrc = lpf_mpi_initialize_over_tcp(
						hostname.c_str(), port.c_str(), // server info
						120000,                         // time out
						process_id, nprocs,             // process info
						&init
					);

					// check for success
					if( initrc != LPF_SUCCESS ) {
#ifndef _GRB_NO_STDIO
						throw std::runtime_error(
							"LPF could not connect launcher group over TCP/IP."
						);
#endif
					}
				} else {
					// sanity check: we should be in automatic mode
					assert( mode == AUTOMATIC );
					// otherwise, we don't need init
					init = LPF_INIT_NONE;
				}

			}

			/** Disable copy constructor. */
			Launcher( const Launcher & ) = delete;

			/** Disable copy constructor. */
			Launcher & operator=( const Launcher & ) = delete;

			/**
			 * Implementation note: this Launcher may need to clear a field of
			 * type \a lpf_init_t when used in MANUAL mode.
			 */
			~Launcher() {
				if( mode == MANUAL ) {
					assert( init != LPF_INIT_NONE );
					// try and destroy the lpf_init_t
					const lpf_err_t finrc = lpf_mpi_finalize( init );
					if( finrc != LPF_SUCCESS ) {
#ifndef _GRB_NO_STDIO
						std::cerr << "Warning: could not destroy launcher::init from ~launcher.\n";
#endif
					}
					init = LPF_INIT_NONE;
				} else {
					assert( init == LPF_INIT_NONE );
				}
			}

			/** No implementation notes. */
			template< typename U >
			RC exec(
				void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) const {
				// prepare args
				lpf_func_t fargs[ 2 ];
				lpf_args_t args;
				fargs[ 0 ] = reinterpret_cast< lpf_func_t >( grb_program );
				args = { data_in, in_size, &data_out, sizeof( U ), fargs, 1 };

				// launch
				lpf_err_t spmdrc = LPF_SUCCESS;
				if( mode == MANUAL ) {
					// do hook
					spmdrc = broadcast ?
						lpf_hook( init, &(_grb_exec_varin_spmd< U, true >), args ) :
						lpf_hook( init, &(_grb_exec_varin_spmd< U, false >), args );
				} else {
					assert( mode == AUTOMATIC );
					// do exec
					spmdrc = lpf_exec( LPF_ROOT, LPF_MAX_P,
						&(_grb_exec_varin_spmd< U >), args );
				}

				// check error code
				if( spmdrc != LPF_SUCCESS ) {
					return PANIC;
				}

				// done
				return SUCCESS;
			}

			/** No implementation notes. */
			template< typename T, typename U >
			RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
				const T &data_in, U &data_out,            // input & output data
				const bool broadcast = false
			) {
				// prepare args
				lpf_func_t fargs[ 2 ];
				lpf_args_t args;
				fargs[ 0 ] = reinterpret_cast< lpf_func_t >( grb_program );
				args = { &data_in, sizeof( T ), &data_out, sizeof( U ), fargs, 1 };

				// launch
				lpf_err_t spmdrc = LPF_SUCCESS;
				if( mode == MANUAL ) {
					// do hook
					spmdrc = broadcast ?
						lpf_hook( init, &(_grb_exec_spmd< T, U, true >), args ) :
						lpf_hook( init, &(_grb_exec_spmd< T, U, false >), args );
				} else {
					assert( mode == AUTOMATIC );
					// do exec
					spmdrc = lpf_exec( LPF_ROOT, LPF_MAX_P, &(_grb_exec_spmd< T, U >), args );
				}

				// check error code
				if( spmdrc != LPF_SUCCESS ) {
					return PANIC;
				}

				// done
				return SUCCESS;
			}

			/**
			 * This implementation needs to release MPI resources in manual mode.
			 */
			static RC finalize() {
				// finalise MPI when in manual mode
				if( mode == MANUAL && _grb_mpi_initialized ) {
					_grb_mpi_initialized = false;
					if( MPI_Finalize() != MPI_SUCCESS ) {
#ifndef _GRB_NO_STDIO
						std::cerr << "Warning: MPI_Finalize returned non-SUCCESS exit code.\n";
#endif
						return grb::PANIC;
					}
				}
				return grb::SUCCESS;
			}

	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_EXEC''

