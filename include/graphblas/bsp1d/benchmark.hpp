
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

#ifndef _H_GRB_BSP1D_BENCH
#define _H_GRB_BSP1D_BENCH

#include <lpf/core.h>

#include <graphblas/base/benchmark.hpp>
#include <graphblas/exec.hpp>
#include <graphblas/rc.hpp>

#include "exec.hpp"

namespace grb {

	namespace internal {

		struct packedBenchmarkerInput {
			const void * blob;
			size_t blob_size;
			size_t inner;
			size_t outer;
			bool bcast_blob;
		};

	} // namespace internal

} // namespace grb

/** Global internal function used to call lpf_hook with. */
template< typename T, typename U >
void _grb_bench_spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	assert( P > 0 );
	assert( s < P );

	// construct default input type
	T data_in_local;
	// get input struct
	assert( args.input_size == sizeof( struct grb::internal::packedBenchmarkerInput ) );
	const struct grb::internal::packedBenchmarkerInput input = *static_cast< const struct grb::internal::packedBenchmarkerInput * >( args.input );

	// get input data from PID 0
	if( input.bcast_blob && P > 1 ) {
		// init BSP & collectives
		lpf_coll_t coll;
		lpf_err_t brc = lpf_resize_message_queue( ctx, P - 1 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_resize_memory_register( ctx, 2 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_collectives_init( ctx, s, P, 0, 0, 0, &coll );
		assert( brc == LPF_SUCCESS );

		// we need input fields from root
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		lpf_memslot_t global;
		if( s == 0 ) {
			assert( input.blob_size == sizeof( T ) );
			brc = lpf_register_global( ctx, const_cast< void * >( input.blob ), input.blob_size, &global );
		} else {
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
	} else {
		// if we do not broadcast then everyone should have their own local input
		assert( input.blob_size == sizeof( T ) );
	}

	// get input data
	const T & data_in = input.bcast_blob ?
        // then get unified view of input data after broadcast
        ( s == 0 ? *static_cast< const T * >( input.blob ) : data_in_local ) :
        // otherwise just copy from args_in if there is one (to catch automatic mode)
        *static_cast< const T * >( input.blob );

	// we need an output field
	U data_out_local = U();
	U & data_out = args.output_size == sizeof( U ) ? *static_cast< U * >( args.output ) : // if we were passed output area, use it
                                                     data_out_local;                                                                   // otherwise use local empty output area

	// init graphblas
	if( grb::init( s, P, ctx ) != grb::SUCCESS ) {
		return; // note that there is no way to return error codes
	}

	// retrieve and run the function to be executed
	assert( args.f_size == 2 );
	// retrieve benchmarking functions
	typedef void ( *grb_func_t )( const T &, U & );
	typedef void ( *bench_func_t )( void ( *grb_program )( const T &, U & ), const T &, U &, size_t, size_t, lpf_pid_t );
	bench_func_t bench_program = reinterpret_cast< bench_func_t >( args.f_symbols[ 0 ] );
	grb_func_t grb_program = reinterpret_cast< grb_func_t >( args.f_symbols[ 1 ] );
	// execute benchmark
	( *bench_program )( grb_program, data_in, data_out, input.inner, input.outer, s );

	// close GraphBLAS context and done!
	(void)grb::finalize();
}

/** Global internal function used to call lpf_hook with. */
template< typename U >
void _grb_bench_varin_spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	assert( P > 0 );
	assert( s < P );

	// input data to grbProgram
	void * data_in = NULL;
	// get input struct
	assert( args.input_size == sizeof( struct grb::internal::packedBenchmarkerInput ) );
	const struct grb::internal::packedBenchmarkerInput input = *static_cast< const struct grb::internal::packedBenchmarkerInput * >( args.input );

	// size of the data_in block
	size_t size;

	// we need input fields from root. First synchronise on input size
	if( input.bcast_blob && P > 1 ) {

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
			size = input.blob_size;
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
			data_in = const_cast< void * >( input.blob );
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
		data_in = const_cast< void * >( input.blob );
		size = input.blob_size;
	}

	// we need an output field
	U data_out_local = U();
	U & data_out = args.output_size == sizeof( U ) ? *static_cast< U * >( args.output ) : data_out_local;
	// note: the above switch handily catches automatic mode

	// init graphblas
	if( grb::init( s, P, ctx ) != grb::SUCCESS ) {
		return; // note that there is no way to return error codes
	}

	// retrieve and run the function to be executed
	assert( args.f_size == 2 );
	// assume we are performing benchmarks
	typedef void ( *grb_func_t )( void *, size_t, U & );
	typedef void ( *bench_func_t )( void ( *grb_program )( void *, size_t, U & ), void *, size_t, U &, size_t, size_t, lpf_pid_t );
	bench_func_t bench_program = reinterpret_cast< bench_func_t >( args.f_symbols[ 0 ] );
	grb_func_t grb_program = reinterpret_cast< grb_func_t >( args.f_symbols[ 1 ] );
	// run benchmark
	( *bench_program )( grb_program, (void *)data_in, size, data_out, input.inner, input.outer, s );

	// close GraphBLAS context and done!
	(void)grb::finalize();
}

/** Global internal function used to call lpf_exec with. */
template< typename T, typename U, bool varin >
void _grb_bench_exec( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	assert( P > 0 );
	assert( s < P );

	grb::internal::packedBenchmarkerInput input;
	constexpr size_t size = sizeof( struct grb::internal::packedBenchmarkerInput );

	// only call broadcast if P > 1, or otherwise UB
	if( P > 1 ) {
		// init and use collectives to broadcast input
		lpf_coll_t coll;
		const size_t nmsgs = P + 1 > 2 * P - 3 ? P + 1 : 2 * P - 3; // see LPF collectives doc
		lpf_err_t brc = lpf_resize_message_queue( ctx, nmsgs );
		assert( brc == LPF_SUCCESS );
		brc = lpf_resize_memory_register( ctx, 3 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_collectives_init( ctx, s, P, 1, 0, size, &coll );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		lpf_memslot_t destination, source;
		brc = lpf_register_global( ctx, &input, size, &destination );
		assert( brc == LPF_SUCCESS );
		if( s == 0 ) {
			assert( args.input_size == size );
			brc = lpf_register_global( ctx, const_cast< void * >( args.input ), size, &source );
		} else {
			brc = lpf_register_global( ctx, const_cast< void * >( args.input ), 0, &source );
		}
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_broadcast( coll, source, destination, size, 0 );
		assert( brc == LPF_SUCCESS );
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		assert( brc == LPF_SUCCESS );
		brc = lpf_deregister( ctx, source );
		assert( brc == LPF_SUCCESS );
		brc = lpf_deregister( ctx, destination );
		assert( brc == LPF_SUCCESS );
#ifdef NDEBUG
		(void)brc;
#endif
	}

	// non-root processes update args
	if( s > 0 ) {
		input.blob = NULL;
		input.blob_size = 0;
		args.input = &input;
		args.input_size = size;
		assert( input.bcast_blob );
	}

	// now we are at exactly the equal state as a hook-induced function
	if( varin ) {
		_grb_bench_varin_spmd< U >( ctx, s, P, args );
	} else {
		_grb_bench_spmd< T, U >( ctx, s, P, args );
	}
}

namespace grb {

	template<>
	class Benchmarker< FROM_MPI, BSP1D > : protected Launcher< FROM_MPI, BSP1D >, protected internal::BenchmarkerBase {

	public:
		Benchmarker( const MPI_Comm comm = MPI_COMM_WORLD ) : Launcher< FROM_MPI, BSP1D >( comm ) {}

		template< typename U >
		RC
		exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const size_t inner, const size_t outer, const bool broadcast = false )
			const {
			// prepare packed input
			struct internal::packedBenchmarkerInput input;
			input.blob = data_in;
			input.blob_size = in_size;
			input.inner = inner;
			input.outer = outer;
			input.bcast_blob = broadcast;

			// prepare args
			lpf_func_t fargs[ 2 ];
			lpf_args_t args;
			fargs[ 0 ] = reinterpret_cast< lpf_func_t >( benchmark< U > );
			fargs[ 1 ] = reinterpret_cast< lpf_func_t >( grb_program );
			args = { &input, sizeof( struct internal::packedBenchmarkerInput ), &data_out, sizeof( U ), fargs, 2 };

			// do hook
			const lpf_err_t spmdrc = lpf_hook( init, &(_grb_bench_varin_spmd< U >), args );

			// check error code
			if( spmdrc != LPF_SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		template< typename T, typename U >
		RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
			const T & data_in,
			U & data_out, // input & output data
			const size_t inner,
			const size_t outer,
			const bool broadcast = false ) {
			// prepare packed input
			struct internal::packedBenchmarkerInput input;
			input.blob = data_in;
			input.blob_size = sizeof( T );
			input.inner = inner;
			input.outer = outer;
			input.bcast_blob = broadcast;

			// prepare args
			lpf_func_t fargs[ 2 ];
			lpf_args_t args;
			fargs[ 0 ] = reinterpret_cast< lpf_func_t >( benchmark< T, U > );
			fargs[ 1 ] = reinterpret_cast< lpf_func_t >( grb_program );
			args = { &data_in, sizeof( T ), &data_out, sizeof( U ), fargs, 2 };

			// do hook
			const lpf_err_t spmdrc = lpf_hook( init, &(_grb_bench_spmd< T, U >), args );

			// check error code
			if( spmdrc != LPF_SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/** This implementation needs to release MPI resources in manual mode. */
		static enum RC finalize() {
			// done
			return Launcher< FROM_MPI, BSP1D >::finalize();
		}
	};

	template< enum EXEC_MODE mode >
	class Benchmarker< mode, BSP1D > : protected Launcher< mode, BSP1D >, protected internal::BenchmarkerBase {

	public:
		Benchmarker( const size_t process_id = 0,     // user process ID
			const size_t nprocs = 1,                  // total number of user processes
			const std::string hostname = "localhost", // one of the user process hostnames
			const std::string port = "0",             // a free port at hostname
			const bool is_mpi_inited = false ) :
			Launcher< mode, BSP1D >( process_id, nprocs, hostname, port, is_mpi_inited ) {}

		template< typename U >
		enum RC
		exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const size_t inner, const size_t outer, const bool broadcast = false )
			const {
			// prepare packed input
			struct internal::packedBenchmarkerInput input;
			input.blob = data_in;
			input.blob_size = in_size;
			input.inner = inner;
			input.outer = outer;
			input.bcast_blob = broadcast;

			// prepare args
			lpf_func_t fargs[ 2 ];
			lpf_args_t args;
			fargs[ 0 ] = reinterpret_cast< lpf_func_t >( benchmark< U > );
			fargs[ 1 ] = reinterpret_cast< lpf_func_t >( grb_program );
			args = { &input, sizeof( struct internal::packedBenchmarkerInput ), &data_out, sizeof( U ), fargs, 2 };

			// launch
			lpf_err_t spmdrc = LPF_SUCCESS;
			if( mode == MANUAL ) {
				// do hook
				spmdrc = lpf_hook( init, &(_grb_bench_varin_spmd< U >), args );
			} else {
				assert( mode == AUTOMATIC );
				// do exec
				spmdrc = lpf_exec( LPF_ROOT, LPF_MAX_P, &(_grb_bench_exec< void, U, true >), args );
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
		enum RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
			const T & data_in,
			U & data_out, // input & output data
			const size_t inner,
			const size_t outer,
			const bool broadcast = false ) {
			// prepare packed input
			struct internal::packedBenchmarkerInput input;
			input.blob = &data_in;
			input.blob_size = sizeof( T );
			input.inner = inner;
			input.outer = outer;
			input.bcast_blob = broadcast;

			// prepare args
			lpf_func_t fargs[ 2 ];
			lpf_args_t args;
			fargs[ 0 ] = reinterpret_cast< lpf_func_t >( benchmark< T, U > );
			fargs[ 1 ] = reinterpret_cast< lpf_func_t >( grb_program );
			args = { &input, sizeof( struct internal::packedBenchmarkerInput ), &data_out, sizeof( U ), fargs, 2 };

			// launch
			lpf_err_t spmdrc = LPF_SUCCESS;
			if( mode == MANUAL ) {
				// do hook
				spmdrc = lpf_hook( this->init, &(_grb_bench_spmd< T, U >), args );
			} else {
				assert( mode == AUTOMATIC );
				// do exec
				spmdrc = lpf_exec( LPF_ROOT, LPF_MAX_P, &(_grb_bench_exec< T, U, false >), args );
			}

			// check error code
			if( spmdrc != LPF_SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/** This implementation needs to release MPI resources in manual mode. */
		static enum RC finalize() {
			return Launcher< mode, BSP1D >::finalize();
		}
	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_BENCH''
