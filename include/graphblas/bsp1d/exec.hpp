
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
 * @author A. N. Yzelman; Alberto Scolari
 * @date 17th of April, 2017; 28 of August 2023
 */

#ifndef _H_GRB_BSP1D_EXEC
#define _H_GRB_BSP1D_EXEC

#include <memory>
#include <type_traits>
#include <atomic>
#include <stdexcept>
#include <functional>
#include <string>

#ifndef _GRB_NO_STDIO
 #include <iostream> //for std::cerr
#endif

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

#include "../bsp/exec_broadcast_routines.hpp"


/** Forward declaration. */

namespace grb {

	namespace internal {

		/**
		 * Base data structure storing necessary data to run a GRB function through LPF.
		 *
		 * @tparam InputType type of function input
		 */
		template< typename InputType >
		struct PackedExecInput {

			/** Pointer to input. */
			const InputType *input;

			/**
			 * Size of input.
			 *
			 * Equal to <tt>sizeof( InputType )</tt> for typed ALP functions.
			 */
			size_t input_size;

			/**
			 * Whether the user requested broadcast of input from the user process with
			 * node zero.
			 */
			bool broadcast_input;

			/**
			 * Default constructor, required by _grb_exec_dispatch() in case of object
			 * serialization.
			 */
			PackedExecInput() = default;

			/**
			 * Constructor with member initialization, for actual construction within
			 * node 0.
			 */
			PackedExecInput( const InputType *in, size_t s, bool bc ) :
				input( in ), input_size( s ), broadcast_input( bc )
			{}

		};

		/**
		 * Adaptor type to run a typed GRB function: it stores relevant parameters for
		 * data broadcast.
		 *
		 * Inherited from PackedExecInput.
		 *
		 * Adapts the function call to the underlying type.
		 */
		template< typename InputType, typename OutputType, bool variable_input >
		struct ExecDispatcher : PackedExecInput< InputType > {

			using PackedExecInput< InputType >::PackedExecInput;

			constexpr static bool is_input_size_variable = false;

			/**
			 * Static adapter for typed ALP functions.
			 *
			 * To be used also outside of this struct.
			 */
			static inline void lpf_grb_call(
				const lpf_func_t fun, size_t, const InputType *in,
				OutputType *out, lpf_pid_t, lpf_pid_t
			) {
				reinterpret_cast< AlpTypedFunc< InputType, OutputType > >( fun )
					( *in, *out );
			}

			/** Functor operator to call a typed ALP function. */
			inline grb::RC operator()(
				const lpf_func_t fun,
				size_t in_size, const InputType *in,
				OutputType *out,
				lpf_pid_t s, lpf_pid_t P
			) const {
				lpf_grb_call( fun, in_size, in, out, s, P );
				return grb::SUCCESS;
			}

		};

		/**
		 * Adaptor type to run an untyped ALP function.
		 *
		 * It stores relevant parameters for data braodcast (inherited from
		 * PackedExecInput) and adapts the function call to the underlying type.
		 */
		template< typename InputType, typename OutputType >
		struct ExecDispatcher< InputType, OutputType, true > :
			PackedExecInput< InputType >
		{

			using PackedExecInput< InputType >::PackedExecInput;

			constexpr static bool is_input_size_variable = true;

			static inline void lpf_grb_call(
				const lpf_func_t fun, size_t in_size,
				const InputType *in, OutputType *out,
				lpf_pid_t, lpf_pid_t
			) {
				reinterpret_cast< AlpUntypedFunc< InputType, OutputType > >( fun )
					( in, in_size, *out );
			}

			inline grb::RC operator()(
				const lpf_func_t fun, size_t in_size,
				const InputType *in, OutputType *out,
				lpf_pid_t s, lpf_pid_t P
			) const {
				lpf_grb_call( fun, in_size, in, out, s, P );
				return grb::SUCCESS;
			}

		};

		/**
		 * Allocator for data structures: if \p typed_allocation is \a true, then
		 * allocate \p T on the heap via its default contructor \p T(), otherwise as a
		 * byte array (without construction).
		 */
		template< typename T, bool typed_allocation >
		struct exec_allocator {

			// internal: this is a reasonable assert, since T should be a POD type
			static_assert( std::is_default_constructible< T >::value,
				"T must be default constructible" );

			typedef std::function< void( T * ) > Deleter;
			typedef std::unique_ptr< T, Deleter > PointerHolder;

			static PointerHolder make_pointer( size_t ) {
				return std::unique_ptr< T, Deleter >(
					new T(), // allocate with default construction
					[] ( T * ptr ) { delete ptr; }
				);
			}
		};

		/**
		 * Template specialization for untyped allocation: data is allocated as a byte
		 * array * and not initialized.
		 */
		template< typename T >
		struct exec_allocator< T, false > {

			typedef std::function< void( T * ) > Deleter;
			typedef std::unique_ptr< T, Deleter > PointerHolder;

			static PointerHolder make_pointer( size_t size ) {
				return std::unique_ptr< T, Deleter >( reinterpret_cast< T * >( new char[ size ] ),
					[] ( T * ptr ) { delete [] reinterpret_cast< char * >( ptr ); } );
			}
		};


		/**
		 * Dispatcher to be called via LPF for distributed execution of an ALP
		 * function.
		 *
		 * It handles type information of the called function via the
		 * \p DispatcherType structure.
		 *
		 * This call may perform memory allocations and initializations depending
		 * on several conditions; in general, it performs these operations only
		 * if strictly needed and if \a sensible.
		 *
		 * Depending on the \p mode type parameter, it attempts to create an input
		 * data structure if this is not available. This is especially important
		 * in AUTOMATIC mode, where processes with \p s > 0 have no data
		 * pre-allocated. In AUTOMATIC mode, indeed, this function does its best
		 * to supply the user function with input data:
		 * - if broadcast was requested, data must be copied from the node with
		 *  s == 0 to the other nodes; memory on s > 0 is allocated via \p T's
		 * default constructor if possible, or as a byte array; in the end,
		 * data on s > 0 is anyway overwritten by data from s == 0;
		 * - if broadcast was not requested, this function strives to allocate
		 *  a \a sensible input by calling \p T's default constructor if possible;
		 *  if this is not possible, the execution is aborted.
		 *
		 * For modes other than AUTOMATIC, typed ALP functions are assumed to
		 * always have a pre-allocated input, allocated by the function that
		 * "hooked" into LPF; no memory is allocated in this case. If broadcast
		 * is requested, the input for s > 0 is simply overwritten with that from
		 * s == 0. For untyped functions, memory is allocated only if broadcast
		 * is requested (because the size is not known a priori), otherwise no
		 * allocation occurs and each ALP function takes the original input from
		 * the launching function.
		 *
		 * @tparam T              ALP function input type.
		 * @tparam U              ALP function outut type.
		 * @tparam mode           grb::EXEC_MODE of the LPF call.
		 * @tparam DispatcherType Information on the ALP function to run.
		 *
		 * @param[in,out] ctx  LPF context to run in.
		 * @param[in] s        User process identifier (in the range [0, P)).
		 * @param[in] P        Number of parallel processes.
		 * @param[in,out] args Input and output information for LPF calls.
		 */
		template<
			typename T, typename U,
			enum EXEC_MODE mode,
			typename DispatcherType
		>
		void _grb_exec_dispatch(
			lpf_t ctx,
			const lpf_pid_t s, const lpf_pid_t P,
			lpf_args_t args
		) {
			static_assert(
				std::is_base_of< ExecDispatcher< T, U, true >, DispatcherType >::value ||
					std::is_base_of< ExecDispatcher< T, U, false >, DispatcherType >::value,
				"DispatcherType must derive from ExecDispatcher"
			);
			static_assert( std::is_default_constructible< DispatcherType >::value,
				"DispatcherType must be default-constructible" );


			static_assert( std::is_same< T, void >::value || std::is_standard_layout< T >::value, "crap" );

			constexpr bool is_typed_alp_prog = not DispatcherType::is_input_size_variable;
			constexpr bool is_input_def_constructible = std::is_default_constructible< T >::value;

			assert( P > 0 );
			assert( s < P );
#ifdef _DEBUG
			if( s == 0 ) {
				std::cout << "Info: launcher spawned or hooked " << P << " ALP user "
					<< "processes.\n";
			}
#endif
			// call information for the ALP function
			const DispatcherType *dispatcher =
				static_cast< const DispatcherType* >( args.input );
			std::unique_ptr< DispatcherType > dispatcher_holder;

			lpf_coll_t coll;
			lpf_err_t brc = LPF_SUCCESS;
			if( P > 1 ) {
				if( mode == AUTOMATIC ) {
					if( s > 0 ) {
						// this is not the root process -- potentially need to allocate memory for
						// the dispatcher
						dispatcher_holder.reset( new DispatcherType );
						dispatcher = dispatcher_holder.get();
					}

					// AUTOMATIC mode: we must
					//  1. initialize communication
					brc = lpf_init_collectives_for_broadcast( ctx, s, P, 2, coll );
					if( brc != LPF_SUCCESS ) {
						std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
							<< std::endl;
					}
					assert( brc == LPF_SUCCESS );

					// then 2. fetch the dispatcher
					brc = lpf_register_and_broadcast(
						ctx, coll,
						const_cast< void * >( reinterpret_cast< const void * >( dispatcher ) ),
						sizeof( DispatcherType )
					);
					if( brc != LPF_SUCCESS ) {
						std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
							<< std::endl;
					}
					assert( brc == LPF_SUCCESS );
				} else if( dispatcher->broadcast_input ) {
					// the dispatcher is already valid and the user requested broadcasting: init communication
					brc = lpf_init_collectives_for_broadcast( ctx, s, P, 2, coll );
					if( brc != LPF_SUCCESS ) {
						std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
							<< std::endl;
					}
					assert( brc == LPF_SUCCESS );
				}
			}


			if(
				not is_input_def_constructible &&
				is_typed_alp_prog &&
				mode == AUTOMATIC &&
				not dispatcher->broadcast_input &&
				P > 1
			) {
				std::cerr <<
					"Error: cannot locally construct input type"
					" for ALP program in AUTOMATIC mode" << std::endl;
				return;
			}

			// dispatcher is now valid: assign initial value for size
			size_t in_size = dispatcher->input_size;
			if( P > 1 ) {
				if(
					mode != AUTOMATIC &&
					dispatcher->broadcast_input
					&& !is_typed_alp_prog
				) {
					// user requested broadcast and the input size is user-given:
					//   fetch size from master
					in_size = s == 0 ? dispatcher->input_size : 0UL;
					lpf_err_t brc = lpf_register_and_broadcast(
							ctx, coll,
							reinterpret_cast< void * >( &in_size ), sizeof( size_t )
						);
					if( brc != LPF_SUCCESS ) {
						std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
							<< std::endl;
					}
					assert( brc == LPF_SUCCESS );
					assert( in_size != 0 );
				} else if (
					mode == AUTOMATIC &&
					not is_typed_alp_prog &&
					not dispatcher->broadcast_input && s > 0
				) {
					// AUTOMATIC mode, untyped, no broadcast: we cannot
					//  reconstruct the object, so pass 0 as size
					in_size = 0;
				}
			}

			constexpr bool typed_alloc = is_typed_alp_prog &&
				is_input_def_constructible;
			typedef exec_allocator< T, typed_alloc > InputAllocator;

			typename InputAllocator::PointerHolder data_in_holder;
			// input data: by default user-given input
			const T * data_in =
				reinterpret_cast< const T * >( dispatcher->input );

			if( s > 0 ) {
				if (
					mode == AUTOMATIC &&
					not is_typed_alp_prog &&
					not dispatcher->broadcast_input
				) {
					// AUTOMATIC mode, untyped, no broadcast: we cannot
					//  reconstruct the object, so pass nullptr
					data_in = nullptr;
				} else if(
					mode == AUTOMATIC ||
					( dispatcher->broadcast_input &&
						not is_typed_alp_prog )
				) {
					// if no memory exists (mode == AUTOMATIC) or the size was not known and
					// the user requested broadcast, then allocate input data
					data_in_holder = InputAllocator::make_pointer( in_size );
					data_in = data_in_holder.get();
				}
			}
			if( dispatcher->broadcast_input && P > 1 ) {
				// retrieve data from master
				lpf_err_t brc = lpf_register_and_broadcast(
						ctx, coll,
						const_cast< void * >( reinterpret_cast< const void * >( data_in ) ),
						in_size
					);
					if( brc != LPF_SUCCESS ) {
						std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
							<< std::endl;
					}
					assert( brc == LPF_SUCCESS );
			}
			typedef exec_allocator< U, std::is_default_constructible< U >::value >
				OutputAllocator;
			typename OutputAllocator::PointerHolder data_out_holder;
			U * data_out = reinterpret_cast< U * >( args.output );
			if( mode == AUTOMATIC && s > 0 ) {
				data_out_holder = OutputAllocator::make_pointer( sizeof( U ) );
				data_out = reinterpret_cast< U * >( data_out_holder.get() );
			}
			// initialise ALP/GraphBLAS
			grb::RC grb_rc = grb::init< BSP1D >( s, P, ctx );
			if( grb_rc != grb::SUCCESS ) {
				std::cerr << "Error: could not initialise ALP/GraphBLAS" << std::endl;
				assert( false );
				return;
			}
			// retrieve and run the function to be executed
			assert( args.f_size == 1 );
			grb_rc = ( *dispatcher )( args.f_symbols[ 0 ], in_size, data_in, data_out, s, P  );
			if( grb_rc != grb::SUCCESS ) {
				std::cerr << "Error: dispatcher failed" << std::endl;
				assert( false );
				return;
			}

			// finalise ALP/GraphBLAS
			grb_rc = grb::finalize< BSP1D >();
			if( grb_rc != grb::SUCCESS ) {
				std::cerr << "Error: could not finalise ALP/GraphBLAS" << std::endl;
				assert( false );
			}
		}

	} // namespace internal

	/**
	 * Base class for Launcher's, with common logic and information; mainly
	 * wrapping user #exec() parameters into internal data structures and calling
	 * LPF.
	 *
	 * @tparam mode grb::EXEC_MODE LPF execution mode
	 */
	template< enum EXEC_MODE mode >
	class BaseLauncher {

		private:

			/**
			 * Pack data received from user into an internal::ExecDispatcher data
			 * structure and run the LPF call.
			 *
			 * @tparam T            Input type.
			 * @tparam U            Output type.
			 * @tparam untyped_call Whether the ALP function is typed.
			 *
			 * @param[in] alp_program ALP function to run in parallel.
			 * @param[in] data_in     Pointer to input data.
			 * @param[in] in_size     Size of the input data
			 *
			 * \note \a in_size equals <tt>sizeof( T )</tt> if \a untyped_call equals
			 *       <tt>false</tt>.
			 *
			 * @param[out] data_out  Pointer to where to write output data.
			 * @param[in]  broadcast Whether to broadcast input from node 0 to all
			 *                       others.
			 *
			 * @return RC status code of the LPF call.
			 */
			template< typename T, typename U, bool untyped_call >
			RC pack_data_and_run(
				const lpf_func_t alp_program,
				const T *data_in,
				const size_t in_size,
				U * const data_out,
				const bool broadcast
			) {
				typedef internal::ExecDispatcher< T, U, untyped_call > Disp;
				Disp disp_info = { data_in, in_size, broadcast };
				return run_lpf< T, U, Disp >( alp_program, disp_info, data_out );
			}


		protected:

			/** The LPF init struct. Will be initialised during construction. */
			lpf_init_t init;

			BaseLauncher() = default;

			/** Disable copy constructor. */
			BaseLauncher( const BaseLauncher< mode > & ) = delete;

			/** Disable copy constructor. */
			BaseLauncher & operator=( const BaseLauncher< mode > & ) = delete;

			/**
			 * Run the given \p alp_program with the given input information \p disp_info
			 * via LPF.
			 *
			 * @tparam T              Input type.
			 * @tparam U              Output type.
			 * @tparam DispatcherType Type of the data structure that holds input and
			 *                        call information.
			 *
			 * @param[in]  alp_program ALP function to run distributed.
			 * @param[in]  disp_info   Data structure that holds input and call
			 *                         information.
			 * @param[out] data_out    Pointer to where to write output.
			 *
			 * @return RC status code of the LPF call.
			 */
			template< typename T, typename U, typename DispatcherType >
			RC run_lpf(
				const lpf_func_t alp_program,
				const DispatcherType &disp_info,
				U * const data_out
			) const {
				lpf_args_t args = {
					&disp_info, sizeof( DispatcherType ),
					data_out, sizeof( U ),
					&alp_program, 1
				};

				lpf_spmd_t fun = reinterpret_cast< lpf_spmd_t >(
					internal::_grb_exec_dispatch< T, U, mode, DispatcherType > );

				const lpf_err_t spmdrc = init == LPF_INIT_NONE
					? lpf_exec( LPF_ROOT, LPF_MAX_P, fun, args )
					: lpf_hook( init, fun, args );

				// check error code
				if( spmdrc != LPF_SUCCESS ) {
					return PANIC;
				}

				return SUCCESS;
			}

		public:

			/**
			 * Run a typed ALP function distributed via LPF.
			 *
			 * In case of AUTOMATIC mode, input data is allocated by default (if the type
			 * allows) or as a sequence of bytes. This assumes the default allocator does
			 * not have \b any side affect (like memory allocation). In case of broadcast
			 * request, data is trivially serialized: hence, non-trivial objects (e.g.,
			 * storing pointers to memory buffers) are not valid anymore in processes
			 * other than the master.
			 *
			 * @tparam T Input type.
			 * @tparam U Output type.
			 *
			 * @param[in]  alp_program ALP function to run in parallel.
			 * @param[in]  data_in     Input data.
			 * @param[out] data_out    Output data.
			 * @param[in]  broadcast   Whether to broadcast input from node 0 to the
			 *                         others.
			 *
			 * @return RC status code of the LPF call
			 */
			template< typename T, typename U >
			RC exec(
				const AlpTypedFunc< T, U > alp_program,
				const T &data_in,
				U &data_out,
				const bool broadcast = false
			) {
				return pack_data_and_run< T, U, false >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					&data_in, sizeof( T ), &data_out, broadcast
				);
			}

			/**
			 * Run an untyped ALP function in parallel via LPF.
			 *
			 * Input data has variable size, known only at runtime. Therefore, input
			 * data cannot be costructed by default, but are serialized and replicated as
			 * a mere sequence of bytes.
			 *
			 * @tparam T Input type.
			 * @tparam U Output type.
			 *
			 * @param[in] alp_program ALP function to run in parallel.
			 * @param[in] data_in     Pointer to input data.
			 * @param[in] in_size     Size of input data.
			 * @param[out] data_out   Output data.
			 * @param[in]  broadcast  Whether to broadcast input from node 0 to the
			 *                        others.
			 *
			 * @return RC status code of the LPF call.
			 */
			template< typename U >
			RC exec(
				const AlpUntypedFunc< void, U > alp_program,
				const void * const data_in, const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) {
				return pack_data_and_run< void, U, true >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					data_in, in_size, &data_out, broadcast
				);
			}

	};

	/**
	 * Specialization of Launcher to be used when MPI has already been
	 * initialised but not LPF.
	 */
	template<>
	class Launcher< FROM_MPI, BSP1D > : public BaseLauncher< FROM_MPI > {

		public:

			/**
			 * No implementation notes.
			 *
			 * @param[in] MPI communicator to hook into.
			 *
			 * @throws runtime_error When a standard MPI call fails.
			 */
			Launcher( const MPI_Comm comm = MPI_COMM_WORLD ) {
				// init from communicator
				const lpf_err_t initrc = lpf_mpi_initialize_with_mpicomm( comm, &init );

				// check for success
				if( initrc != LPF_SUCCESS ) {
					throw std::runtime_error(
						"LPF could not be initialized via the given MPI communicator."
					);
				}
				// done!
			}

			/**
			 * Implementation note: this Launcher will clear #init.
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
	 * Specialization of Launcher to be used when both MPI and LPF have already been initialized.
	 */
	template<>
	class Launcher< AUTOMATIC, BSP1D > : public BaseLauncher< AUTOMATIC > {

		public:

			Launcher() {
				this->init = LPF_INIT_NONE;
			}

			~Launcher() {
				assert( init == LPF_INIT_NONE );
			}

			static RC finalize() {
				return grb::SUCCESS;
			}

	};

	/**
	 * Specialisation of Launcher to be used when whishing to create an LPF context
	 * by manually connecting nodes together via TCP.
	 *
	 * The detection of the available nodes is done via MPI, which can be already
	 * initialised, or can be initialised here.
	 */
	template< enum EXEC_MODE mode >
	class Launcher< mode, BSP1D > : public BaseLauncher< mode > {

		static_assert( mode == MANUAL, "mode is not manual" );

		public:

			/**
			 * When \a mode is #AUTOMATIC, this implementation adheres to the base
			 * specification. When \a mode is #MANUAL, this implementation specifies
			 * additionally the following:
			 *
			 * The time-out of this constructor is two minutes.
			 *
			 * @param[in] process_id User process ID.
			 * @param[in] nprocs     Total number of user processes.
			 * @param[in] hostname   Host name (or IP) of one of the user processes
			 *                       involved in the collective construction of this
			 *                       launcher. May not be empty.
			 * @param[in] port       A free port for connecting to \a hostname during the
			 *                       collective construction of this launcher. May not be
			 *                       empty. Must be either a port number of a registered
			 *                       service name.
			 *
			 * \note If giving a \a hostname as a string, it must resolve to an IP; if
			 *       resolution fails, this constructor call will fail.
			 *
			 * \note If giving a \a port as a string, it must resolve to a port number;
			 *       if resolution fails, this constructor call will fail.
			 *
			 * In addition to the standard-defined exceptions, the following may
			 * additionally be thrown:
			 *
			 * @throws invalid_argument When hostname or port are empty.
			 * @throws runtime_error    When the requested launcher group could not be
			 *                          created.
			 */
			Launcher(
				const size_t process_id = 0,
				const size_t nprocs = 1,
				const std::string &hostname = "localhost",
				const std::string &port = "0",
				const bool is_mpi_inited = false
			) {
				// sanity check
				if( nprocs == 0 ) {
					throw std::invalid_argument( "Total number of user processes must be "
						"strictly larger than zero." );
				}
				if( process_id >= nprocs ) {
					throw std::invalid_argument( "Process ID must be strictly smaller than "
						"total number of user processes." );
				}

				// initialise MPI if not already done
				// FIXME the MPI_Init should not be here. See GitHub issue #240.
				if( !is_mpi_inited && !internal::grb_mpi_initialized ) {
					if( MPI_Init( NULL, NULL ) != MPI_SUCCESS ) {
						throw std::runtime_error( "Call to MPI_Init failed." );
					} else {
						internal::grb_mpi_initialized = true;
					}
				}

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
					&(this->init)
				);

				// check for success
				if( initrc != LPF_SUCCESS ) {
#ifndef _GRB_NO_STDIO
					throw std::runtime_error(
						"LPF could not connect launcher group over TCP/IP."
					);
#endif
				}
			}

			/**
			 * Implementation note: this Launcher may need to clear a field of type
			 * \a lpf_init_t when used in MANUAL mode.
			 */
			~Launcher() {
				assert( this->init != LPF_INIT_NONE );
				// try and destroy the lpf_init_t
				const lpf_err_t finrc = lpf_mpi_finalize( this->init );
				if( finrc != LPF_SUCCESS ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Warning: could not destroy launcher::init from ~launcher.\n";
#endif
				}
				this->init = LPF_INIT_NONE;
			}

			/**
			 * This implementation needs to release MPI resources in manual mode.
			 */
			static RC finalize() {
				// finalise MPI when in manual mode
				if( internal::grb_mpi_initialized && MPI_Finalize() != MPI_SUCCESS ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Warning: MPI_Finalize returned non-SUCCESS exit code.\n";
#endif
					return grb::PANIC;
				}
				internal::grb_mpi_initialized = false;
				return grb::SUCCESS;
			}

	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_EXEC''

