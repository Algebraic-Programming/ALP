
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

#include <atomic>
#include <memory>
#include <string>
#include <typeinfo>
#include <stdexcept>
#include <functional>
#include <type_traits>

#ifndef _GRB_NO_STDIO
 #include <iostream> //for std::cerr
#endif

#include <lpf/collectives.h>
#include <lpf/core.h>
#include <lpf/mpi.h>

#include <mpi.h> //for EXEC_MODE::FROM_MPI support

#include <graphblas/rc.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/base/exec.hpp>

#include <graphblas/collectives.hpp>

#include "init.hpp"

#include "../bsp/exec_broadcast_routines.hpp"


namespace grb {

	namespace internal {

		/**
		 * Base data structure storing necessary data to run an ALP function through
		 * LPF.
		 *
		 * @tparam InputType            The type of function input.
		 * @tparam mode                 The grb::EXEC_MODE of the launcher.
		 * @tparam _requested_broadcast Whether inputs shall be broadcast.
		 */
		template<
			typename InputType,
			EXEC_MODE _mode,
			bool _requested_broadcast
		>
		struct DispatchInfo {

			/** Make available the launcher mode. */
			static constexpr EXEC_MODE mode = _mode;

			/** Make available whether input broadcast was requested. */
			static constexpr bool requested_broadcast = _requested_broadcast;

			/** Note: benchmarker classes may require initial broadcasts */
			static constexpr bool needs_initial_broadcast = false;

			/** Pointer to input argument. */
			const InputType * in;

			/** Byte size of input argument. */
			size_t in_size;

			/**
			 * Construct from base information.
			 *
			 * @param[in] _in      Pointer to the input argument.
			 * @param[in] _in_size Byte size of the input argument.
			 */
			DispatchInfo( const InputType * const _in, const size_t _in_size ) :
				in( _in ), in_size( _in_size )
			{}

			/**
			 * Construct from LPF arguments, following a call to lpf_hook() or
			 * lpf_exec().
			 *
			 * @param[in] s    The user process ID.
			 * @param[in] args The LPF I/O arguments.
			 */
			DispatchInfo( const lpf_pid_t s, const lpf_args_t args ) {
				if( s > 0 && mode == AUTOMATIC ) {
					in = nullptr;
					in_size = 0;
				} else {
					in = static_cast< const InputType *>( args.input );
					in_size = args.input_size;
				}
			}

			/** @returns in */
			const InputType * get_input() const { return in; }

			/** @returns in_size */
			size_t get_input_size() const { return in_size; }

		};

		/**
		 * Adaptor to run a typed ALP function: it stores relevant parameters for data
		 * broadcast.
		 *
		 * Inherited from DispatchInfo.
		 *
		 * Adapts the function call to the underlying type.
		 */
		template<
			typename InputType, typename OutputType,
			EXEC_MODE _mode,
			bool _requested_broadcast, bool _variable_input
		>
		class ExecDispatcher :
			public DispatchInfo< InputType, _mode, _requested_broadcast >
		{

			protected:

				/**
				 * Static adapter for typed ALP functions.
				 *
				 * Casts and calls the opaque \a fun function.
				 *
				 * This function is factored out so as to allow its call from the BSP
				 * #grb::Benchmarker.
				 *
				 * @param[in]  fun     Pointer to the typed ALP function.
				 * @param[in]  s       The user process ID.
				 * @param[in]  P       The total number of user processes.
				 * @param[in]  in      Pointer to the input argument.
				 * @param[in]  in_size Byte size of the input argument.
				 * @param[out] out     Pointer to where to store the output.
				 */
				static inline void lpf_grb_call(
					const lpf_func_t fun,
					const lpf_pid_t s, const lpf_pid_t P,
					const InputType * const in,
					const size_t in_size,
					OutputType *out
				) {
					(void) in_size;
					(void) s;
					(void) P;
					reinterpret_cast< AlpTypedFunc< InputType, OutputType > >( fun )
						( *in, *out );
				}


			public:

				/** Use base constructor */
				using DispatchInfo< InputType, _mode, _requested_broadcast >::DispatchInfo;

				/** Typed dispatching has static size inputs */
				constexpr static bool is_input_size_variable = false;

				/**
				 * Functor operator to call a typed ALP function.
				 *
				 * @param[in]  fun     Pointer to the typed ALP function.
				 * @param[in]  s       The user process ID.
				 * @param[in]  P       The total number of user processes.
				 * @param[in]  in      Pointer to the input argument.
				 * @param[in]  in_size Byte size of the input argument.
				 * @param[out] out     Pointer to where to store the output.
				 */
				inline grb::RC operator()(
					const lpf_func_t fun,
					const lpf_pid_t s, const lpf_pid_t P,
					const InputType *in, const size_t in_size,
					OutputType * out
				) const {
					lpf_grb_call( fun, s, P, in, in_size, out );
					return grb::SUCCESS;
				}

		};

		/**
		 * Adaptor to run an untyped ALP function.
		 *
		 * It stores relevant parameters for data broadcast (inherited from
		 * DispatchInfo) and adapts the function call to the underlying type.
		 */
		template<
			typename OutputType,
			EXEC_MODE _mode,
			bool _requested_broadcast
		>
		class ExecDispatcher< void, OutputType, _mode, _requested_broadcast, true > :
			public DispatchInfo< void, _mode, _requested_broadcast >
		{

			protected:

				/**
				 * Calls an untyped ALP function.
				 *
				 * Factored out as a separate function to allow its use from the BSP
				 * #grb::Benchmarker.
				 *
				 * @param[in]  fun     Pointer to the untyped ALP function.
				 * @param[in]  s       The user process ID.
				 * @param[in]  P       The total number of user processes.
				 * @param[in]  in      Pointer to the input argument.
				 * @param[in]  in_size Byte size of the input argument.
				 * @param[out] out     Pointer to where to store the output.
				 */
				static inline void lpf_grb_call(
					const lpf_func_t fun,
					const lpf_pid_t s, const lpf_pid_t P,
					const void * const in, const size_t in_size,
					OutputType * const out
				) {
					(void) s;
					(void) P;
					reinterpret_cast< AlpUntypedFunc< OutputType > >( fun )
						( in, in_size, *out );
				}


			public:

				/** Use base class constructor. */
				using DispatchInfo< void, _mode, _requested_broadcast >::DispatchInfo;

				/** Untyped inputs have variably-sized inputs. */
				constexpr static bool is_input_size_variable = true;

				/**
				 * Functor operator to call an untyped ALP function.
				 *
				 * @param[in]  fun     Pointer to the untyped ALP function.
				 * @param[in]  s       The user process ID.
				 * @param[in]  P       The total number of user processes.
				 * @param[in]  in      Pointer to the input argument.
				 * @param[in]  in_size Byte size of the input argument.
				 * @param[out] out     Pointer to where to store the output.
				 */
				inline grb::RC operator()(
					const lpf_func_t fun,
					lpf_pid_t s, lpf_pid_t P,
					const void * const in, const size_t in_size,
					OutputType * const out
				) const {
					lpf_grb_call( fun, s, P, in, in_size, out );
					return grb::SUCCESS;
				}

		};

		/**
		 * Allocator for data structures: if \a typed_allocation is \a true, then
		 * allocate \a T on the heap via its default contructor \a T(), otherwise as a
		 * byte array (without construction).
		 *
		 * @tparam T The type of the object that should be allocated.
		 *
		 * @tparam typed_allocation Whether or not we may rely on the default
		 *                          constructor of \a T.
		 *
		 * This allocator is only used for typed ALP functions.
		 */
		template< typename T, bool typed_allocation >
		struct ExecAllocator {

			static_assert( std::is_default_constructible< T >::value,
				"T must be default constructible" );

			typedef std::function< void( T * ) > Deleter;
			typedef std::unique_ptr< T, Deleter > PointerHolder;

			static PointerHolder make_pointer( size_t ) {
				return PointerHolder(
					new T(), // allocate with default construction
					[] ( T * const ptr ) { delete ptr; }
				);
			}

		};

		/**
		 * Template specialisation for untyped allocation: data is allocated as a byte
		 * array and not initialised.
		 *
		 * This allocator is used for launching untyped ALP programs \em and may be
		 * used for launching typed ALP programs where inputs are not-default
		 * constructible but copiable. The latter only applies in broadcasting mode.
		 */
		template< typename T >
		struct ExecAllocator< T, false > {

			typedef std::function< void( T * ) > Deleter;
			typedef std::unique_ptr< T, Deleter > PointerHolder;

			static PointerHolder make_pointer( const size_t size ) {
				return PointerHolder( reinterpret_cast< T * >( new char[ size ] ),
					[] ( T * const ptr ) { delete [] reinterpret_cast< char * >( ptr ); } );
			}

		};

		/**
		 * Dispatcher to be called via LPF for distributed execution of an ALP
		 * function.
		 *
		 * It handles type information of the called function via the
		 * \a DispatcherType structure.
		 *
		 * This call may perform memory allocations and initialisations depending
		 * on several conditions; in general, it performs these operations only
		 * if strictly needed.
		 *
		 * Depending on the \a mode type parameter, it attempts to create an input
		 * data structure if this is not available. This is especially important
		 * in AUTOMATIC mode, where processes with \a s > 0 have no data
		 * pre-allocated.
		 *
		 * In AUTOMATIC mode, indeed, this function does its best to supply the user
		 * function with input data:
		 * - if broadcast was requested, data must be copied from the node with
		 *  s == 0 to the other nodes; memory on s > 0 is allocated via \a T's
		 * default constructor if possible, or as a byte array; in the end,
		 * data on s > 0 is anyway overwritten by data from s == 0;
		 * - if broadcast was not requested, this function allocates sensible input
		 *   by calling \a T's default constructor, if possible. If this is not
		 *   possible, the call to this function shall have no other effect than
		 *   (immediately) returning #grb::ILLEGAL.
		 *
		 * For modes other than AUTOMATIC, typed ALP functions are assumed to
		 * always have a pre-allocated input, allocated by the function that
		 * \em hooked into LPF; no memory is allocated in this case. If broadcast
		 * is requested, the input for s > 0 is simply overwritten with that from
		 * s == 0. For untyped functions, memory is allocated only if broadcasting
		 * is requested (because the size is known a priori only at user process 0),
		 * otherwise no allocation occurs and each ALP function takes the original
		 * input from the launching function.
		 *
		 * \note Thus, implicitly, if in #grb::MANUAL or in #grb::FROM_MPI modes with
		 *       \a broadcast <tt>true</tt>, any input pointers at user processes
		 *       \f$ s > 0 \f$ will be ignored.
		 *
		 * @tparam T              ALP function input type.
		 * @tparam U              ALP function outut type.
		 * @tparam DispatcherType Information on the ALP function to run.
		 *
		 * @param[in,out] ctx  LPF context to run in.
		 * @param[in] s        User process identifier (in the range [0, P)).
		 * @param[in] P        Number of parallel processes.
		 * @param[in,out] args Input and output information for LPF calls.
		 */
		template<
			typename T, typename U,
			typename DispatcherType
		>
		void alp_exec_dispatch(
			lpf_t ctx,
			const lpf_pid_t s, const lpf_pid_t P,
			lpf_args_t args
		) {
			static_assert(
				std::is_same< T, void >::value ||
					std::is_trivially_copyable< T >::value ||
					std::is_standard_layout< T >::value,
				"The input type \a T must be void or memcpy-able (trivially copyable or"
				"standard layout)."
			);

			constexpr bool is_typed_alp_prog = !(DispatcherType::is_input_size_variable);
			constexpr bool is_input_def_constructible =
				std::is_default_constructible< T >::value;
			constexpr grb::EXEC_MODE mode = DispatcherType::mode;
			constexpr bool broadcast_input = DispatcherType::requested_broadcast;
			constexpr bool dispatcher_needs_broadcast =
				DispatcherType::needs_initial_broadcast;

			assert( P > 0 );
			assert( s < P );
#ifdef _DEBUG
			if( s == 0 ) {
				std::cout << "Info: launcher spawned or hooked " << P << " ALP user "
					<< "processes.\n";
			}
#endif
			if(
				!is_input_def_constructible &&
				is_typed_alp_prog &&
				mode == AUTOMATIC &&
				!broadcast_input &&
				P > 1
			) {
				std::cerr << "Error: cannot locally construct input type (typeid name \""
					<< typeid(T).name() << "\"for an ALP program that is launched "
					<< "in automatic mode, with broadcasting, and using more than one user"
					<< "one user process.\n"
					<< "Additionally, this error should have been caught prior to the "
					<< "attempted launch of the ALP program-- please submit a bug report."
					<< std::endl;
				assert( false );
				return;
			}

			lpf_coll_t coll;
			lpf_err_t brc = LPF_SUCCESS;

			// initialise collectives if they are needed
			if( P > 1 && (broadcast_input || dispatcher_needs_broadcast) ) {
				brc = lpf_init_collectives_for_broadcast( ctx, s, P, 2, coll );
				if( brc != LPF_SUCCESS ) {
					std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
						<< std::endl;
				}
				assert( brc == LPF_SUCCESS );
			}

			// call information for the ALP function, reconstructed from the arguments
			DispatcherType dispatcher( s, args );

			// ensure dispatcher is valid
			if( P > 1 && dispatcher_needs_broadcast ) {
				// fetch the dispatcher
				brc = lpf_register_and_broadcast(
					ctx, coll,
					static_cast< void * >( &dispatcher ),
					sizeof( DispatcherType )
				);
				if( brc != LPF_SUCCESS ) {
					std::cerr << __FILE__ << ", " << __LINE__ << ": LPF collective failed"
						<< std::endl;
				}
				assert( brc == LPF_SUCCESS );
			}

			// dispatcher is now valid on all processes: assign initial value for size
			size_t in_size = dispatcher.get_input_size();

			// set in_size on user processes with IDs larger than 0
			if( P > 1 ) {
				// check if input args should come from PID 0
				if( broadcast_input ) {
					// user requested broadcast and the input size is user-given: fetch size
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
				} else if( mode == AUTOMATIC && !broadcast_input && s > 0 ) {
					// AUTOMATIC mode, untyped, no broadcast: pass zero as size
					in_size = 0;
				}
			}

			// now set the input argument (in) itself
			constexpr bool typed_alloc = is_typed_alp_prog && is_input_def_constructible;
			typedef ExecAllocator< T, typed_alloc > InputAllocator;
			typename InputAllocator::PointerHolder data_in_holder;

			// set default value
			const T * data_in = dispatcher.get_input();

			// set in on user processes with IDs larger than 0
			if( s > 0 ) {
				if( mode == AUTOMATIC && !is_typed_alp_prog && !broadcast_input ) {
					// AUTOMATIC mode, untyped, no broadcast: pass nullptr
					data_in = nullptr;
				} else if( mode == AUTOMATIC || (broadcast_input && !is_typed_alp_prog) ) {
					// if no memory exists (mode == AUTOMATIC) or the size was not known and
					// the user requested broadcast, then allocate input data
					data_in_holder = InputAllocator::make_pointer( in_size );
					data_in = data_in_holder.get();
				}
			}

			// set contents of in
			if( broadcast_input && P > 1 ) {
				// retrieve data
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

			// now set the output argument
			typedef ExecAllocator< U, std::is_default_constructible< U >::value >
				OutputAllocator;
			typename OutputAllocator::PointerHolder data_out_holder;

			// set default value
			U * data_out = reinterpret_cast< U * >( args.output );

			// set out on user processes with ID larger than 0
			if( mode == AUTOMATIC && s > 0 ) {
				// allocate output if memory does not exist
				data_out_holder = OutputAllocator::make_pointer( sizeof( U ) );
				data_out = reinterpret_cast< U * >( data_out_holder.get() );
			}

			// at this point, the dispatcher, input, and output are all good to go

			// now, initialise ALP
			grb::RC grb_rc = grb::init< BSP1D >( s, P, ctx );
			if( grb_rc != grb::SUCCESS ) {
				std::cerr << "Error: could not initialise ALP/GraphBLAS" << std::endl;
				assert( false );
				return;
			}

			// retrieve and run the function to be executed
			assert( args.f_size == 1 );
			grb_rc = dispatcher( args.f_symbols[ 0 ], s, P, data_in, in_size, data_out );
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

		/**
		 * Base class for Launcher's, with common logic and information; mainly
		 * wrapping user #exec() parameters into internal data structures and calling
		 * LPF.
		 *
		 * @tparam mode grb::EXEC_MODE LPF execution mode
		 */
		template< enum EXEC_MODE mode >
		class BaseLpfLauncher {

			protected:

				/** The LPF init struct. Will be initialised during construction. */
				lpf_init_t init;

				/** Base constructor. */
				BaseLpfLauncher() : init( LPF_INIT_NONE ) {}

				/** Disable copy constructor. */
				BaseLpfLauncher( const BaseLpfLauncher< mode > & ) = delete;

				/** Disable copy constructor. */
				BaseLpfLauncher & operator=( const BaseLpfLauncher< mode > & ) = delete;

				/**
				 * Run the given \a alp_program with the given pointers to input and output
				 * arguments.
				 *
				 * @tparam T              Input type.
				 * @tparam U              Output type.
				 * @tparam DispatcherType Type of the data structure that holds input and
				 *                        call information.
				 *
				 * @param[in]  alp_program The ALP program to execute.
				 * @param[in]  data_in     Pointer to the input argument.
				 * @param[in]  in_size     Byte size of the input arugment.
				 * @param[out] data_out    Pointer to where to write output.
				 *
				 * @return RC status code of the LPF call.
				 *
				 * \warning Issues with default-constructibility of the input type \a T
				 *          (in the case of AUTOMATIC mode and no-broadcasting), while
				 *          caught in the SPMD program itself as a safety measure, should
				 *          be caught before a call to this function in order to comply with
				 *          the specification.
				 *
				 * \note This function is factored out for use with the BSP
				 *       #grb::Benchmarker.
				 */
				template<
					typename T, typename U,
					typename DispatcherType
				>
				RC run_lpf(
					const lpf_func_t alp_program,
					const void * const data_in,
					const size_t in_size,
					U * const data_out
				) const {
					// construct LPF I/O args
					lpf_args_t args = {
						data_in, in_size,
						data_out, sizeof( U ),
						&alp_program, 1
					};

					// get LPF function pointer
					lpf_spmd_t fun = reinterpret_cast< lpf_spmd_t >(
						internal::alp_exec_dispatch< T, U, DispatcherType > );

					// execute
					const lpf_err_t spmdrc = init == LPF_INIT_NONE
						? lpf_exec( LPF_ROOT, LPF_MAX_P, fun, args )
						: lpf_hook( init, fun, args );

					// check error code
					if( spmdrc != LPF_SUCCESS ) {
						return PANIC;
					}

					// done
					return SUCCESS;
				}


			private:

				/**
				 * Pack data received from user into an internal::ExecDispatcher data
				 * structure and run the ALP program.
				 *
				 * @tparam T            Input type.
				 * @tparam U            Output type.
				 * @tparam untyped_call Whether the ALP function is typed.
				 *
				 * \note If \a untyped_call is <tt>true</tt>, then \a T must be
				 *       <tt>void</tt>.
				 *
				 * @param[in] alp_program The ALP program to execute.
				 * @param[in] data_in     Pointer to input data.
				 * @param[in] in_size     Size of the input data
				 *
				 * \warning \a in_size must equal <tt>sizeof( T )</tt> if \a untyped_call
				 *          equals <tt>false</tt>.
				 *
				 * @param[out] data_out  Pointer to where to write output data.
				 * @param[in]  broadcast Whether to broadcast input from node 0 to all
				 *                       others.
				 *
				 * \warning Issues with default-constructibility of the input type \a T
				 *          (in the case of AUTOMATIC mode and no-broadcasting), while
				 *          caught in the SPMD program itself as a safety measure, should
				 *          be caught before a call to this function in order to comply with
				 *          the specification.
				 *
				 * @returns #grb::SUCCESS When the ALP program was launched successfully.
				 * @returns #grb::PANIC   On error in the communication layer while
				 *                        launching the program, during program execution,
				 *                        or while terminating the program.
				 */
				template< typename T, typename U, bool untyped_call >
				RC pack_data_and_run(
					const lpf_func_t alp_program,
					const T * const data_in,
					const size_t in_size,
					U * const data_out,
					const bool broadcast
				) const {
					static_assert( std::is_void< T >::value || !untyped_call,
						"If T is not void, this must refer to a typed ALP program call" );
					if( !untyped_call ) {
						assert( grb::utils::SizeOf< T >::value == in_size );
					}
					if( broadcast ) {
						typedef internal::ExecDispatcher< T, U, mode, true, untyped_call > Disp;
						return run_lpf< T, U, Disp >( alp_program, data_in, in_size, data_out );
					} else {
						typedef internal::ExecDispatcher< T, U, mode, false, untyped_call > Disp;
						return run_lpf< T, U, Disp >( alp_program, data_in, in_size, data_out );
					}
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
				 * @returns #grb::SUCCESS When the ALP program was launched successfully.
				 * @returns #grb::ILLEGAL When the ALP program was launched in AUTOMATIC
				 *                        mode, without broadcasting, while \a T was not
				 *                        default-constructible.
				 * @returns #grb::PANIC   On error in the communication layer while
				 *                        launching the program, during program execution,
				 *                        or while terminating the program.
				 */
				template< typename T, typename U >
				RC exec(
					const AlpTypedFunc< T, U > alp_program,
					const T &data_in,
					U &data_out,
					const bool broadcast = false
				) {
					if(
						mode == AUTOMATIC && broadcast == false &&
						!std::is_default_constructible< T >::value
					) {
						return grb::ILLEGAL;
					} else {
						return pack_data_and_run< T, U, false >(
							reinterpret_cast< lpf_func_t >( alp_program ),
							&data_in, sizeof( T ),
							&data_out, broadcast
						);
					}
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
				 * @param[in]  alp_program ALP function to run in parallel.
				 * @param[in]  data_in     Pointer to input data.
				 * @param[in]  in_size     Size of input data.
				 * @param[out] data_out    Output data.
				 * @param[in]  broadcast   Whether to broadcast input from node 0 to the
				 *                         others.
				 *
				 * @returns #grb::SUCCESS When the ALP program was launched successfully.
				 * @returns #grb::PANIC   On error in the communication layer while
				 *                        launching the program, during program execution,
				 *                        or while terminating the program.
				 */
				template< typename U >
				RC exec(
					const AlpUntypedFunc< U > alp_program,
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

	} // end namespace internal

	/**
	 * Specialization of Launcher to be used when MPI has already been
	 * initialised but not LPF.
	 */
	template<>
	class Launcher< FROM_MPI, BSP1D > :
		public internal::BaseLpfLauncher< FROM_MPI >
	{

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
	 * Specialisation of Launcher for the automatic mode.
	 *
	 * Assumes LPF takes care of any initialisation requirements.
	 */
	template<>
	class Launcher< AUTOMATIC, BSP1D > :
		public internal::BaseLpfLauncher< AUTOMATIC >
	{

		public:

			Launcher() = default;

			~Launcher() {
				assert( init == LPF_INIT_NONE );
			}

			static RC finalize() {
				return grb::SUCCESS;
			}

	};

	/**
	 * Specialisation of Launcher for the manual mode.
	 *
	 * The callee here manually connects existing processes into a joint LPF
	 * context, that is then used to execute (parallel) ALP programs.
	 *
	 * Assumes the pre-existing processes may be connected via TCP/IP.
	 */
	template< enum EXEC_MODE mode >
	class Launcher< mode, BSP1D > : public internal::BaseLpfLauncher< mode > {

		static_assert( mode == MANUAL, "Expected manual launcher mode" );

		public:

			/**
			 * Constructs a manual mode launcher.
			 *
			 * This implementation specifies the following constraints on the specified
			 * input arguments.
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
			 * The time-out of this constructor is two minutes.
			 *
			 * If giving a \a hostname as a string, it must resolve to an IP; if
			 * resolution fails, this constructor call will fail.
			 *
			 * If giving a \a port as a string, it must resolve to a port number; if
			 * resolution fails, this constructor call will fail.
			 *
			 * In addition to the standard-defined exceptions, the following errors may
			 * additionally be thrown:
			 *
			 * @throws invalid_argument When hostname or port are empty but \a nprocs is
			 *                          larger than one.
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
				if( nprocs > 1 && (hostname.empty() || port.empty()) ) {
					throw std::invalid_argument( "Host or port names may not be empty if the "
						"launcher group contains more than one process." );
				}

				// initialise MPI if not already done
				// TODO FIXME the MPI_Init should not be here. See GitHub issue #240.
				if( !is_mpi_inited && !internal::grb_mpi_initialized ) {
					if( MPI_Init( NULL, NULL ) != MPI_SUCCESS ) {
						throw std::runtime_error( "Call to MPI_Init failed." );
					} else {
						internal::grb_mpi_initialized = true;
					}
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
				// TODO FIXME the MPI_Finalize should not be here. See GitHub issue #240.
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

