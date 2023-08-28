
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
#include "exec_broadcast_routines.hpp"


/** Forward declaration. */

namespace grb {

	/**
	 * Type definition for GRB function with input type information. */
	template< typename InputType, typename OutputType > using grb_typed_func_t =
		void ( * )( const InputType &, OutputType & );

	/* Type definition for GRB function without input type information. */
	template< typename InputType, typename OutputType > using grb_untyped_func_t =
		void ( * )( const InputType *, size_t, OutputType & );

	namespace internal {

		/**
		 * Base data structure storing necessary data to run a GRB function through LPF.
		 * 
		 * @tparam InputType type of function input
		 */
		template< typename InputType > struct PackedExecInput {
			const InputType *input; ///< pointer to input
			size_t input_size; ///< size fo input (== sizeof( InputType ) for GRB typed calls)
			bool broadcast_input; ///< whether the user requested broadcast fo input from node 0

			/** Default constructor, required by _grb_exec_dispatch() in case of object serialization. */
			PackedExecInput() = default;

			/** Constructor with member initialization, for actual construction within node 0.  */
			PackedExecInput( const InputType *in, size_t s, bool bc ) :
				input( in ),
				input_size( s ),
				broadcast_input( bc ) {}
		};

		/**
		 * Adaptor type to run a typed GRB function: it stores relevant parameters for data braodcast
		 * (inherited from PackedExecInput) and adapts the function call to the underlying type.
		 */
		template< typename InputType, typename OutputType, bool variable_input >
		struct ExecDispatcher : PackedExecInput< InputType > {

			using PackedExecInput< InputType >::PackedExecInput;

			constexpr static bool is_input_size_variable= false;

			/** Static adapter for typed GRB calls, to be used also outside of this struc. */
			static inline void lpf_grb_call( const lpf_func_t fun, size_t, const InputType *in,
				OutputType *out, lpf_pid_t, lpf_pid_t ) {
				reinterpret_cast< grb_typed_func_t< InputType, OutputType > >( fun )( *in, *out );
			}

			/** Functor operator to call a typed GRB function. */
			inline void operator()( const lpf_func_t fun, size_t in_size, const InputType *in,
				OutputType *out, lpf_pid_t s, lpf_pid_t P ) const {
				lpf_grb_call( fun, in_size, in, out, s, P );
			}
		};

		/**
		 * Adaptor type to run an untyped GRB function: it stores relevant parameters for data braodcast
		 * (inherited from PackedExecInput) and adapts the function call to the underlying type.
		 */
		template< typename InputType, typename OutputType >
		struct ExecDispatcher< InputType, OutputType, true > : PackedExecInput< InputType > {

			using PackedExecInput< InputType >::PackedExecInput;

			constexpr static bool is_input_size_variable = true;

			static inline void lpf_grb_call( const lpf_func_t fun, size_t in_size,
				const InputType *in, OutputType *out, lpf_pid_t, lpf_pid_t ) {
				reinterpret_cast< grb_untyped_func_t< InputType, OutputType > >( fun )( in, in_size, *out );
			}

			inline void operator()( const lpf_func_t fun, size_t in_size,
				const InputType *in, OutputType *out, lpf_pid_t s, lpf_pid_t P ) const {
				lpf_grb_call( fun, in_size, in, out, s, P );
			}
		};

		/**
		 * Allocator for data structures: if \p typed_allocation is \a true, then allocate \p T
		 * on the heap via its default contructor \p T(), otherwise as a byte array (without construction).
		 */
		template< typename T, bool typed_allocation > struct exec_allocator {

			static_assert( std::is_default_constructible< T >::value, "T must be default constructible" );

			using AllocatedType = T;
			using Deleter = std::function< void( AllocatedType * ) >;
			using PointerHolder = std::unique_ptr< AllocatedType, Deleter >;

			static PointerHolder make_pointer( size_t ) {
				return std::unique_ptr< AllocatedType, Deleter >( new T(), // allocate with default construction
					[] ( AllocatedType * ptr ) { delete ptr; } );
			}
		};

		/**
		 * Template specialization for untyped allocation: data is allocated as a byte array
		 * and not initialized.
		 */
		template< typename T > struct exec_allocator< T, false > {

			using AllocatedType = char;
			using Deleter = std::function< void( AllocatedType * ) >;
			using PointerHolder = std::unique_ptr< AllocatedType, Deleter >;

			static PointerHolder make_pointer( size_t size ) {
				return std::unique_ptr< AllocatedType, Deleter >( new AllocatedType[ size ],
					[] ( AllocatedType * ptr ) { delete [] ptr; } );
			}
		};

/** Macro to check an LPF return code and throw in case it indicated an error. */
#define __LPF_REPORT_COMM_ERROR( brc ) assert( brc == LPF_SUCCESS );						\
	if( ( brc ) != LPF_SUCCESS ) { 															\
		std::cerr << __FILE__ << "," << __LINE__ << ": LPF collective failed" << std::endl; \
	}

		/**
		 * Dispatcher to be called via LPF for distributed execution of a GRB function. It handles
		 * type information of the called function via the \p DispatcherType structure.
		 * 
		 * @tparam T GRB function input type
		 * @tparam U GRB function outut type
		 * @tparam mode grb::EXEC_MODE of the LPF call
		 * @tparam DispatcherType data structure with information about the GRB function to run
		 * @param ctx LPf context to run in
		 * @param s node identifier (in the range [0, P))
		 * @param P number of parallel processes
		 * @param args record with input and output information for LPF calls
		 */
		template< typename T, typename U, enum EXEC_MODE mode, typename DispatcherType >
		void _grb_exec_dispatch( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
			static_assert( std::is_base_of< ExecDispatcher< T, U, true >, DispatcherType >::value ||
				std::is_base_of< ExecDispatcher< T, U, false >, DispatcherType >::value,
				"DispatcherType must derive from ExecDispatcher"
			);
			static_assert( std::is_default_constructible< DispatcherType >::value,
				"DispatcherType must be default-constructible" );
			assert( P > 0 );
			assert( s < P );
#ifdef _DEBUG
			if( s == 0 ) {
				std::cout << "Info: launcher spawned or hooked " << P << " ALP/GraphBLAS "
					<< "user processes.\n";
			}
#endif
			// call information for the GRB function
			const DispatcherType *dispatcher = static_cast< const DispatcherType* >( args.input );
			std::unique_ptr< DispatcherType > dispatcher_holder;

			lpf_coll_t coll;
			if( P > 1 ) {
				if( mode == AUTOMATIC ) {
					// AUTOMATIC mode: we must initialize communication and fetch the dispatcher
					lpf_err_t brc =lpf_init_collectives_for_bradocast( ctx, coll, s, P, 1 );
					__LPF_REPORT_COMM_ERROR( brc );

					// allocate memory for the dispatcher
					if( s > 0 ) {
						dispatcher_holder.reset( new DispatcherType );
						dispatcher = dispatcher_holder.get();
					}
					// fetch the dispatcher
					brc = lpf_register_and_broadcast( ctx, coll,
						const_cast< void * >( reinterpret_cast< const void * >( dispatcher ) ), sizeof( DispatcherType ) );
					__LPF_REPORT_COMM_ERROR( brc );
				} else if( dispatcher->broadcast_input ) {
					// the dispatcher is already valid and the user requested broadcasting: init communication
					lpf_err_t brc = lpf_init_collectives_for_bradocast( ctx, coll, s, P, 1 );
					__LPF_REPORT_COMM_ERROR( brc );
				}
			}

			// dispatcher is now valid: assign initial value for size
			size_t in_size = dispatcher->input_size;
			if( mode != AUTOMATIC && dispatcher->broadcast_input && P > 1
				&& DispatcherType::is_input_size_variable ) {
				// user requested broadcast and the input size is user-given: fetch size from master
				in_size = s == 0 ? dispatcher->input_size : 0UL;
				lpf_err_t brc = lpf_register_and_broadcast( ctx, coll,
					reinterpret_cast< void * >( &in_size ), sizeof( size_t ) );
				__LPF_REPORT_COMM_ERROR( brc );
				assert( in_size != 0 );
			}

			constexpr bool typed_alloc = not DispatcherType::is_input_size_variable
				&& std::is_default_constructible< T >::value;
			using input_allocator_t = exec_allocator< T, typed_alloc >;
			using input_allocated_t = typename input_allocator_t::AllocatedType;
			typename input_allocator_t::PointerHolder data_in_holder;
			// input data: by default user-given input
			const input_allocated_t * data_in = reinterpret_cast< const input_allocated_t * >( dispatcher->input );
			if( ( mode == AUTOMATIC || ( dispatcher->broadcast_input && DispatcherType::is_input_size_variable ) )
				&& s > 0 ) {
				// if no memory exists (mode == AUTOMATIC) or the size was not known and the user requested broadcast
				// then allocate input data
				data_in_holder = input_allocator_t::make_pointer( in_size );
				data_in = data_in_holder.get();
			}
			if( ( mode == AUTOMATIC || dispatcher->broadcast_input ) && P > 1 ) {
				// retrieve data from master
				lpf_err_t brc = lpf_register_and_broadcast( ctx, coll,
					const_cast< void * >( reinterpret_cast< const void * >( data_in ) ), in_size );
				__LPF_REPORT_COMM_ERROR( brc );
			}
			using output_allocator_t = exec_allocator< U, std::is_default_constructible< U >::value >;
			typename output_allocator_t::PointerHolder data_out_holder;
			U * data_out = reinterpret_cast< U * >( args.output );
			if( mode == AUTOMATIC && s > 0 ) {
				data_out_holder = output_allocator_t::make_pointer( sizeof( U ) );
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
			( *dispatcher )( args.f_symbols[ 0 ], in_size, data_in, data_out, s, P  );

			// finalise ALP/GraphBLAS
			grb_rc = grb::finalize< BSP1D >();
			if( grb_rc != grb::SUCCESS ) {
				std::cerr << "Error: could not finalise ALP/GraphBLAS" << std::endl;
				assert( false );
			}
		}
		#undef __LPF_REPORT_COMM_ERROR

	} // namespace internal

	/**
	 * Base class for Launcher's, with common logic and information; mainly wrapping user #exec()
	 * parameters into internal data structures and calling LPF.
	 * 
	 * @tparam mode grb::EXEC_MODE LPF execution mode
	 */
	template< enum EXEC_MODE mode >
	class BaseLauncher {

	private:
		/**
		 * Pack data received from user into an internal::ExecDispatcher data structure and run the LPF call.
		 * 
		 * @tparam T input type
		 * @tparam U output type
		 * @tparam untyped_call whether the GRB function is typed
		 * @param grb_program grb function to run distributed
		 * @param data_in pointer to input data
		 * @param in_size size of input data ( == sizeof( T ) if untyped_call == false )
		 * @param data_out pointer to output data
		 * @param broadcast whether to broadcast input from node 0 to the others
		 * @return RC status code of the LPF call
		 */
		template< typename T, typename U, bool untyped_call >
		RC pack_data_and_run(
			lpf_func_t grb_program, // user GraphBLAS program
			const T *data_in,
			size_t in_size,
			U *data_out,
			bool broadcast
		) {
			using disp_t = internal::ExecDispatcher< T, U, untyped_call >;
			disp_t disp_info{ data_in, in_size, broadcast };
			return run_lpf< T, U, disp_t >( grb_program, disp_info, data_out );
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
		 * Run the given \p grb_program with the given input information \p disp_info via LPF.
		 * 
		 * @tparam T input type
		 * @tparam U output type
		 * @tparam DispatcherType tpye of the data structure that holds input and call information
		 * @param grb_program grb function to run distributed
		 * @param disp_info data structure that holds input and call information
		 * @param data_out pointer to output data
		 * @return RC status code of the LPF call
		 */
		template< typename T, typename U, typename DispatcherType >
		RC run_lpf(
			lpf_func_t grb_program, // user GraphBLAS program
			DispatcherType &disp_info,
			U *data_out
		) const {
			lpf_args_t args{ &disp_info, sizeof( DispatcherType ), data_out, sizeof( U ), &grb_program, 1 };
			lpf_spmd_t fun = reinterpret_cast< lpf_spmd_t >(
				internal::_grb_exec_dispatch< T, U, mode, DispatcherType > );

			const lpf_err_t spmdrc =  init == LPF_INIT_NONE ? lpf_exec( LPF_ROOT, LPF_MAX_P, fun, args ) :
				lpf_hook( init, fun, args );

			// check error code
			if( spmdrc != LPF_SUCCESS ) {
				return PANIC;
			}
			return SUCCESS;
		}

	public:
		/**
		 * Run a typed GRB function distributed via LPF. In case of AUTOMATIC mode, input data is allocated
		 * by default (if the type allows) or as a sequence of bytes. This assumes the default allocator does not
		 * have \b any side affect (like memory allocation). In case of broadcast request, data is trivially serialized:
		 * hence, non-trivial objects (e.g., storing pointers to memory buffers) are not valid anymore in processes
		 * othern than the master.
		 * 
		 * @tparam T input type
		 * @tparam U output type
		 * @param grb_program grb function to run distributed
		 * @param data_in input data
		 * @param data_out output data
		 * @param broadcast whether to broadcast input from node 0 to the others
		 * @return RC status code of the LPF call
		 */
		template< typename T, typename U >
		RC exec(
			grb_typed_func_t< T, U > grb_program, // user GraphBLAS program
			const T &data_in, U &data_out,           // input & output data
			const bool broadcast = false
		) {
			return pack_data_and_run< T, U, false >( reinterpret_cast< lpf_func_t >( grb_program ),
				&data_in, sizeof( T ), &data_out, broadcast );
		}

		/**
		 * Run an untyped GRB function distributed via LPF. Input data has variable size, known only at runtime.
		 * Therefore, input data cannot be costructed by default, but are serialized and replicated as a mere
		 * sequence of bytes.
		 * 
		 * @tparam T input type
		 * @tparam U output type
		 * @param grb_program grb function to run distributed
		 * @param data_in pointer to input data
		 * @param in_size size of input data
		 * @param data_out output data
		 * @param broadcast whether to broadcast input from node 0 to the others
		 * @return RC status code of the LPF call
		 */		template< typename U >
		RC exec(
			grb_untyped_func_t< void, U > grb_program,
			const void * data_in, const size_t in_size,
			U &data_out,
			const bool broadcast = false
		) {
			return pack_data_and_run< void, U, true >( reinterpret_cast< lpf_func_t >( grb_program ),
				data_in, in_size, &data_out, broadcast );
		}
	};


	/**
	 * Specialization of Launcher to be used when MPI has already been initialized but not LPF.
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

			/**
			 * Implementation note: this Launcher may need to clear a field of
			 * type \a lpf_init_t when used in MANUAL mode.
			 */
			~Launcher() {
				assert( init == LPF_INIT_NONE );
			}

			/**
			 * This implementation needs to release MPI resources in manual mode.
			 */
			static RC finalize() {
				return grb::SUCCESS;
			}

	};

	/**
	 * Specialization of Launcher to be used when whishihg to create an LPF context
	 * by manually connecting nodes together via TCP.
	 * The detection of the available nodes is done via MPI, which can be already initialized
	 * or can be initialized here.
	 */
	template< enum EXEC_MODE mode >
	class Launcher< mode, BSP1D > : public BaseLauncher< mode > {

		static_assert( mode == MANUAL, "mode is not manual" );

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
		Launcher(
			const size_t process_id = 0,              // user process ID
			const size_t nprocs = 1,                  // total number of user processes
			const std::string& hostname = "localhost", // one of the process' hostnames
			const std::string& port = "0",             // a free port at hostname
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
				&this->init
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
		 * Implementation note: this Launcher may need to clear a field of
		 * type \a lpf_init_t when used in MANUAL mode.
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

