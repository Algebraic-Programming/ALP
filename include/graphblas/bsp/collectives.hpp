
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#ifndef _H_GRB_BSP_COLL
#define _H_GRB_BSP_COLL

#include <type_traits>

#include <assert.h>
#include <string.h>

#include <graphblas/ops.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/base/collectives.hpp>

#include <graphblas/bsp1d/init.hpp>

#define NO_CAST_ASSERT_BLAS0( x, y, z )                                            \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a value of the same type as the first domain " \
		"of the given operator.\n"                                                 \
		"* Possible fix 3 | Ensure the operator given to this call to " y " has "  \
		"all "                                                                     \
		"of "                                                                      \
		"its "                                                                     \
		"domain"                                                                   \
		"s "                                                                       \
		"equal "                                                                   \
		"to "                                                                      \
		"each "                                                                    \
		"other."                                                                   \
		"\n"                                                                       \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );


namespace grb {

	namespace {

		/**
		 * This is a reducer function of the signature specified by lpf_reducer_t.
		 */
		template< typename OP >
		void generic_reducer( size_t n, const void * array_p, void * value_p ) {
			typedef typename OP::D1 lhs_type;
			typedef typename OP::D2 rhs_type;
			typedef typename OP::D3 out_type;
			static_assert(
					std::is_same< rhs_type, out_type >::value,
					"A generic operator-only reducer from an array into a scalar requires the "
					"RHS and output types to be the same"
				);
			assert( array_p != value_p );

			const lhs_type * const __restrict__ array =
				static_cast< const lhs_type * >( array_p );
			out_type * const __restrict__ value =
				static_cast< out_type * >( value_p );

			lhs_type left_buffer[ OP::blocksize ];

			// SIMD loop
			size_t i = 0;
			while( i + OP::blocksize < n ) {
				// load
				for( size_t k = 0; k < OP::blocksize; ++k ) {
					left_buffer[ k ] = array[ i + k ];
				}
				// compute
				for( size_t k = 0; k < OP::blocksize; ++k ) {
					OP::foldr( left_buffer[ k ], *value );
				}
				// increment
				i += OP::blocksize;
			}

			// scalar coda
			for( ; i < n; ++i ) {
				OP::foldr( array[ i ], *value );
			}

			// done
		}

	} // end anon namespace

	/**
	 * Collective communications using ALP operators for reduce-style operations.
	 *
	 * This is the BSP1D implementation.
	 *
	 * TODO internal issue #198
	 */
	template<>
	class collectives< BSP1D > {

		private:

			/** Disallow instantiation of this class. */
			collectives() {}


		public:

			/**
			 * Schedules an allreduce operation of a single object of type IOType per
			 * process. The allreduce shall be complete by the end of the call. This is a
			 * collective graphBLAS operation.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Operator \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
			 * \endparblock
			 *
			 * This function may place an alloc of \f$ P\mathit{sizeof}(IOType) \f$ bytes
			 * if the internal buffer was not sufficiently large.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Operator, typename IOType
			>
			static RC allreduce( IOType &inout, const Operator &op = Operator() ) {
				// We use the operator type to create a matching lpf_reducer_t. As a result,
				// the operator instance is never explicitly used.
				(void) op;
#ifdef _DEBUG
				std::cout << "Entered grb::collectives< BSP1D >::allreduce with inout = "
					<< inout << " and op = " << &op << std::endl;
#endif

				// static sanity check
				NO_CAST_ASSERT_BLAS0( ( !( descr & descriptors::no_casting ) ||
						std::is_same< IOType, typename Operator::D1 >::value ||
						std::is_same< IOType, typename Operator::D2 >::value ||
						std::is_same< IOType, typename Operator::D3 >::value
					),
					"grb::collectives::allreduce",
					"Incompatible given value type and operator domains while "
					"no_casting descriptor was set"
				);

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// catch trivial case early
				if( data.P == 1 ) {
					return SUCCESS;
				}

				// ensure the global buffer has enough capacity
				RC rc = data.ensureBufferSize( sizeof( IOType ) );
				if( rc != SUCCESS ) { return rc; }

				// ensure we can execute the requested collective call
				rc = data.ensureCollectivesCapacity( 1, sizeof( IOType ), 0 );
				rc = rc != SUCCESS ? rc : data.ensureMaxMessages( 2 * data.P - 2 );
				if( rc != SUCCESS ) { return rc; }

				// retrieve buffer area
				IOType * const __restrict__ buffer = data.template getBuffer< IOType >();

				// copy payload into buffer
				// rationale: this saves one global registration, which otherwise is likely
				//            to dominate most uses for this collective call
				*buffer = inout;

				// get the lpf_reducer_t
				lpf_reducer_t reducer = &(generic_reducer< Operator >);

				// schedule allreduce
				lpf_err_t lpf_rc = lpf_allreduce(
						data.coll,
						buffer, data.slot,
						sizeof(IOType),
						reducer
					);

				// execute allreduce
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// catch LPF errors
				if( lpf_rc != LPF_SUCCESS ) {
					/* LCOV_EXCL_START */
					if( lpf_rc != LPF_ERR_FATAL ) {
						std::cerr << "Error: unspecified LPF error was returned, please submit a "
							<< "bug report!\n";
#ifndef NDEBUG
						const bool lpf_spec_says_this_should_not_occur = false;
						assert( lpf_spec_says_this_should_not_occur );
#endif
						return PANIC;
					}
					std::cerr << "Error (allreduce, distributed): LPF reports unmitigable "
						<< "error\n";
					return PANIC;
					/* LCOV_EXCL_STOP */
				}

				// copy back
				inout = *buffer;

				// done
				return SUCCESS;
			}

			/**
			 * Schedules a reduce operation of a single object of type IOType per process.
			 * The reduce shall be complete by the end of the call. This is a collective
			 * graphBLAS operation. The BSP costs are as for the PlatformBSP #reduce.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Operator \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
			 * \endparblock
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Operator, typename IOType
			>
			static RC reduce(
				IOType &inout, const lpf_pid_t root = 0,
				const Operator op = Operator()
			) {
				// We use the operator type to create a matching lpf_reducer_t. As a result,
				// the operator instance is never explicitly used.
				(void) op;

				// static sanity check
				NO_CAST_ASSERT_BLAS0( ( !(descr & descriptors::no_casting) ||
						std::is_same< IOType, typename Operator::D1 >::value ||
						std::is_same< IOType, typename Operator::D2 >::value ||
						std::is_same< IOType, typename Operator::D3 >::value
					), "grb::collectives::reduce",
					"Incompatible given value type and operator domains while "
					"no_casting descriptor was set"
				);

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// catch trivial case early
				if( data.P == 1 ) {
					return SUCCESS;
				}

				// ensure the global buffer has enough capacity
				RC rc = data.ensureBufferSize( sizeof( IOType ) );
				if( rc != SUCCESS ) { return rc; }

				// ensure we can execute the requested collective call
				rc = data.ensureCollectivesCapacity( 1, sizeof( IOType ), 0 );
				rc = rc != SUCCESS ? rc : data.ensureMaxMessages( data.P - 1 );
				if( rc != SUCCESS ) { return rc; }

				// retrieve buffer area
				IOType * const __restrict__ buffer = data.template getBuffer< IOType >();

				// copy payload into buffer
				// rationale: this saves one global registration, which otherwise is likely
				//            to dominate most uses for this collective call
				*buffer = inout;

				// get the lpf_reducer_t
				lpf_reducer_t reducer = &(generic_reducer< Operator >);

				// schedule allreduce
				lpf_err_t lpf_rc = lpf_reduce(
						data.coll,
						buffer, data.slot,
						sizeof(IOType),
						reducer,
						root
					);

				// execute allreduce
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// catch LPF errors
				if( lpf_rc != LPF_SUCCESS ) {
					/* LCOV_EXCL_START */
					if( lpf_rc != LPF_ERR_FATAL ) {
						std::cerr << "Error (reduce, distributed): unspecified LPF error was "
							<< "returned, please submit a bug report!\n";
#ifndef NDEBUG
						const bool lpf_spec_says_this_should_not_occur = false;
						assert( lpf_spec_says_this_should_not_occur );
#endif
						return PANIC;
					}
					std::cerr << "Error (reduce, distributed): LPF reports unmitigable error"
						<< "\n";
					return PANIC;
					/* LCOV_EXCL_STOP */
				}

				// copy back
				if( data.s == static_cast< size_t >( root ) ) {
					inout = *buffer;
				}

				// done
				return SUCCESS;
			}

			/**
			 * Schedules a broadcast operation of a single object of type IOType per process.
			 * The broadcast shall be complete by the end of the call. This is a collective
			 * graphBLAS operation. The BSP costs are as for the PlatformBSP #broadcast.
			 *
			 * @tparam IOType The type of the to-be broadcast value.
			 *
			 * @param[in,out] inout On input: the value at the root process to be broadcast.
			 *                      On output at process \a root: the same value.
			 *                      On output at non-root processes: the value at root.
			 *
			 * \parblock
			 * \par Performance semantics: common
			 * Whether system calls will happen depends on the LPF engine compiled with,
			 * as does whether buffer space is proportional to the payload size is
			 * required. In principle, when using a fabric like Inifiband and when using
			 * the LPF ibverbs engine, the intended IB zero-copy behaviour is attained.
			 *
			 * All below variants in any backend shall not result in dynamic memory
			 * allocations.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: serial
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ NP \f$ ;
			 * -# BSP cost: \f$ NPg + l \f$;
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: two phase
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ 2N \f$ ;
			 * -# BSP cost: \f$ 2(Ng + l) \f$;
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: two level tree
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ 2\sqrt{P}N \f$ ;
			 * -# BSP cost: \f$ 2(\sqrt{P}Ng + l) \f$;
			 * \endparblock
			 */
			template< typename IOType >
			static RC broadcast( IOType &inout, const lpf_pid_t root = 0 ) {
				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// dynamic checks
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// catch trivial request first
				if( data.P == 1 ) {
					return SUCCESS;
				}

				// ensure the global buffer has enough capacity
				RC rc = data.ensureBufferSize( sizeof( IOType ) );
				if( rc != SUCCESS ) { return rc; }

				// ensure we have enough memory slots for local registration
				rc = data.ensureMemslotAvailable();
				if( rc != SUCCESS ) { return rc; }

				// ensure we can execute the requested collective call
				rc = data.ensureCollectivesCapacity( 1, 0, sizeof( IOType ) );

				// ensure we have the required h-relation capacity
				// note the below cannot overflow since we guarantee data.P > 1
				rc = rc != SUCCESS
					? rc
					: data.ensureMaxMessages( std::max( data.P + 1, 2 * data.P - 3 ) );
				if( rc != SUCCESS ) { return rc; }

				// root retrieve buffer area and copies payload into buffer
				// rationale: this saves one global registration, which otherwise is likely
				//            to dominate most uses for this collective call
				if( data.s == static_cast< size_t >( root ) ) {
					IOType * const __restrict__ buffer = data.template getBuffer< IOType >();
					*buffer = inout;
				}

				// register destination area
				lpf_memslot_t dest_slot = LPF_INVALID_MEMSLOT;
				lpf_err_t lpf_rc = lpf_register_local(
					data.context, &inout, sizeof( IOType ), &dest_slot );
				if( lpf_rc == SUCCESS ) {
					lpf_rc = lpf_broadcast( data.coll, data.slot, dest_slot, sizeof( IOType ),
						root );
				}

				// finish communication
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// cleanup
				if( dest_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
					(void) lpf_deregister( data.context, dest_slot );
				}

				// LPF error handling
				if( lpf_rc != LPF_SUCCESS ) {
					/* LCOV_EXCL_START */
					if( lpf_rc != LPF_ERR_FATAL ) {
#ifndef NDEBUG
						const bool lpf_spec_says_this_should_never_happen = false;
						assert( lpf_spec_says_this_should_never_happen );
#endif
						std::cerr << "Error, broadcast (BSP): unexpected error. Please submit a "
							<< "bug report\n";
					}
					std::cerr << "Error, broadcast (BSP): LPF encountered a fatal error\n";
					return PANIC;
					/* LCOV_EXCL_STOP */
				}

				// done
				return SUCCESS;
			}

			/**
			 * Schedules a broadcast of a raw array of a given type.
			 *
			 * @tparam IOType The array element type.
			 *
			 * @param[in,out] inout A pointer to the array to broadcast (for the root
			 *                      user process), or a pointer where to store the array
			 *                      to be broadcast (for all other user processes).
			 * @param[in] size      The size, in number of array elements, of the array
			 *                      to be broadcast. Must match across all user processes
			 *                      in the collective call.
			 * @param[in] root      Which user process ID is the root.
			 *
			 * \parblock
			 * \par Performance semantics
			 *
			 * Please refer to the LPF collectives higher-level library for the
			 * performance semantics of this call. (This function does not implements
			 * its own custom logic for this primitive.)
			 *
			 * This cost should be appended with the cost of registering \a inout as a
			 * memory space for global RDMA communication.
			 * \endparblock
			 *
			 * @returns grb::SUCCESS On successful broadcast of the requested array.
			 * @returns grb::PANIC   If the communication layer has failed.
			 */
			template< Descriptor descr = descriptors::no_operation, typename IOType >
			static RC broadcast(
				IOType * inout, const size_t size, const size_t root = 0
			) {
				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// check contract
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// catch trivial cases
				if( data.P == 1 || size == 0 ) {
					return SUCCESS;
				}

				// an array of arbitrary size is probably best not copied, we hence incur
				// the extra latency of registering inout

				// ensure we have enough memory slots for global registration
				RC rc = data.ensureMemslotAvailable();

				// ensure we can execute the requested collective call
				rc = rc != SUCCESS
					? rc
					: data.ensureCollectivesCapacity( 1, 0, size * sizeof( IOType ) );

				// ensure we have the required h-relation capacity
				// note the below cannot overflow since we guarantee data.P > 1
				rc = rc != SUCCESS
					? rc
					: data.ensureMaxMessages( std::max( data.P + 1, 2 * data.P - 3 ) );

				// propagate any errors
				if( rc != SUCCESS ) { return rc; }

				// get byte size
				const size_t bsize = size * sizeof( IOType );

				// register array
				lpf_memslot_t user_slot = LPF_INVALID_MEMSLOT;
				lpf_err_t lpf_rc = lpf_register_global(
					data.context, &inout, bsize, &user_slot );

				// activate registration
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// schedule broadcast
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_broadcast( data.coll, user_slot, user_slot, bsize, root );
				}

				// execute broadcast
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// cleanup
				if( user_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
					(void) lpf_deregister( data.context, user_slot );
				}

				// LPF error handling
				if( lpf_rc != LPF_SUCCESS ) {
					/* LCOV_EXCL_START */
					if( lpf_rc != LPF_ERR_FATAL ) {
#ifndef NDEBUG
						const bool lpf_spec_says_this_should_never_happen = false;
						assert( lpf_spec_says_this_should_never_happen );
#endif
						std::cerr << "Error, array broadcast (BSP): unexpected error. Please "
							<< "submit a bug report\n";
					}
					std::cerr << "Error, array broadcast (BSP): LPF encountered a fatal error"
						<< "\n";
					return PANIC;
					/* LCOV_EXCL_STOP */
				}

				// done
				return SUCCESS;
			}

	};

} // namespace grb

#undef NO_CAST_ASSERT_BLAS0

#endif // end ``_H_GRB_BSP_COLL''

