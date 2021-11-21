
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

#include <graphblas/backends.hpp>
#include <graphblas/base/collectives.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/bsp1d/init.hpp>
#include <graphblas/ops.hpp>

#include "collectives_blas1_raw.hpp"
#include "internal-collectives.hpp"

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

	/**
	 * Collective communications using the GraphBLAS operators for
	 * reduce-style operations. This is the BSP1D implementation.
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
		template< Descriptor descr = descriptors::no_operation, typename Operator, typename IOType >
		static RC allreduce( IOType & inout, const Operator & op = Operator() ) {
			// this is the serial algorithm only
			// TODO internal issue #19
#ifdef _DEBUG
			std::cout << "Entered grb::collectives< BSP1D >::allreduce with "
						 "inout = "
					  << inout << " and op = " << &op << std::endl;
#endif

			// static sanity check
			NO_CAST_ASSERT_BLAS0( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, typename Operator::D1 >::value || std::is_same< IOType, typename Operator::D2 >::value ||
									  std::is_same< IOType, typename Operator::D3 >::value ),
				"grb::collectives::allreduce",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set" );

			// we need access to LPF context
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();

			// catch trivial case early
			if( data.P == 1 ) {
				return SUCCESS;
			}

			// we need to register inout
			lpf_memslot_t inout_slot = LPF_INVALID_MEMSLOT;
			if( data.ensureMemslotAvailable() != grb::SUCCESS ) {
				assert( false );
				return PANIC;
			}
			if( lpf_register_local( data.context, &inout, sizeof( IOType ), &inout_slot ) != LPF_SUCCESS ) {
				assert( false );
				return PANIC;
			} else {
				data.signalMemslotTaken();
			}

			// allgather inout values
			// note: buffer size check is done by the below function
			if( internal::allgather( inout_slot, 0, data.slot, data.s * sizeof( IOType ), sizeof( IOType ), data.P * sizeof( IOType ), true ) != grb::SUCCESS ) {
				assert( false );
				return PANIC;
			}

			// deregister
			if( lpf_deregister( data.context, inout_slot ) != LPF_SUCCESS ) {
				assert( false );
				return PANIC;
			} else {
				data.signalMemslotReleased();
			}

			// fold everything
			IOType * __restrict__ const buffer = data.getBuffer< IOType >();
			for( size_t i = 0; i < data.P; ++i ) {
				if( i == data.s ) {
					continue;
				}
#ifdef _DEBUG
				std::cout << data.s
						  << ": in Collectives< BSP1D >::allreduce. Buffer "
							 "index "
						  << i << ", folding " << buffer[ i ] << " into " << inout << ", yields ";
#endif
				// if casting is required to apply op, foldl will take care of this
				if( foldl< descr >( inout, buffer[ i ], op ) != SUCCESS ) {
					assert( false );
				}
#ifdef _DEBUG
				std::cout << inout << std::endl;
#endif
			}

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
		 *
		 */
		template< Descriptor descr = descriptors::no_operation, typename Operator, typename IOType >
		static RC reduce( IOType & inout, const lpf_pid_t root = 0, const Operator op = Operator() ) {
			// this is the serial algorithm only
			// TODO internal issue #19

			// static sanity check
			NO_CAST_ASSERT_BLAS0( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, typename Operator::D1 >::value || std::is_same< IOType, typename Operator::D2 >::value ||
									  std::is_same< IOType, typename Operator::D3 >::value ),
				"grb::collectives::reduce",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set" );

			// we need access to LPF context
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();

			// catch trivial case early
			if( data.P == 1 ) {
				return SUCCESS;
			}

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, data.P * sizeof( IOType ), 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// create a local register slot
			lpf_memslot_t inout_slot = LPF_INVALID_MEMSLOT;
			if( lpf_register_global( data.context, &inout, sizeof( IOType ), &inout_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// gather together values
			if( lpf_gather( coll, inout_slot, data.slot, sizeof( IOType ), root ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// finish the communication
			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// do deregister
			if( lpf_deregister( data.context, inout_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// fold everything: root only
			if( data.s == root ) {
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();
				for( size_t i = 0; i < data.P; ++i ) {
					if( i == root ) {
						continue;
					}
					// if casting is required to apply op, foldl will take care of this
					// note: the no_casting check could be deferred to foldl but this would result in unclear error messages
					if( foldl< descr >( inout, buffer[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
				}
			}

			if( commsPostamble( data, &coll, data.P, data.P * sizeof( IOType ), 0, 1 ) != SUCCESS ) {
				return PANIC;
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
		 * \par Performance semantics: serial
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ NP \f$ ;
		 * -# BSP cost: \f$ NPg + l \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two hase
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2N \f$ ;
		 * -# BSP cost: \f$ 2(Ng + l) \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2\sqrt{P}N \f$ ;
		 * -# BSP cost: \f$ 2(\sqrt{P}Ng + l) \f$;
		 * \endparblock
		 *
		 */
		template< typename IOType >
		static RC broadcast( IOType & inout, const lpf_pid_t root = 0 ) {
			// we need access to LPF context
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: IOType -> IOType
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// register inout
			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			if( data.ensureMemslotAvailable() != SUCCESS ) {
				return PANIC;
			}
			if( lpf_register_global( data.context, &inout, sizeof( IOType ), &slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// broadcast value
			if( lpf_broadcast( coll, slot, slot, sizeof( IOType ), root ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// finish communication
			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// coda
			if( lpf_deregister( data.context, slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			if( commsPostamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/** TODO documentation */
		template< Descriptor descr = descriptors::no_operation, typename IOType >
		static RC broadcast( IOType * inout, const size_t size, const size_t root = 0 ) {
			return internal::broadcast< descr >( inout, size, root );
		}
	};

} // namespace grb

#undef NO_CAST_ASSERT_BLAS0

#endif // end ``_H_GRB_BSP_COLL''
