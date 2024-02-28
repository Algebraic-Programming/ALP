
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
 * Implements BLAS1 collectives both on raw arrays as well as on GraphBLAS
 * (reference) vectors.
 *
 * \warning Never include this file directly!
 *
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#include <cmath>

#include <lpf/collectives.h>

#include <graphblas/base/vector.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/bsp1d/init.hpp>

#include "internal-collectives.hpp"

/**
 * The difference between pid and root, modulus P - circumvents weird modulus
 * behaviour under -ve numbers
 */
#define DIFF( pid, root, P ) ( (pid < root) ? pid + P - root : pid - root ) % P


namespace grb {

	/**
	 * Collective communications using the GraphBLAS operators for reduce-style
	 * operations.
	 */
	namespace internal {

		/**
		 * Schedules a gather operation of a single object of type IOType per process
		 * to a vector of \a P elements.
		 * The gather shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #gather.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be gathered value.
		 *
		 * @param[in]  in  The value at the calling process to be gathered.
		 * @param[out] out The vector of gathered values, available at the root
		 *                 process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC gather(
			const IOType &in,
#ifdef BLAS1_RAW
			IOType * out,
#else
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// run-time sanity check
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( out, data.P )
#endif

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				// if failure at this point, no harm done and exit immediately
				return PANIC;
			}

			// copy input to buffer
			const size_t pos = ( data.s == root ) ? data.s : 0;

			// prevent self-copy
#ifndef BLAS1_RAW
			if( internal::getRaw( out ) + pos != &in ) {
				internal::getRaw( out )[ pos ] = in;
			}
#else
			if( out + pos != &in ) {
				out[ pos ] = in;
			}
#endif

			// create memslot on output vector
			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
#ifndef BLAS1_RAW
			if( lpf_register_global(
					data.context, internal::getRaw( out ), data.P * sizeof( IOType ), &slot
				) != LPF_SUCCESS
			) {
#else
			if( lpf_register_global(
					data.context, out, data.P * sizeof( IOType ), &slot
				) != LPF_SUCCESS
			) {
#endif
				// failure at this point will have to be cleaned up as best as possible
				ret = PANIC;
			}

			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) ) {
				ret = PANIC;
			}

			// gather values
			if( ret == SUCCESS &&
					lpf_gather( coll, slot, slot, sizeof( IOType ), root )
				!= LPF_SUCCESS
			) {
				// failure at this point will have to be cleaned up as best as possible
				ret = PANIC;
			}

			// perform communication
			if( ret == SUCCESS &&
					lpf_sync( data.context, LPF_SYNC_DEFAULT )
				!= LPF_SUCCESS
			) {
				// failure at this point will have to be cleaned up as best as possible
				ret = PANIC;
			}

#ifndef BLAS1_RAW
			// make sure sparsity info is correct
			for(
				size_t i = 0;
				data.s == root &&
					ret == SUCCESS &&
					internal::getCoordinates( out ).size() != internal::getCoordinates( out ).nonzeroes()
					&& i < data.P;
				++i
			) {
				(void) internal::getCoordinates( out ).assign( i );
			}
#endif

			// deregister slot
			if( slot != LPF_INVALID_MEMSLOT &&
					lpf_deregister( data.context, slot )
				!= LPF_SUCCESS
			) {
				// error during cleanup of memslot
				ret = PANIC;
			}

			if( commsPostamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				// error during cleanup of postamble
				ret = PANIC;
			}

			// done
			return ret;
		}

		/**
		 * Schedules a gather operation of a vector of \a N/P elements of type IOType
		 * per process to a vector of \f$ N \f$ elements.
		 * The gather shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #gather.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be gathered value.
		 *
		 * @param[in]  in:  The vector at the calling process to be gathered.
		 * @param[out] out: The vector of gathered values, available at the root process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC gather(
#ifdef BLAS1_RAW
			const IOType * in,
			const size_t size,
			IOType * out,
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: Vector IOType -> P * Vector IOType
#ifndef BLAS1_RAW
			TEST_VEC_MULTIPLE( in, out, data.P )
			const size_t size = internal::getCoordinates( in ).size();
#endif

			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// create memslot on output vector
			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
#ifndef BLAS1_RAW
			if( lpf_register_global( data.context, internal::getRaw( out ),
					size * data.P * sizeof( IOType ), &slot )
				!= LPF_SUCCESS
			) {
#else
			if( lpf_register_global( data.context, out, size * data.P * sizeof( IOType ),
					&slot )
				!= LPF_SUCCESS
			) {
#endif
				// failure at this point will have to be cleaned up as best as possible
				ret = PANIC;
			}

			// copy input to buffer
			const size_t pos = ( data.s == root ) ? data.s : 0;
#ifdef BLAS1_RAW
			for(
				size_t i = 0;
				ret == SUCCESS && ( out + pos * size ) != in && i < size;
				i++
			) {
				out[ pos * size + i ] = in[ i ];
			}
#else
			for(
				size_t i = 0;
				ret == SUCCESS &&
					(internal::getRaw( out ) + pos * size) != internal::getRaw( in ) &&
					i < size;
				i++
			) {
				internal::getRaw( out )[ pos * size + i ] = internal::getRaw( in )[ i ];
			}
#endif
			// activate registrations
			if( ret == SUCCESS &&
					lpf_sync( data.context, LPF_SYNC_DEFAULT )
				!= LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// gather values
			if( ret == SUCCESS &&
					lpf_gather( coll, slot, slot, size * sizeof( IOType ), root )
				!= LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// complete requested communication
			if( ret == SUCCESS &&
					lpf_sync( data.context, LPF_SYNC_DEFAULT )
				!= LPF_SUCCESS
			) {
				ret = PANIC;
			}

#ifndef BLAS1_RAW
			// set sparsity of output
			for(
				size_t i = 0;
				data.s == root &&
					ret == SUCCESS &&
					internal::getCoordinates( out ).size() !=
						internal::getCoordinates( out ).nonzeroes() &&
					i < data.P * size;
				++i
			) {
				(void) internal::getCoordinates( out ).assign( i );
			}
#endif

			// destroy memory slot
			if( slot != LPF_INVALID_MEMSLOT &&
					lpf_deregister( data.context, slot ) !=
				LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				ret = PANIC;
			}

			// done
			return ret;
		}

		/**
		 * Schedules a scatter operation of a vector of P elements of type IOType
		 * to a single element per process.
		 * The scatter shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #scatter.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be scattered value.
		 *
		 * @param[in]  in  The vector of \a P elements at the root process to be
		 *                 scattered.
		 * @param[out] out The scattered value of the root process \f$ vector[i] \f$
		 *                 at process \a i.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC scatter(
#ifdef BLAS1_RAW
			const IOType * in,
#else
			const Vector< IOType, reference, Coords > &in,
#endif
			IOType &out,
			const lpf_pid_t root
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: P * IOType -> IOType
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( in, data.P )
#endif

			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 2 ) != SUCCESS ) {
				return PANIC;
			}

			// create memslot on output vector
			lpf_memslot_t src, dest;
			src = dest = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
			if( lpf_register_global( data.context, &out, sizeof( IOType ), &dest )
				!= LPF_SUCCESS
			) {
				ret = PANIC;
			}
#ifndef BLAS1_RAW
			if( ret == SUCCESS && lpf_register_global(
					data.context,
					const_cast< IOType * >( internal::getRaw( in ) ),
					data.P * sizeof( IOType ),
					&src
				) != LPF_SUCCESS
			) {
#else
			if( ret == SUCCESS && lpf_register_global(
					data.context,
					const_cast< IOType * >( in ),
					data.P * sizeof( IOType ),
					&src
				) != LPF_SUCCESS
			) {
#endif
				// failure at this point will have to be cleaned up as best as possible
				ret = PANIC;
			}

			// root copies output
#ifndef BLAS1_RAW
			if( ret == SUCCESS && data.s == root &&
				&out != internal::getRaw( in ) + data.s
			) {
#else
			if( ret == SUCCESS && data.s == root && &out != in + data.s ) {
#endif
				out = in[ data.s ];
			}

			// activate global regs
			if( ret == SUCCESS &&
				lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// scatter values
			if( ret == SUCCESS &&
				lpf_scatter( coll, src, dest, sizeof( IOType ), root ) != LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// wait for completion of requested collective
			if( ret == SUCCESS &&
				lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// destroy memory slots
			if( src != LPF_INVALID_MEMSLOT &&
				lpf_deregister( data.context, src ) != LPF_SUCCESS
			) {
				ret = PANIC;
			}
			if( dest != LPF_INVALID_MEMSLOT &&
				lpf_deregister( data.context, dest ) != LPF_SUCCESS
			) {
				ret = PANIC;
			}

			// perform postamble
			if( commsPostamble( data, &coll, data.P, 0, 0, 2 ) != SUCCESS ) {
				ret = PANIC;
			}

			// done
			return ret;
		}

		/**
		 * Schedules a scatter operation of a vector of \a N elements of type IOType
		 * to a vector of \f$ N/P elements \f$ per process. It is assumed that \a N is
		 * a multiple of \a P. The gather shall be complete by the end of the call.
		 * This is a collective graphBLAS operation. The BSP costs are as for the LPF
		 * #gather.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be scattered value.
		 *
		 * @param[in]  in  The vector of N elements at the root process to be
		 *                 scattered.
		 * @param[out] out The scattered vector of the root process, such that process
		 *                 \a i has \f$ N/P \f$ elements located at offset
		 *                 \f$ (N/P)*i \f$.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ in.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC scatter(
#ifdef BLAS1_RAW
			const IOType * in,
			const size_t size,
			IOType * out,
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
			const size_t procs = data.P;

#ifndef BLAS1_RAW
			// make sure we can support comms pattern: Vector IOType -> Vector IOType
			TEST_VEC_MULTIPLE( out, in, procs )
			const size_t size = internal::getCoordinates( in ).size();
#endif

			lpf_coll_t coll;
			RC ret = commsPreamble( data, &coll, procs, 0, 0, 2 );

			lpf_memslot_t src = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dst = LPF_INVALID_MEMSLOT;
#ifdef BLAS1_RAW
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_register_global( data.context, const_cast< IOType * >( in ), size * sizeof( IOType ), &src );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_register_global( data.context, out, size / data.P * sizeof( IOType ), &dst );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
#else
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_register_global( data.context, const_cast< IOType * >( internal::getRaw( in ) ), internal::getCoordinates( in ).nonzeroes() * sizeof( IOType ), &src );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_register_global( data.context, internal::getRaw( out ), size / data.P * sizeof( IOType ), &dst );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
#endif

			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// scatter values
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_scatter( coll, src, dst, ( size / procs ) * sizeof( IOType ), root );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// local copy
			if( ret == SUCCESS && data.s == root ) {
				const size_t offset = root * ( size / procs );
#ifdef BLAS1_RAW
				(void)memcpy( out + offset, in + offset, ( size / procs ) * sizeof( IOType ) );
#else
				(void)memcpy( internal::getRaw( out ) + offset, internal::getRaw( in ) + offset, ( size / procs ) * sizeof( IOType ) );
#endif
			}

			// wait for requested communication to complete
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// destroy memory slots
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_deregister( data.context, src );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_deregister( data.context, dst );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// postamble
			if( ret == SUCCESS ) {
				ret = commsPostamble( data, &coll, procs, 0, 0, 2 );
			}

			// done
			return ret;
		}

		/**
		 * Schedules an allgather operation of a single object of type IOType per
		 * process to a vector of P elements.
		 *
		 * The allgather shall be complete by the end of the call. This is a
		 * collective graphBLAS operation. The BSP costs are as for the LPF
		 * #allgather.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be gathered value.
		 *
		 * @param[in]  in:  The value at the calling process to be gathered.
		 * @param[out] out: The vector of gathered values, available at each process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC allgather(
			IOType &in,
#ifdef BLAS1_RAW
			IOType * out
#else
			Vector< IOType, reference, Coords > &out
#endif
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: IOType -> P * IOType
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( out, data.P )
#endif

			// preamble
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 1, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// create a local register slot for in
			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
			if( ret == SUCCESS && lpf_register_local( data.context, &in, sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
#ifdef BLAS1_RAW
			if( ret == SUCCESS && lpf_register_global( data.context, out, data.P * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#else
			if( ret == SUCCESS && lpf_register_global( data.context, internal::getRaw( out ), data.P * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#endif

			// perform local copy
#ifndef BLAS1_RAW
			internal::getRaw( out )[ data.s ] = in;
#else
			out[ data.s ] = in;
#endif

			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// allgather values
			if( ret == SUCCESS && lpf_allgather( coll, in_slot, dest, sizeof( IOType ), false ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// complete communication
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

#ifndef BLAS1_RAW
			// correct sparsity info
			for( size_t i = 0; ret == SUCCESS && i < data.P; i++ ) {
				const size_t index = i;
				(void)internal::getCoordinates( out ).assign( index );
			}
#endif

			// do deregister
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, 0, 1, 1 ) != SUCCESS ) {
				ret = PANIC;
			}

			// done
			return ret;
		}

		/**
		 * Schedules an allgather operation of a vector of \a N/P elements of type
		 * IOType per process to a vector of \f$ N \f$ elements.
		 *
		 * The allgather shall be complete by the end of the call. This is a
		 * collective graphBLAS operation. The BSP costs are as for the LPF
		 * #allgather.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be gathered value.
		 *
		 * @param[in]  in:  The vector at the calling process to be gathered.
		 * @param[out] out: The vector of gathered values, available at each process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC allgather(
#ifdef BLAS1_RAW
			const IOType * in,
			const size_t size,
			IOType * out
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out
#endif
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: IOType -> P * IOType
#ifndef BLAS1_RAW
			TEST_VEC_MULTIPLE( in, out, data.P )
			const size_t size = in._cap;
#endif

			// preabmle
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, data.P * size * sizeof( IOType ), 1, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// create a local register slot for input and output
			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
#ifdef BLAS1_RAW
			if( ret == SUCCESS && lpf_register_local( data.context, in, size * sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( ret == SUCCESS && lpf_register_global( data.context, out, data.P * size * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#else
			if( ret == SUCCESS && lpf_register_local( data.context, internal::getRaw( in ), size * sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( ret == SUCCESS && lpf_register_global( data.context, internal::getRaw( out ), size * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#endif

			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// allgather values
			if( ret == SUCCESS && lpf_allgather( coll, in_slot, dest, size * sizeof( IOType ), false ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// complete the allgather
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// do deregister
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// copy local results into output vector
			for( size_t i = 0; i < size; i++ ) {
#ifdef BLAS1_RAW
				out[ data.s * size + i ] = in[ i ];
#else
				const size_t index = data.s * size + i;
				internal::getRaw( out )[ index ] = internal::getRaw( in )[ index ];
				(void)internal::getCoordinates( out ).assign( index );
#endif
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, data.P * size * sizeof( IOType ), 1, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/**
		 * Schedules an alltoall operation of a vector of P elements of type IOType
		 * per process to a vector of \a P elements.
		 *
		 * The alltoall shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #alltoall.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the vector elements.
		 *
		 * @param[in]  in  The vector of \a P elements at each process.
		 * @param[out] out The resulting vector of \a P elements, such that process
		 *                 \f$ i \f$ will receive (in order) the element at
		 *                 \f$ vector[i] \f$ from each process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC alltoall(
#ifdef BLAS1_RAW
			IOType * in,
			IOType * out
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out
#endif
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( in, data.P )
			TEST_VEC_SIZE( out, data.P )
#endif

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 2 ) != SUCCESS ) {
				return PANIC;
			}

			// create a global register slot for in
			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
#ifdef BLAS1_RAW
			if( ret == SUCCESS && lpf_register_global( data.context, in, data.P * sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( ret == SUCCESS && lpf_register_global( data.context, out, data.P * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#else
			if( ret == SUCCESS && lpf_register_global( data.context, const_cast< IOType * >( internal::getRaw( in ) ), data.P * sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( ret == SUCCESS && lpf_register_global( data.context, internal::getRaw( out ), data.P * sizeof( IOType ), &dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
#endif
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// alltoall values
			if( ret == SUCCESS && lpf_alltoall( coll, in_slot, dest, sizeof( IOType ) ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// finish communication request
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// do deregister
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, in_slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, dest ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// copy local results into output vector
#ifdef BLAS1_RAW
			if( ret == SUCCESS && out != in ) {
				out[ data.s ] = in[ data.s ];
			}
#else
			if( ret == SUCCESS && internal::getRaw( out ) != internal::getRaw( in ) ) {
				internal::getRaw( out )[ data.s ] = internal::getRaw( in )[ data.s ];
			}
			// update sparsity info
			for( size_t i = 0; ret == SUCCESS && internal::getCoordinates( out ).nonzeroes() != internal::getCoordinates( out ).size() && i < data.P; ++i ) {
				(void)internal::getCoordinates( out ).assign( i );
			}
#endif

			// postamble
			if( commsPostamble( data, &coll, data.P, 0, 0, 2 ) != SUCCESS ) {
				ret = PANIC;
			}

			// done
			return ret;
		}

		/**
		 * Schedules an allcombine operation of a vector of \a N/P elements of type
		 * IOType per process to a vector of \a N/P elements.
		 *
		 * The allcombine shall be complete by the end of the call. This is a
		 * collective graphBLAS operation. The BSP costs are as for the LPF
		 * #allcombine.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for combining.
		 * @tparam IOType   The type of the vector elements.
		 *
		 * @param[in,out]  inout The vector of \a N/P elements at each process. At
		 *                       the end of the call, each process shall hold the
		 *                       combined vectors.
		 * @param[in]      op    The associative operator to combine by.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics: allgather (N < P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ N*Operator \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$ ;
		 * -# transferred bytes: \f$ 2(N/P) \f$ ;
		 * -# BSP cost: \f$ 2(N/P)g + (N/P)*Operator + 2l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename IOType
#ifndef BLAS1_RAW
			, typename Coords
#endif
		>
		RC allcombine(
#ifdef BLAS1_RAW
			IOType * inout,
			const size_t size,
#else
			Vector< IOType, reference, Coords > &inout,
#endif
			const Operator op
		) {
			// static sanity check
			NO_CAST_ASSERT_BLAS1( ( !(descr & descriptors::no_casting) ||
					std::is_same< IOType, typename Operator::D1 >::value ||
					std::is_same< IOType, typename Operator::D2 >::value ||
					std::is_same< IOType, typename Operator::D3 >::value
				), "grb::collectives::allcombine",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set" );

			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif
			const size_t bytes = sizeof( IOType );

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, 2 * data.P, data.P * size * bytes, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// gather machine params
			lpf_machine_t machine;
			if( lpf_probe( data.context, &machine ) != LPF_SUCCESS ) {
				return PANIC;
			}
			const size_t P = data.P;
			const size_t me = data.s;
			const size_t N = size * bytes; // size on each process
			const double g = machine.g( P, size, LPF_SYNC_DEFAULT );
			const double l = machine.l( P, size, LPF_SYNC_DEFAULT );

			// one superstep basic approach: pNg + l
			// p small
			const double basic_cost = ( P * N * g ) + l;
			// two supersteps using transpose and gather: 2Ng + 2l
			// p large, N >= p
			const double transpose_cost = ( 2 * N * g ) + ( 2 * l );

			// choose basic approach if lowest cost or size small
			if( basic_cost < transpose_cost || size < P ) {
				// prepare buffer
				IOType * results = data.template getBuffer< IOType >();
				lpf_memslot_t out_slot = data.slot;

				// temporarily register inout vector memory area
				lpf_memslot_t vector_slot = LPF_INVALID_MEMSLOT;
#ifdef BLAS1_RAW
				lpf_err_t brt = lpf_register_global( data.context, inout, N, &vector_slot );
#else
				lpf_err_t brt = lpf_register_global( data.context, internal::getRaw( inout ), internal::getCoordinates( inout ).size() * bytes, &vector_slot );
#endif
				if( brt != LPF_SUCCESS ) {
					return PANIC;
				}

				if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// alltoall values
				if( lpf_allgather( coll, vector_slot, out_slot, size * bytes, true ) != LPF_SUCCESS ) {
					return PANIC;
				}
				if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// combine results into output vector
				for( size_t i = 0; i < data.P; i++ ) {
					if( i == data.s ) {
						continue;
					}
					for( size_t j = 0; j < size; j++ ) {
#ifdef BLAS1_RAW
						if( foldl< descr >( inout[ j ], results[ i * size + j ], op ) != SUCCESS ) {
#else
						if( foldl< descr >( internal::getRaw( inout )[ j ], results[ i * size + j ], op ) != SUCCESS ) {
#endif
							return PANIC;
						}
					}
				}

				// do deregister
				if( lpf_deregister( data.context, vector_slot ) != LPF_SUCCESS ) {
					return PANIC;
				}
			} else {
				// choose transpose and allgather
				// share message equally apart from maybe the final process
				size_t chunk = ( size + P - 1 ) / P;

				// step 1: my_chunk*size bytes from each process to my collectives slot

				lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
				lpf_t ctx = lpf_collectives_get_context( coll );
#ifdef BLAS1_RAW
				if( lpf_register_global( ctx, inout, N, &slot ) != LPF_SUCCESS ) {
#else
				if( lpf_register_global( ctx, internal::getRaw( inout ), N, &slot ) != LPF_SUCCESS ) {
#endif
					return PANIC;
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				size_t offset = chunk * me;
				size_t my_chunk = ( offset + chunk ) <= size ? chunk : ( size - offset );
				for( size_t pid = 0; pid < P; pid++ ) {
					const lpf_err_t rc = lpf_get( ctx, pid, slot, offset * bytes, data.slot, pid * my_chunk * bytes, my_chunk * bytes, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// step 2: combine the chunks and write to each process
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();
				for( size_t pid = 0; pid < P; pid++ ) {
					if( pid == me ) {
						continue;
					}
					for( size_t j = 0; j < my_chunk; j++ ) {
#ifdef BLAS1_RAW
						if( foldl< descr >( inout[ offset + j ], buffer[ pid * my_chunk + j ], op ) != SUCCESS ) {
#else
						if( foldl< descr >( internal::getRaw( inout )[ offset + j ], buffer[ pid * my_chunk + j ], op ) != SUCCESS ) {
#endif
							return PANIC;
						}
					}
				}
				for( size_t pid = 0; pid < P; pid++ ) {
					if( pid == me ) {
						continue;
					}
					const lpf_err_t rc = lpf_put( ctx, slot, offset * bytes, pid, slot, offset * bytes, my_chunk * bytes, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}
				// do deregister
				if( lpf_deregister( ctx, slot ) != LPF_SUCCESS ) {
					return PANIC;
				}
			}

			// postamble
			if( commsPostamble( data, &coll, 2 * data.P, data.P * size * bytes, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/**
		 * Schedules a combine operation of a vector of N/P elements of type IOType
		 * per process to a vector of N elements.
		 *
		 * The combine shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #combine.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for combining.
		 * @tparam IOType   The type of the vector elements.
		 *
		 * @param[in,out]  inout The vector of \a N/P elements at each process. At
		 *                       the end of the call, the root process shall hold the
		 *                       combined vectors.
		 * @param[in]      op    The associative operator to combine by.
		 * @param[in]      root  The root process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics: allgather (N < P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ N*Operator \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$ ;
		 * -# transferred bytes: \f$ 2(N/P) \f$ ;
		 * -# BSP cost: \f$ 2(N/P)g + (N/P)*Operator + 2l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 2(N/\sqrt{P})*Operator \f$ ;
		 * -# transferred bytes: \f$ 2(N/\sqrt{P}) \f$ ;
		 * -# BSP cost: \f$ 2(N/\sqrt{P})g + (N/\sqrt{P})*Operator + 2l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC combine(
#ifdef BLAS1_RAW
			IOType * inout,
			const size_t size,
#else
			Vector< IOType, reference, Coords > &inout,
#endif
			const Operator op,
			const lpf_pid_t root
		) {
			// static sanity check
			NO_CAST_ASSERT_BLAS1( ( !(descr & descriptors::no_casting) ||
					std::is_same< IOType, typename Operator::D1 >::value ||
					std::is_same< IOType, typename Operator::D2 >::value ||
					std::is_same< IOType, typename Operator::D3 >::value
				), "grb::collectives::combine",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set"
			);

			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif
			const size_t bytes = sizeof( IOType );

			if( commsPreamble( data, &coll, data.P, data.P * size * bytes, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// gather machine params
			lpf_machine_t machine;
			if( lpf_probe( data.context, &machine ) != LPF_SUCCESS ) {
				return PANIC;
			}
			const size_t P = data.P;
			const size_t me = data.s;
			const size_t N = size * bytes; // size on each process
			const double g = machine.g( P, size, LPF_SYNC_DEFAULT );
			const double l = machine.l( P, size, LPF_SYNC_DEFAULT );

			// one superstep basic approach: pNg + l
			// p small
			const double basic_cost = ( P * N * g ) + l;
			// two supersteps using transpose and gather: 2Ng + 2l
			// p large, N >= p
			const double transpose_cost = ( 2 * N * g ) + ( 2 * l );
			// two supersteps using sqrt(p) degree tree: 2sqrt(p)Ng + 2l
			// p large, N < p
			const double tree_cost = ( 2 * sqrt( (unsigned int)P ) * N * g ) + ( 2 * l );

			lpf_t ctx = lpf_collectives_get_context( coll );

			// choose basic cost if its the lowest
			if( basic_cost < transpose_cost && basic_cost < tree_cost ) {
				// copy input to buffer
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();
				size_t pos = ( me == root ) ? data.s : 0;
				for( size_t i = 0; i < size; i++ ) {
#ifdef BLAS1_RAW
					buffer[ pos * size + i ] = inout[ i ];
#else
					buffer[ pos * size + i ] = internal::getRaw( inout )[ i ];
#endif
				}

				// gather together values
				if( lpf_gather( coll, data.slot, data.slot, N, root ) != LPF_SUCCESS ) {
					return PANIC;
				}
				if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// fold everything: root only
				if( me == root ) {
					for( size_t j = 0; j < size; j++ ) {
						IOType tmp = buffer[ j ];
						for( size_t i = 1; i < P; i++ ) {
							// if casting is required to apply op, foldl will take care of this
							// note: the no_casting check could be deferred to foldl but this would result in unclear error messages
							if( foldl< descr >( tmp, buffer[ i * size + j ], op ) != SUCCESS ) {
								return PANIC;
							}
						}
#ifdef BLAS1_RAW
						inout[ j ] = tmp;
#else
						const size_t index = j;
						internal::getRaw( inout )[ index ] = tmp;
						(void)internal::getCoordinates( inout ).assign( index );
#endif
					}
				}
			} else if( size < P ) {
				// choose tree if N is too small to transpose
				// the (max) interval between each core process
				const size_t hop = sqrt( P );
				// the offset from my core process
				const size_t core_offset = DIFF( me, root, P ) % hop;
				// my core process
				const size_t core_home = DIFF( me, core_offset, P );
				// am i a core process
				const bool is_core = ( core_offset == 0 );
				// number of processes in my core group
				size_t core_count = hop;
				while( core_count > 1 ) {
					const size_t tmp_proc = me + ( core_count - 1 );
					const size_t tmp_core_offset = DIFF( tmp_proc, root, P ) % hop;
					const size_t tmp_core_home = DIFF( tmp_proc, tmp_core_offset, P );
					if( tmp_core_home == core_home ) {
						break;
					}
					--core_count;
				}

				// create a local register slot pointing at the vector
				lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
#ifdef BLAS1_RAW
				if( lpf_register_local( ctx, inout, N, &slot ) != LPF_SUCCESS ) {
#else
				if( lpf_register_local( ctx, internal::getRaw( inout ), N, &slot ) != LPF_SUCCESS ) {
#endif
					return PANIC;
				}
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();

				// step 1: all non-core processes write to their designated core process
				if( ! is_core ) {
					const lpf_err_t rc = lpf_put( ctx, slot, 0, core_home, data.slot, me * N, N, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// step 2: all core processes combine their results into their vector
				if( is_core ) {
					for( size_t k = 1; k < core_count; ++k ) {
						for( size_t j = 0; j < size; j++ ) {
#ifdef BLAS1_RAW
							if( foldl< descr >( inout[ j ], buffer[ ( ( me + k ) % P ) * size + j ], op ) != SUCCESS ) {
#else
							if( foldl< descr >( internal::getRaw( inout )[ j ], buffer[ ( ( me + k ) % P ) * size + j ], op ) != SUCCESS ) {
#endif
								return PANIC;
							}
						}
					}
				}
				// non-root processes will write their result to root
				if( is_core && me != root ) {
					const lpf_err_t rc = lpf_put( ctx, slot, 0, root, data.slot, me * N, N, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}
				// do deregister
				if( lpf_deregister( ctx, slot ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// step 3: root process combines its results from the core processes
				if( me == root ) {
					for( size_t k = hop; k < P; k += hop ) {
						for( size_t j = 0; j < size; j++ ) {
#ifdef BLAS1_RAW
							if( foldl< descr >( inout[ j ], buffer[ ( ( k + root ) % P ) * size + j ], op ) != SUCCESS ) {
#else
							if( foldl< descr >( internal::getRaw( inout )[ j ], buffer[ ( ( k + root ) % P ) * size + j ], op ) != SUCCESS ) {
#endif
								return PANIC;
							}
						}
					}
				}
			} else {
				// choose transpose and gather
				// share message equally apart from maybe the final process
				size_t chunk = ( size + P - 1 ) / P;

				// step 1: my_chunk*size bytes from each process to my collectives slot
				lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
#ifdef BLAS1_RAW
				if( lpf_register_global( ctx, inout, N, &slot ) != LPF_SUCCESS ) {
#else
				if( lpf_register_global( ctx, internal::getRaw( inout ), N, &slot ) != LPF_SUCCESS ) {
#endif
					return PANIC;
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				const size_t offset = chunk * me;
				const size_t my_chunk = ( offset + chunk ) <= size ? chunk : ( size - offset );
				for( size_t pid = 0; pid < P; pid++ ) {
					const lpf_err_t rc = lpf_get( ctx, pid, slot, offset * bytes, data.slot, pid * my_chunk * bytes, my_chunk * bytes, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}

				// step 2: combine the chunks and write to the root process
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();
				for( size_t pid = 0; pid < P; pid++ ) {
					if( pid == me ) {
						continue;
					}
					for( size_t j = 0; j < my_chunk; j++ ) {
#ifdef BLAS1_RAW
						if( foldl< descr >( inout[ offset + j ], buffer[ pid * my_chunk + j ], op ) != SUCCESS ) {
#else
						if( foldl< descr >( internal::getRaw( inout )[ offset + j ], buffer[ pid * my_chunk + j ], op ) != SUCCESS ) {
#endif
							return PANIC;
						}
					}
				}
				if( me != root ) {
					const lpf_err_t rc = lpf_put( ctx, slot, offset * bytes, root, slot, offset * bytes, my_chunk * bytes, LPF_MSG_DEFAULT );
					if( rc != LPF_SUCCESS ) {
						return PANIC;
					}
				}
				if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
					return PANIC;
				}
				// do deregister
				if( lpf_deregister( ctx, slot ) != LPF_SUCCESS ) {
					return PANIC;
				}
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, data.P * size * bytes, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		/**
		 * Schedules a reduce operation of a vector of N/P elements of type IOType per
		 * process to a single element.
		 *
		 * The reduce shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #reduce.
		 *
		 * Since this is a collective call, there are \a N/P values \a in at each
		 * process. Let these vectors be denoted by \f$ x_s \f$, with
		 * \f$ s \in \{ 0, 1, \ldots, P-1 \}, \f$ such that \f$ x_s \f$ equals the
		 * argument \a in on input at the user process with ID \a s. Let
		 * \f$ \pi:\ \{ 0, 1, \ldots, P-1 \} \to \{ 0, 1, \ldots, P-1 \} \f$ be a
		 * bijection, some unknown permutation of the process ID. This permutation is
		 * must be fixed for any given combination of GraphBLAS implementation and value
		 * \a P. Let the binary operator \a op be denoted by \f$ \odot \f$.
		 *
		 * This function computes \f$ \odot_{i=0}^{P-1} x_{\pi(i)} \f$ and writes the
		 * result to \a out at the user process with ID \a root.
		 *
		 * In summary, this the result is reproducible across different runs using the
		 * same input and \a P. Yet it does \em not mean that the order of addition is
		 * fixed.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for reduction.
		 * @tparam InputType The type of the input vector elements.
		 * @tparam IOType   The type of the to-be reduced value.
		 *
		 * @param[in]  in:   The vector at the calling process to be reduced.
		 * @param[out] out:  The value of the result of the reduction, at the root process.
		 * @param[in]  op:   The associative operator to reduce by.
		 * @param[in]  root: The id of the root process.
		 *
		 * \note If \op is commutative, the implementation is free to employ a different
		 *       allreduce algorithm, as long as it is documented well enough so that
		 *       its cost can be quantified.
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{InputType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$ ;
		 * -# transferred bytes: \f$ P \f$ ;
		 * -# BSP cost: \f$ Pg + (N/P)*Operator + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC reduce(
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > &in,
#endif
			IOType &out,
			const Operator op,
			const lpf_pid_t root
		) {
			// static sanity check
			NO_CAST_ASSERT_BLAS1( ( !(descr & descriptors::no_casting) ||
					std::is_same< InputType, typename Operator::D1 >::value ||
					std::is_same< IOType, typename Operator::D2 >::value ||
					std::is_same< IOType, typename Operator::D3 >::value
				), "grb::collectives::reduce",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set"
			);

			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: IOType -> P * IOType
			lpf_coll_t coll;
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( in ).size();
#endif
			if( commsPreamble( data, &coll, data.P, data.P * sizeof( IOType ), 1 ) != SUCCESS ) {
				return PANIC;
			}

			// reduce our values locally
			// if casting is required to apply op, foldl will take care of this
			// note: the no_casting check could be deferred to foldl but this would result in unclear error messages
			if( data.s == root ) {
				for( size_t i = 0; i < size; i++ ) {
#ifdef BLAS1_RAW
					if( foldl< descr >( out, in[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
#else
					if( foldl< descr >( out, internal::getRaw( in )[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
#endif
				}
			} else {
#ifdef BLAS1_RAW
				out = in[ 0 ];
#else
				out = internal::getRaw( in )[ 0 ];
#endif
				for( size_t i = 1; i < size; i++ ) {
#ifdef BLAS1_RAW
					if( foldl< descr >( out, in[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
#else
					if( foldl< descr >( out, internal::getRaw( in )[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
#endif
				}
			}

			// create a register slot
			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			if( lpf_register_local( data.context, &out, sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// gather together values
			if( lpf_gather( coll, in_slot, data.slot, sizeof( IOType ), root ) != LPF_SUCCESS ) {
				return PANIC;
			}
			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// do deregister
			if( lpf_deregister( data.context, in_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// fold gathered results
			if( data.s == root ) {
				IOType * __restrict__ const buffer = data.getBuffer< IOType >();
				for( size_t i = 0; i < data.P; i++ ) {
					if( i == root ) {
						continue;
					}
					// if casting is required to apply op, foldl will take care of this
					// note: the no_casting check could be deferred to foldl but this would result in unclear error messages
					if( foldl< descr >( out, buffer[ i ], op ) != SUCCESS ) {
						return PANIC;
					}
				}
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, data.P * sizeof( IOType ), 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		// reduce to the left
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC reducel(
			IOType &out,
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > &in,
#endif
			const Operator op,
			const lpf_pid_t root
		) {
#ifdef BLAS1_RAW
			return reduce( in, size, out, op, root );
#else
			return reduce( in, out, op, root );
#endif
		}

		// reduce to the right
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC reducer(
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > & in,
#endif
			IOType &out,
			const Operator op,
			const lpf_pid_t root
		) {
#ifdef BLAS1_RAW
			return reduce( in, size, out, op, root );
#else
			return reduce( in, out, op, root );
#endif
		}

		/**
		 * Schedules an allreduce operation of a vector of N/P elements of type IOType
		 * per process to a single element.
		 *
		 * The allreduce shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the LPF #allreduce.
		 *
		 * Since this is a collective call, there are \a N/P values \a in at each process
		 * Let these vectors be denoted by \f$ x_s \f$, with
		 * \f$ s \in \{ 0, 1, \ldots, P-1 \}, \f$ such that \f$ x_s \f$ equals the
		 * argument \a in on input at the user process with ID \a s. Let
		 * \f$ \pi:\ \{ 0, 1, \ldots, P-1 \} \to \{ 0, 1, \ldots, P-1 \} \f$ be a
		 * bijection, some unknown permutation of the process ID. This permutation is
		 * must be fixed for any given combination of GraphBLAS implementation and value
		 * \a P. Let the binary operator \a op be denoted by \f$ \odot \f$.
		 *
		 * This function computes \f$ \odot_{i=0}^{P-1} x_{\pi(i)} \f$ and writes the
		 * result to \a out at each process.
		 *
		 * In summary, this the result is reproducible across different runs using the
		 * same input and \a P. Yet it does \em not mean that the order of addition is
		 * fixed.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for reduction.
		 * @tparam InputType The type of the input vector elements.
		 * @tparam IOType   The type of the to-be reduced value.
		 *
		 * @param[in]  in:   The vector at the calling process to be reduced.
		 * @param[out] out:  The value of the result of the reduction, at each process.
		 * @param[in]  op:   The associative operator to reduce by.
		 *
		 * \note If \op is commutative, the implementation is free to employ a different
		 *       allreduce algorithm, as long as it is documented well enough so that
		 *       its cost can be quantified.
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{InputType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$ ;
		 * -# transferred bytes: \f$ P \f$ ;
		 * -# BSP cost: \f$ Pg + (N/P)*Operator + l \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC allreduce(
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > &in,
#endif
			IOType &out,
			const Operator op
		) {
			// static sanity check
			NO_CAST_ASSERT_BLAS1( ( !(descr & descriptors::no_casting) ||
					std::is_same< InputType, typename Operator::D1 >::value ||
					std::is_same< IOType, typename Operator::D2 >::value ||
					std::is_same< IOType, typename Operator::D3 >::value
				), "grb::collectives::allreduce",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set"
			);

			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// make sure we can support comms pattern: P * IOType
			lpf_coll_t coll;
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( in ).size();
#endif
			if( commsPreamble( data, &coll, data.P, data.P * sizeof( IOType ), 1 ) != SUCCESS ) {
				return PANIC;
			}

			// reduce our values locally
			// if casting is required to apply op, foldl will take care of this
			// note: the no_casting check could be deferred to foldl but this would
			//       result in unclear error messages
			for( size_t i = 0; i < size; i++ ) {
#ifdef BLAS1_RAW
				if( foldl< descr >( out, in[ i ], op ) != SUCCESS ) {
					return PANIC;
				}
#else
				if( foldl< descr >( out, internal::getRaw( in )[ i ], op ) != SUCCESS ) {
					return PANIC;
				}
#endif
			}

			// create a register slot
			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			if( lpf_register_local( data.context, &out, sizeof( IOType ), &in_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// gather together values
			if( lpf_allgather( coll, in_slot, data.slot, sizeof( IOType ), 1 ) != LPF_SUCCESS ) {
				return PANIC;
			}
			if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// do deregister
			if( lpf_deregister( data.context, in_slot ) != LPF_SUCCESS ) {
				return PANIC;
			}

			// fold everything
			IOType * __restrict__ const buffer = data.getBuffer< IOType >();
			for( size_t i = 0; i < data.P; i++ ) {
				if( i == data.s ) {
					continue;
				}
				// if casting is required to apply op, foldl will take care of this
				// note: the no_casting check could be deferred to foldl but this would
				//       result in unclear error messages
				if( foldl< descr >( out, buffer[ i ], op ) != SUCCESS ) {
					return PANIC;
				}
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, data.P * sizeof( IOType ), 1 ) != SUCCESS ) {
				return PANIC;
			}

			// done
			return SUCCESS;
		}

		// allreduce to the left
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC allreducel(
			IOType &out,
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > &in,
#endif
			const Operator op
		) {
#ifdef BLAS1_RAW
			return allreduce( in, size, out, op );
#else
			return allreduce( in, out, op );
#endif
		}

		// allreduce to the right
		template<
			Descriptor descr = descriptors::no_operation,
			typename Operator,
			typename InputType,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC allreducer(
#ifdef BLAS1_RAW
			const InputType * in,
			const size_t size,
#else
			const Vector< InputType, reference, Coords > & in,
#endif
			IOType &out,
			const Operator op
		) {
#ifdef BLAS1_RAW
			return allreduce( in, size, out, op );
#else
			return allreduce( in, out, op );
#endif
		}

		/**
		 * Schedules a broadcast operation of a vector of N elements of type IOType
		 * to a vector of N elements per process.
		 *
		 * The broadcast shall be complete by the end of the call. This is a
		 * collective graphBLAS operation. The BSP costs are as for the LPF
		 * #broadcast.
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam IOType   The type of the to-be broadcast vector element values.
		 *
		 * @param[in,out] inout On input: the vector at the root process to be
		 *                      broadcast.
		 *                      On output at process \a root: the same value.
		 *                      On output at non-root processes: the vector at root.
		 *
		 * \parblock
		 * \par Performance semantics: serial
		 * -# Problem size N: \f$ inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ NP \f$ ;
		 * -# BSP cost: \f$ NPg + l \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two hase
		 * -# Problem size N: \f$ inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2N \f$ ;
		 * -# BSP cost: \f$ 2(Ng + l) \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2\sqrt{P}N \f$ ;
		 * -# BSP cost: \f$ 2(\sqrt{P}Ng + l) \f$;
		 * \endparblock
		 *
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType
#ifndef BLAS1_RAW
			,
			typename Coords
#endif
		>
		RC broadcast(
#ifdef BLAS1_RAW
			IOType * inout,
			const size_t size,
#else
			Vector< IOType, reference, Coords > &inout,
#endif
			const lpf_pid_t root
		) {
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif

			// preamble
			lpf_coll_t coll;
			if( commsPreamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				return PANIC;
			}

			// create memslot
			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			RC ret = SUCCESS;
#ifndef BLAS1_RAW
			if( ret == SUCCESS && lpf_register_global( data.context, internal::getRaw( inout ), size * sizeof( IOType ), &slot ) != LPF_SUCCESS ) {
#else
			if( ret == SUCCESS && lpf_register_global( data.context, const_cast< typename std::remove_const< IOType >::type * >( inout ), size * sizeof( IOType ), &slot ) != LPF_SUCCESS ) {
#endif
				ret = PANIC;
			}
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// broadcast value
			if( ret == SUCCESS && lpf_broadcast( coll, slot, slot, size * sizeof( IOType ), root ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// finish requested comm
			if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// destroy memslot
			if( slot != LPF_INVALID_MEMSLOT && lpf_deregister( data.context, slot ) != LPF_SUCCESS ) {
				ret = PANIC;
			}

			// postamble
			if( commsPostamble( data, &coll, data.P, 0, 0, 1 ) != SUCCESS ) {
				ret = PANIC;
			}

			// done
			return ret;
		}

	} // namespace internal

} // namespace grb

