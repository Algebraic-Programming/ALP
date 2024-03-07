
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

#include <graphblas/blas0.hpp>
#include <graphblas/final.hpp>

#include <graphblas/base/vector.hpp>

#include <graphblas/bsp/error.hpp>

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
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			IOType * const out,
#else
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::gather (BSP), raw variant, scalar" << std::endl;
 #else
			std::cout << "In internal::gather (BSP), grb variant, scalar" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// run-time sanity check
			if( root >= data.P ) {
				return ILLEGAL;
			}
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( out, data.P )
#else
			if( out == nullptr ) {
				return ILLEGAL;
			}
#endif

			// copy input to buffer -- saves one LPF registration
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

			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			RC ret = SUCCESS;
			if( data.P > 1 ) {
				// preliminaries
				ret = data.ensureMemslotAvailable();
				if( ret == SUCCESS ) {
					ret = data.ensureMaxMessages( data.P - 1 );
				}
				if( ret == SUCCESS ) {
					ret = data.ensureCollectivesCapacity( 1, 0, sizeof( IOType ) );
				}
				if( ret != SUCCESS ) { return ret; }

				// create memslot on output vector
#ifndef BLAS1_RAW
				lpf_rc = lpf_register_global( data.context, internal::getRaw( out ),
					data.P * sizeof( IOType ), &slot );
#else
				lpf_rc = lpf_register_global( data.context, out, data.P * sizeof( IOType ),
					&slot );
#endif
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
				ret = checkLPFerror( lpf_rc, "internal::gather (scalar, BSP)" );

				// gather values
				if( ret == SUCCESS ) {
					lpf_rc = lpf_gather( data.coll, slot, slot, sizeof( IOType ), root );
					if( lpf_rc == LPF_SUCCESS ) {
						lpf_sync( data.context, LPF_SYNC_DEFAULT );
					}
					ret = checkLPFerror( lpf_rc, "internal::gather (scalar, BSP)" );
				}
			}

#ifndef BLAS1_RAW
			if( ret == SUCCESS ) {
				if( data.s == root ) {
					// make sure sparsity info is correct
					internal::getCoordinates( out ).template assignAll< true >();
				}
			}
#endif

			// deregister slot
			if( slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, slot );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::gather (BSP), raw variant, scalar: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::gather (BSP), grb variant, scalar: exiting"
				<< std::endl;
 #endif
#endif
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
		 * @param[in]  in  The vector at the calling process to be gathered.
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
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
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
			const IOType * const in,
			const size_t size,
			IOType * const out,
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::gather (BSP), raw variant, vector" << std::endl;
 #else
			std::cout << "In internal::gather (BSP), grb variant, vector" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// dynamic checks
			if( root >= data.P ) {
				return ILLEGAL;
			}
#ifndef BLAS1_RAW
			TEST_VEC_MULTIPLE( in, out, data.P )
			const size_t size = internal::getCoordinates( in ).size();
#else
			if( in == nullptr || out == nullptr ) {
				return ILLEGAL;
			}
#endif
			const size_t bsize = size * sizeof( IOType );

			// check trivial dispatch
			if( size == 0 ) {
				return SUCCESS;
			}

			lpf_memslot_t in_slot, out_slot;
			in_slot = out_slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			RC ret = SUCCESS;
			if( data.P > 1 ) {
				// preliminaries
				ret = data.ensureCollectivesCapacity( 1, 0, bsize );
				if( ret == SUCCESS ) {
					ret = data.ensureMemslotAvailable( 2 );
				}
				if( ret == SUCCESS ) {
					ret = data.ensureMaxMessages( data.P - 1 );
				}
				if( ret != SUCCESS ) { return ret; }

				// create memslot on output vector
#ifndef BLAS1_RAW
				lpf_rc = lpf_register_global( data.context, internal::getRaw( out ),
					size * data.P * sizeof( IOType ), &out_slot );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_register_local( data.context,
						const_cast< IOType * >(internal::getRaw( in )), bsize, &in_slot );
				}
#else
				lpf_rc = lpf_register_global( data.context, out, data.P * bsize, &out_slot );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_register_local( data.context, const_cast< IOType * >(in), bsize,
						&in_slot );
				}
#endif
				// activate registrations
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// gather values
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_gather( data.coll, in_slot, out_slot, bsize, root );
				}

				// complete requested communication
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// done with LPF section
				ret = checkLPFerror( lpf_rc, "internal::gather (vector, BSP)" );
			}

			// do self-copy, if required
			if( ret == SUCCESS && data.s == root ) {
#ifndef BLAS1_RAW
				const void * const in_p = internal::getRaw(in);
				char * const out_p = reinterpret_cast< char * >(internal::getRaw( out ));
#else
				const void * const in_p = in;
				char * const out_p = reinterpret_cast< char * >(out);
#endif
				if( out_p + data.s * size != in_p ) {
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy(
						out_p + data.s * size, in_p, bsize );
				}
			}

#ifndef BLAS1_RAW
			if( ret == SUCCESS ) {
				// set sparsity of output
				internal::getCoordinates( out ).template assignAll< true >();
			}
#endif

			// destroy memory slot
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, in_slot );
			}
			if( out_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, out_slot );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::gather (BSP), raw variant, vector: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::gather (BSP), grb variant, vector: exiting"
				<< std::endl;
 #endif
#endif
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
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			const IOType * const in,
#else
			const Vector< IOType, reference, Coords > &in,
#endif
			IOType &out,
			const lpf_pid_t root
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::scatter (BSP), raw variant, scalar" << std::endl;
 #else
			std::cout << "In internal::scatter (BSP), grb variant, scalar" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// dynamic checks
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( in, data.P )
#endif
			if( root >= data.P ) {
				return ILLEGAL;
			}
#ifdef BLAS1_RAW
			if( in == nullptr ) {
				return ILLEGAL;
			}
#endif

			lpf_memslot_t src, dest;
			src = dest = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			RC ret = SUCCESS;
			if( data.P > 1 ) {
				// preliminaries
				ret = data.ensureMemslotAvailable( 2 );
				if( ret == SUCCESS ) {
					data.ensureCollectivesCapacity( 1, 0, sizeof( IOType ) );
				}
				if( ret != SUCCESS ) { return ret; }

				// create memslot on output vector
				lpf_rc = lpf_register_local( data.context, &out, sizeof( IOType ),
					&dest );
				if( lpf_rc == LPF_SUCCESS ) {
#ifndef BLAS1_RAW
					lpf_rc = lpf_register_global(
						data.context,
						const_cast< IOType * >( internal::getRaw( in ) ),
						data.P * sizeof( IOType ),
						&src
					);
#else
					lpf_rc = lpf_register_global(
						data.context,
						const_cast< IOType * >( in ),
						data.P * sizeof( IOType ),
						&src
					);
#endif
				}

				// activate global regs
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// scatter values
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_scatter( data.coll, src, dest, sizeof( IOType ), root );
				}

				// wait for completion of requested collective
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// end of LPF section
				ret = checkLPFerror( lpf_rc, "internal::scatter (scalar, BSP)" );
			}

			if( ret == SUCCESS ) {
				// root copies output
#ifndef BLAS1_RAW
				if( data.s == root && &out != internal::getRaw( in ) + data.s ) {
#else
				if( data.s == root && &out != in + data.s ) {
#endif
					out = in[ data.s ];
				}
			}

			// destroy memory slots
			if( src != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, src );
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, dest );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::scatter (BSP), raw variant, scalar: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::scatter (BSP), grb variant, scalar: exiting"
				<< std::endl;
 #endif
#endif
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
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			const IOType * const in,
			const size_t size,
			IOType * const out,
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out,
#endif
			const lpf_pid_t root
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::scatter (BSP), raw variant, vector" << std::endl;
 #else
			std::cout << "In internal::scatter (BSP), grb variant, vector" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
			const size_t procs = data.P;

#ifndef BLAS1_RAW
			// make sure we can support comms pattern: Vector IOType -> Vector IOType
			TEST_VEC_MULTIPLE( out, in, procs )
			const size_t size = internal::getCoordinates( in ).size();
#endif
			// dynamic checks
			if( root >= procs ) {
				return ILLEGAL;
			}
			if( size % procs > 0 ) {
				return ILLEGAL;
			}
#ifdef BLAS1_RAW
			if( in == nullptr || out == nullptr ) {
				return ILLEGAL;
			}
#endif
			if( size == 0 ) {
				return SUCCESS;
			}

			const size_t lsize = size / data.P;
			const size_t bsize = lsize * sizeof( IOType );
			lpf_memslot_t src = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dst = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			RC ret = SUCCESS;
			if( data.P > 1 ) {
				// preliminaries
				ret = data.ensureCollectivesCapacity( 1, 0, bsize );
				if( ret == SUCCESS ) {
					ret = data.ensureMemslotAvailable( 2 );
				}
				if( ret == SUCCESS ) {
					ret = data.ensureMaxMessages( data.P - 1 );
				}
				if( ret != SUCCESS ) { return ret; }

				// create memslots
#ifdef BLAS1_RAW
				if( data.s == root ) {
					lpf_rc = lpf_register_global( data.context, const_cast< IOType * >( in ),
						size * sizeof( IOType ), &src );
				} else {
					lpf_rc = lpf_register_global( data.context, const_cast< IOType * >( in ),
						0, &src );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_local( data.context, out, bsize, &dst );
				}
#else
				if( data.s == root ) {
					lpf_rc = lpf_register_global(
						data.context,
						const_cast< IOType * >( internal::getRaw( in ) ),
						internal::getCoordinates( in ).nonzeroes() * sizeof( IOType ),
						&src
					);
				} else {
					lpf_rc = lpf_register_global( data.context, nullptr, 0, &src );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_local( data.context, internal::getRaw( out ),
						size / data.P * sizeof( IOType ), &dst );
				}
#endif
				// activate memslots
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// schedule & exec scatter
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_scatter( data.coll, src, dst, bsize, root );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// end of LPF section
				ret = checkLPFerror( lpf_rc, "internal::scatter (vector, BSP)" );
			}

			// local copy, if needed
			if( ret == SUCCESS ) {
				if( data.s == root ) {
					const size_t offset = root * ( size / procs );
#ifdef BLAS1_RAW
					if( out + offset != in + offset ) {
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy(
							out + offset, in + offset, bsize );
					}
#else
					if( internal::getRaw(out) + offset != internal::getRaw(in) + offset ) {
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy(
							internal::getRaw( out ) + offset,
							internal::getRaw( in ) + offset,
							bsize
						);
					}
#endif
				}
			}

			// destroy memory slots
			if( src != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, src );
			}
			if( dst != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, dst );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::scatter (BSP), raw variant, vector: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::scatter (BSP), grb variant, vector: exiting"
				<< std::endl;
 #endif
#endif
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
		 * @param[in]  in   The value at the calling process to be gathered.
		 * @param[out] out  The vector of gathered values, available at each process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			IOType * const out
#else
			Vector< IOType, reference, Coords > &out
#endif
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::allgather (BSP), raw variant, scalar"
				<< std::endl;
 #else
			std::cout << "In internal::allgather (BSP), grb variant, scalar"
				<< std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// dynamic checks
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( out, data.P )
#else
			if( out == nullptr ) {
				return ILLEGAL;
			}
#endif
			// check trivial op
			if( data.P == 1 ) {
#ifdef BLAS1_RAW
				*out = in;
				return SUCCESS;
#else
				*(internal::getRaw( out )) = in;
				internal::getCoordinates( out ).template assignAll< true >();
				return SUCCESS;
#endif
			}

			// preliminaries
			const size_t bsize = data.P * sizeof( IOType );
			RC ret = data.ensureMaxMessages( 2 * data.P );
			if( ret == SUCCESS ) {
				ret = data.ensureMemslotAvailable();
			}
			if( ret == SUCCESS ) {
				ret = data.ensureCollectivesCapacity( 1, 0, sizeof( IOType ) );
			}
			if( ret == SUCCESS ) {
				ret = data.ensureBufferSize( sizeof( IOType ) );
			}
			if( ret != SUCCESS ) { return ret; }

			// copy input to buffer -- saves one registration
			IOType * const __restrict__ buffer = data.template getBuffer< IOType >();
			*buffer = in;

			// create and activate a global memslot for out
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
#ifdef BLAS1_RAW
			lpf_rc = lpf_register_global( data.context, out, bsize, &dest );
#else
			lpf_rc = lpf_register_global( data.context, internal::getRaw( out ),
				bsize, &dest );
#endif
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			}

			// schedule and execute allgather
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_allgather( data.coll, data.slot, dest, sizeof( IOType ),
					true );
			}
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			}
			ret = checkLPFerror( lpf_rc, "internal::allgather (scalar, BSP)" );

			// if all is OK, set output vector structure and copy our own local value
			if( ret == SUCCESS ) {
#ifndef BLAS1_RAW
				if( internal::getRaw(out) + data.s != &in ) {
					internal::getRaw(out)[ data.s ] = in;
				}
				internal::getCoordinates( out ).template assignAll< true >();
#else
				if( out + data.s != &in ) {
					out[ data.s ] = in;
				}
#endif
			}

			// do deregister
			if( dest != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, dest );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::allgather (BSP), raw variant, scalar: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::allgather (BSP), grb variant, scalar: exiting"
				<< std::endl;
 #endif
#endif
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
		 * @tparam descr   The GraphBLAS descriptor.
		 *                 Default is grb::descriptors::no_operation.
		 * @tparam IOType  The type of the to-be gathered value.
		 *
		 * @param[in]  in  The vector at the calling process to be gathered.
		 * @param[out] out The vector of gathered values, available at each process.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * in.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			const IOType * const in,
			const size_t size,
			IOType * const out
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out
#endif
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::allgather (BSP), raw variant, vector"
				<< std::endl;
 #else
			std::cout << "In internal::allgather (BSP), grb variant, vector"
				<< std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// dynamic checks
#ifndef BLAS1_RAW
			TEST_VEC_MULTIPLE( in, out, data.P )
			const size_t size = in._cap;
#else
			if( in == nullptr || out == nullptr ) {
				return ILLEGAL;
			}
#endif
			// check for trivial op
			if( size == 0 ) {
				return SUCCESS;
			}

			// preliminaries
			const size_t bsize = size * sizeof( IOType );
			RC ret = data.ensureMemslotAvailable( 2 );
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( 2 * data.P );
			}
			if( ret == SUCCESS ) {
				ret = data.ensureCollectivesCapacity( 1, 0, bsize );
			}
			if( ret != SUCCESS ) { return ret; }

			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			if( data.P > 1 ) {
				// create a local register slot for input and output
#ifdef BLAS1_RAW
				lpf_rc = lpf_register_local( data.context, in, bsize, &in_slot );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_global( data.context, out, data.P * bsize, &dest );
				}
#else
				lpf_rc = lpf_register_local( data.context, internal::getRaw( in ), bsize,
					&in_slot );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_global( data.context, internal::getRaw( out ), bsize,
						&dest );
				}
#endif
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_allgather( data.coll, in_slot, dest, bsize, false );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
				ret = checkLPFerror( lpf_rc, "internal::allgather (vector, BSP)" );
			}

			if( ret == SUCCESS ) {
				// copy local results into output vector
#ifdef BLAS1_RAW
				const void * const in_p = in;
				void * const out_p = out + data.s * size;
#else
				const void * const in_p = internal::getRaw( in );
				void * const out_p = internal::getRaw( out ) + data.s * size;
#endif
				internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy( out_p, in_p, bsize );
				internal::getCoordinates( out ).template assignAll< true >();
			}

			// do deregister
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, in_slot );
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, dest );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::allgather (BSP), raw variant, vector: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::allgather (BSP), grb variant, vector: exiting"
				<< std::endl;
 #endif
#endif
			// done
			return SUCCESS;
		}

		/**
		 * Schedules an alltoall operation of a vector of P elements of type IOType
		 * per process to a vector of \a P elements.
		 *
		 * The alltoall shall be complete by the end of the call. This is a collective
		 * GraphBLAS operation. The BSP costs are as for the LPF #alltoall.
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
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ N \f$;
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
			IOType * const in,
			IOType * const out
#else
			const Vector< IOType, reference, Coords > &in,
			Vector< IOType, reference, Coords > &out
#endif
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::alltoall (BSP), raw variant" << std::endl;
 #else
			std::cout << "In internal::alltoall (BSP), grb variant" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

			// dynamic checks
#ifndef BLAS1_RAW
			TEST_VEC_SIZE( in, data.P )
			TEST_VEC_SIZE( out, data.P )
#else
			if( in == nullptr || out == nullptr ) {
				return ILLEGAL;
			}
#endif

			lpf_memslot_t in_slot = LPF_INVALID_MEMSLOT;
			lpf_memslot_t dest = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			RC ret = SUCCESS;
			if( data.P > 1 ) {
				// preliminaries
				const size_t bsize = data.P * sizeof( IOType );
				ret = data.ensureCollectivesCapacity( 1, 0, bsize );
				if( ret == SUCCESS ) {
					data.ensureMemslotAvailable( 2 );
				}
				if( ret == SUCCESS ) {
					data.ensureMaxMessages( 2 * data.P - 2 );
				}
				if( ret != SUCCESS ) { return ret; }

				// create a global register slot for in
#ifdef BLAS1_RAW
				lpf_rc = lpf_register_global( data.context, in, bsize, &in_slot );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_global( data.context, out, bsize, &dest );
				}
#else
				lpf_rc = lpf_register_global(
						data.context,
						const_cast< IOType * >( internal::getRaw( in ) ), bsize,
						&in_slot
					);
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_register_global(
						data.context,
						internal::getRaw( out ), bsize,
						&dest
					);
				}
#endif
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_alltoall( data.coll, in_slot, dest, sizeof( IOType ) );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
				ret = checkLPFerror( lpf_rc, "internal::alltoall (BSP)" );
			}

			if( ret == SUCCESS ) {
				// copy local results into output vector
#ifdef BLAS1_RAW
				if( out != in ) {
					out[ data.s ] = in[ data.s ];
				}
#else
				if( internal::getRaw( out ) != internal::getRaw( in ) ) {
					internal::getRaw( out )[ data.s ] = internal::getRaw( in )[ data.s ];
				}
				// update sparsity info
				internal::getCoordinates( out ).template assignAll< true >();
#endif
			}

			// do deregister
			if( in_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, in_slot );
			}
			if( dest != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, dest );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::alltoall (BSP), raw variant: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::alltoall (BSP), grb variant: exiting"
				<< std::endl;
 #endif
#endif
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
		 * -# local work: \f$ N*Operator \f$;
		 * -# transferred bytes: \f$ N \f$;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$;
		 * -# transferred bytes: \f$ 2(N/P) \f$;
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
			IOType * const inout,
			const size_t size,
#else
			Vector< IOType, reference, Coords > &inout,
#endif
			const Operator &op
		) {
			// static sanity check
			NO_CAST_ASSERT_BLAS1( ( !(descr & descriptors::no_casting) ||
					std::is_same< IOType, typename Operator::D1 >::value ||
					std::is_same< IOType, typename Operator::D2 >::value ||
					std::is_same< IOType, typename Operator::D3 >::value
				), "grb::collectives::allcombine",
				"Incompatible given value type and operator domains while "
				"no_casting descriptor was set" );
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::allcombine (BSP), raw variant" << std::endl;
 #else
			std::cout << "In internal::allcombine (BSP), grb variant" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif
			const size_t bytes = sizeof( IOType );
			const size_t bsize = size * bytes;

			// dynamic sanity checks
#ifdef BLAS1_RAW
			if( inout == nullptr ) {
				return ILLEGAL;
			}
#endif

			// check trivial calls
			if( size == 0 ) {
				return SUCCESS;
			}
			if( data.P == 1 ) {
				return SUCCESS;
			}

			// determine which variant to follow
			enum Variant { ONE_STEP, TWO_STEP };
			Variant variant = ONE_STEP;
			{
				const size_t P = data.P;
				const size_t N = bsize;
				const double g = data.getMessageGap( bsize );
				const double l = data.getLatency( bsize );

				// one superstep basic approach: pNg + l, applicable for small p, N
				const double basic_cost = ( P * N * g ) + l;

				// two supersteps using transpose and gather: 2Ng + 2l, applicable for large
				// p or when N is very large
				const double transpose_cost = ( 2 * N * g ) + ( 2 * l );

				// go for two-step if it is cheaper and applicable
				if( basic_cost >= transpose_cost && size >= P ) {
					variant = TWO_STEP;
				}
			}

			// preliminaries
			RC ret = data.ensureMemslotAvailable();
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( 2 * data.P );
			}
			if( ret == SUCCESS ) {
				if( variant == ONE_STEP ) {
					ret = data.ensureBufferSize( data.P * bsize );
				} else {
					assert( variant == TWO_STEP );
					ret = data.ensureBufferSize( bsize + data.P * sizeof( IOType ) );
				}
			}
			if( ret == SUCCESS && variant == ONE_STEP ) {
				ret = data.ensureCollectivesCapacity( 1, 0, bsize );
			}
			if( ret != SUCCESS ) { return ret; }


			// register inout
			lpf_memslot_t inout_slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
#ifdef BLAS1_RAW
			lpf_rc = lpf_register_global( data.context, inout, bsize, &inout_slot );
#else
			lpf_rc = lpf_register_global( data.context, internal::getRaw( inout ),
				bsize, &inout_slot );
#endif
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			}

			// implementation check -- the below switch is hardcoded for these two
			// variants:
			if( variant != ONE_STEP && variant != TWO_STEP ) {
					/* LCOV_EXCL_START */
					std::cerr << "Error, internal::allcombine (BSP): unrecognised variant, "
						<< "please submit a bug report\n";
#ifndef NDEBUG
					const bool alp_bsp_implementation_error = false;
					assert( alp_bsp_implementation_error );
#endif
					// abuse LPF fatal error code for propagating panic
					lpf_rc = LPF_ERR_FATAL;
					/* LCOV_EXCL_STOP */
			}

			if( lpf_rc != LPF_SUCCESS ) {
				return checkLPFerror( lpf_rc, "internal::allcombine (intermediate, BSP)" );
			}

			// execute
			IOType * results = data.template getBuffer< IOType >();
			switch( variant ) {

				case ONE_STEP:

					// allgather values
					lpf_rc = lpf_allgather( data.coll, inout_slot, data.slot, bsize, true );
					if( lpf_rc == LPF_SUCCESS ) {
						lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
					}

					// combine results into output vector
#ifdef BLAS1_RAW
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::foldMatrixToVector< descr >(
							inout, results, data.P, size, data.s, op );
#else
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::foldMatrixToVector< descr >(
							internal::getRaw(inout), results, data.P, size, data.s, op );
#endif
					break;

				case TWO_STEP:

					// transpose and allgather

					// step 1: my_chunk*size bytes from each process to my collectives slot
					size_t chunk = ( size + data.P - 1 ) / data.P; // chunk size rounded up
					size_t offset = data.s * chunk;                // start of my chunk
					// my chunk size (corrected for out-of-bounds):
					size_t my_chunk = (offset + chunk) <= size ? chunk : (size - offset);

					// NOTE TODO: this could have been an lpf_gather if that supported offsets
					//            see LPF GitHub issue #19
					for( size_t pid = 0; pid < data.P && lpf_rc == LPF_SUCCESS; pid++ ) {
						if( pid == data.s ) { continue; }
						lpf_rc = lpf_get(
							data.context,
							pid, inout_slot, offset * bytes,
							data.slot, pid * my_chunk * bytes,
							my_chunk * bytes,
							LPF_MSG_DEFAULT
						);
					}
					if( lpf_rc == LPF_SUCCESS ) {
						lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
					}

					// step 2: combine the chunks
					if( lpf_rc == SUCCESS ) {
#ifdef BLAS1_RAW
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( inout + offset,
								results, data.P, my_chunk, data.s, op );
#else
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( internal::getRaw( inout ) + offset,
								results, data.P, my_chunk, data.s, op );
#endif
					}

					// step 3: broadcast local combined chunks
					// NOTE TODO: this could have been an lpf_broadcast if that supported
					//            offsets. See LPF GitHub issue #19
					for( size_t pid = 0; pid < data.P && lpf_rc == LPF_SUCCESS; pid++ ) {
						if( pid == data.s ) { continue; }
						lpf_rc = lpf_put(
							data.context,
							inout_slot, offset * bytes,
							pid, inout_slot, offset * bytes,
							my_chunk * bytes,
							LPF_MSG_DEFAULT
						);
					}
					if( lpf_rc == LPF_SUCCESS ) {
						lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
					}
					break;

			}
			ret = checkLPFerror( lpf_rc, "internal::allcombine (coda, BSP)" );

			// do deregister
			if( inout_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, inout_slot );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::allcombine (BSP), raw variant: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::allcombine (BSP), grb variant: exiting"
				<< std::endl;
 #endif
#endif
			// done
			return ret;
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
		 * -# local work: \f$ N*Operator \f$;
		 * -# transferred bytes: \f$ N \f$;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: transpose, reduce and allgather (N >= P*P)
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ (N/P)*Operator \f$;
		 * -# transferred bytes: \f$ 2(N/P) \f$;
		 * -# BSP cost: \f$ 2(N/P)g + (N/P)*Operator + 2l \f$;
		 * \endparblock
		 *
		 * \parblock
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ P * inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 2(N/\sqrt{P})*Operator \f$;
		 * -# transferred bytes: \f$ 2(N/\sqrt{P}) \f$;
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
			IOType * const inout,
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
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::combine (BSP), raw variant" << std::endl;
 #else
			std::cout << "In internal::combine (BSP), grb variant" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();

#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif
			const size_t bytes = sizeof( IOType );
			const size_t bsize = size * bytes;

			// determine best variant
			enum Variant { ONE_STEP, TREE, TWO_STEP };
			Variant variant = ONE_STEP;
			{
				const double P = data.P;
				const double N = static_cast< double >(bsize);
				const double g = data.getMessageGap( N );
				const double l = data.getLatency( N );
				constexpr double two = 2;

				// one superstep basic approach: pNg + l, useful for small p, N
				const double basic_cost = (P * N * g) + l;
				// two supersteps using transpose and gather: 2Ng + 2l, useful for large P
				// requires N >= P
				const double transpose_cost = (two * N * g) + (two * l);
				// two supersteps using sqrt(p) degree tree: 2sqrt(p)Ng + 2l,
				// useful for large P, does also work for N < p
				const double tree_cost = (two * sqrt( P ) * N * g) + (two * l);

				if( basic_cost >= transpose_cost || basic_cost >= tree_cost ) {
					if( transpose_cost < tree_cost && N >= P ) {
						variant = TWO_STEP;
					} else {
						variant = TREE;
					}
				}
			}

			// preliminaries
			RC ret = SUCCESS;
			if( variant == ONE_STEP || variant == TREE ) {
				ret = data.ensureBufferSize( data.P * bsize );
			} else if( variant == TWO_STEP ) {
				ret = data.ensureBufferSize( bsize + data.P );
			} else {
				/* LCOV_EXCL_START */
				std::cerr << "Error (internal::combine, BSP): unrecognised variant, please "
					<< "submit a bug report\n";
#ifndef NDEBUG
				const bool internal_logic_error = false;
				assert( internal_logic_error );
#endif
				ret = PANIC;
				/* LCOV_EXCL_STOP */
			}
			if( ret == SUCCESS ) {
				if( variant == TREE || variant == TWO_STEP ) {
					ret = data.ensureMemslotAvailable();
				}
			}
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( data.P - 1 );
			}
			if( ret == SUCCESS ) {
				if( variant == ONE_STEP ) {
					ret = data.ensureCollectivesCapacity( 1, 0, bsize );
				}
			}
			if( ret != SUCCESS ) { return ret; }

			// create memslot
			lpf_memslot_t inout_slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
			if( variant == TREE ) {
#ifdef BLAS1_RAW
				lpf_rc = lpf_register_local( data.context, inout, bsize, &inout_slot );
#else
				lpf_rc = lpf_register_local( data.context, internal::getRaw( inout ), bsize,
					&inout_slot );
#endif
			} else if( variant == TWO_STEP ) {
#ifdef BLAS1_RAW
				lpf_rc = lpf_register_global( data.context, inout, bsize, &inout_slot );
#else
				lpf_rc = lpf_register_global( data.context, internal::getRaw( inout ),
					bsize, &inout_slot );
#endif
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
			}
			// note: ONE_STEP variant needs no memory slot registration

			// prelims are done
			if( lpf_rc != LPF_SUCCESS ) {
				return checkLPFerror( lpf_rc, "internal::combine (intermediate, BSP)" );
			}

			// execute
			IOType * __restrict__ const buffer = data.getBuffer< IOType >();
			if( variant == ONE_STEP ) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
				std::cout << "\t internal::combine (BSP), raw: selected one-step variant"
					<< std::endl;
 #else
				std::cout << "\t internal::combine (BSP), grb: selected one-step variant"
					<< std::endl;
 #endif
#endif
				// one-shot variant

				// copy input to buffer
				size_t pos = ( data.s == root ) ? data.s : 0;
#ifdef BLAS1_RAW
				internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy(
					buffer + pos * size, inout, size );
#else
				internal::maybeParallel< _GRB_BSP1D_BACKEND >::memcpy(
					buffer + pos * size, internal::getRaw( inout ), size );
#endif

				// gather together values
				lpf_rc = lpf_gather( data.coll, data.slot, data.slot, bsize, root );
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// fold everything: root only
				if( lpf_rc == LPF_SUCCESS && data.s == root ) {
#ifdef BLAS1_RAW
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::
						foldMatrixToVector< descr >( inout, buffer, data.P, size, data.s, op );
#else
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::
						foldMatrixToVector< descr >( internal::getRaw(inout), buffer, data.P,
							size, data.s, op );
					internal::getCoordinates( inout ).template assignAll< true >();
#endif
				}

				// done

			} else if( variant == TREE ) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
				std::cout << "\t internal::combine (BSP), raw: selected tree variant"
					<< std::endl;
 #else
				std::cout << "\t internal::combine (BSP), grb: selected tree variant"
					<< std::endl;
 #endif
#endif
				// tree variant

				// the (max) interval between each core process
				const size_t hop = sqrt( static_cast< double >(data.P) );
				// the offset from my core process
				const size_t core_offset = DIFF( data.s, root, data.P ) % hop;
				// my core process
				const size_t core_home = DIFF( data.s, core_offset, data.P );
				// am I a core process
				const bool is_core = ( core_offset == 0 );
				// number of processes in my core group
				size_t core_count = hop;
				while( core_count > 1 ) {
					const size_t tmp_proc = data.s + ( core_count - 1 );
					const size_t tmp_core_offset = DIFF( tmp_proc, root, data.P ) % hop;
					const size_t tmp_core_home = DIFF( tmp_proc, tmp_core_offset, data.P );
					if( tmp_core_home == core_home ) {
						break;
					}
					(void) --core_count;
				}

				// step 1: all non-core processes write to their designated core process
				if( !is_core ) {
					lpf_rc = lpf_put(
						data.context,
						inout_slot, 0,
						core_home, data.slot, data.s * bsize,
						bsize,
						LPF_MSG_DEFAULT
					);
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// step 2: all core processes combine their results into their vector
				if( is_core && lpf_rc == LPF_SUCCESS ) {
					for( size_t k = 1; k < core_count; ++k ) {
#ifdef BLAS1_RAW
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( inout,
								buffer + ((data.s + k) % data.P) * size, 1, size, 1, op );
#else
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( internal::getRaw(inout),
								buffer + ((data.s + k) % data.P) * size, 1, size, 1, op );
#endif
					}
				}

				// step 3: non-root processes will write their result to root
				if( is_core && data.s != root && lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_put(
							data.context,
							inout_slot, 0,
							root, data.slot, data.s * bsize,
							bsize,
							LPF_MSG_DEFAULT
						);
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// step 4: root process combines its results from the core processes
				if( data.s == root ) {
					for( size_t k = hop; k < data.P; k += hop ) {
#ifdef BLAS1_RAW
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( inout,
								buffer + ((k + root) % data.P) * size, 1, size, 1, op );
#else
						internal::maybeParallel< _GRB_BSP1D_BACKEND >::
							foldMatrixToVector< descr >( internal::getRaw(inout),
								buffer + ((k + root) % data.P) * size, 1, size, 1, op );
#endif
					}
				}

				// done

			} else if( variant == TWO_STEP ) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
				std::cout << "\t internal::combine (BSP), raw: selected two-step variant"
					<< std::endl;
 #else
				std::cout << "\t internal::combine (BSP), grb: selected two-step variant"
					<< std::endl;
 #endif
#endif
				// transpose and gather

				// step 1: my_chunk*size bytes from each process to my collectives slot
				const size_t chunk = ( size + data.P - 1 ) / data.P;
				const size_t offset = data.s * chunk;
				const size_t my_chunk = ( offset + chunk ) <= size
					? chunk
					: (size - offset);
				for( size_t pid = 0; pid < data.P && lpf_rc == LPF_SUCCESS; pid++ ) {
					if( pid == data.s ) { continue; }
					lpf_rc = lpf_get(
						data.context,
						pid, inout_slot, offset * bytes,
						data.slot, pid * my_chunk * bytes,
						my_chunk * bytes,
						LPF_MSG_DEFAULT
					);
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// step 2: combine the chunks and write to the root process
				if( lpf_rc == LPF_SUCCESS ) {
#ifdef BLAS1_RAW
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::
						foldMatrixToVector< descr >( inout + offset,
							buffer, data.P, my_chunk, data.s, op );
#else
					internal::maybeParallel< _GRB_BSP1D_BACKEND >::
						foldMatrixToVector< descr >( internal::getRaw(inout) + offset,
							buffer, data.P, my_chunk, data.s, op );
#endif
					if( data.s != root ) {
						lpf_rc = lpf_put(
							data.context,
							inout_slot, offset * bytes,
							root, data.slot, offset * bytes,
							my_chunk * bytes,
							LPF_MSG_DEFAULT
						);
					}
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// done

			} else {

				/* LCOV_EXCL_START */
				std::cerr << "Error, internal::combine (BSP): unrecognised variant, "
					<< "please submit a bug report\n";
#ifndef NDEBUG
				const bool alp_bsp_implementation_error = false;
				assert( alp_bsp_implementation_error );
#endif
				// abuse LPF fatal error code for propagating panic
				lpf_rc = LPF_ERR_FATAL;
				/* LCOV_EXCL_STOP */
			}

			// end of LPF section
			ret = checkLPFerror( lpf_rc, "internal::combine (coda, BSP)" );

			// clean up memslots
			if( inout_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, inout_slot );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::combine (BSP), raw variant: exiting" << std::endl;
 #else
			std::cout << "\t internal::combine (BSP), grb variant: exiting" << std::endl;
 #endif
#endif
			// done
			return ret;
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
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ NP \f$;
		 * -# BSP cost: \f$ NPg + l \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two phase
		 * -# Problem size N: \f$ inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ 2N \f$;
		 * -# BSP cost: \f$ 2(Ng + l) \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ inout.size * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$;
		 * -# transferred bytes: \f$ 2\sqrt{P}N \f$;
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
			IOType * const inout,
			const size_t size,
#else
			Vector< IOType, reference, Coords > &inout,
#endif
			const lpf_pid_t root
		) {
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "In internal::broadcast (BSP), raw variant" << std::endl;
 #else
			std::cout << "In internal::broadcast (BSP), grb variant" << std::endl;
 #endif
#endif
			// we need access to BSP context
			internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifndef BLAS1_RAW
			const size_t size = internal::getCoordinates( inout ).size();
#endif
			// dynamic checks
			if( root >= data.P ) {
				return ILLEGAL;
			}
#ifdef BLAS1_RAW
			if( inout == nullptr ) {
				return ILLEGAL;
			}
#endif

			// check trivial dispatch
			if( size == 0 ) {
				return SUCCESS;
			}
			if( data.P == 1 ) {
				return SUCCESS;
			}

			// preliminaries
			const size_t bsize = size * sizeof( IOType );
			RC ret = data.ensureCollectivesCapacity( 1, 0, bsize );
			if( ret == SUCCESS ) {
				ret = data.ensureMemslotAvailable();
			}
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( std::max( data.P + 1, 2 * data.P - 3 ) );
			}
			if( ret != SUCCESS ) { return ret; }

			// create and activate memslot
			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;
#ifndef BLAS1_RAW
			lpf_rc = lpf_register_global( data.context, internal::getRaw(inout), bsize,
				&slot );
#else
			lpf_rc = lpf_register_global(
					data.context,
					const_cast< typename std::remove_const< IOType >::type * >(inout),
					bsize, &slot
				);
#endif
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			}

			// request and wait for broadcast
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_broadcast( data.coll, slot, slot, bsize, root );
			}
			if( lpf_rc == LPF_SUCCESS ) {
				lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			}

			// end of LPF section
			ret = checkLPFerror( lpf_rc, "internal::broadcast (BSP)" );

			// destroy memslot
			if( slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
				(void) lpf_deregister( data.context, slot );
			}
#ifdef _DEBUG
 #ifdef BLAS1_RAW
			std::cout << "\t internal::broadcast (BSP), raw variant: exiting"
				<< std::endl;
 #else
			std::cout << "\t internal::broadcast (BSP), grb variant: exiting"
				<< std::endl;
 #endif
#endif
			// done
			return ret;
		}

	} // namespace internal

} // namespace grb

