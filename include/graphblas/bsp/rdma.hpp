/*
 *   Copyright 2025 Huawei Technologies Co., Ltd.
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
 * @author G. Gaio
 * @date December, 2025
 */

#ifndef _H_GRB_LPF_RDMA
#define _H_GRB_LPF_RDMA

#include <map>
#include <cstddef> //size_t

#include <lpf/core.h>

#include <graphblas/base/spmd.hpp>
#include <graphblas/bsp1d/init.hpp> // get_request, put_request


namespace grb {

	/** Superclass implementation for all LPF-backed implementations. */
	// template<>
	// class rdma< GENERIC_BSP > {
	namespace rdma {
		namespace internal {

			/**
			 * Registers a global buffer for RDMA
			 *
			 * This is a collective operation!
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			template< typename T >
			RC register_global( T* buf, const size_t size ) {
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_err_t lpf_rc = LPF_SUCCESS;
				lpf_memslot_t memslot = LPF_INVALID_MEMSLOT;

				data.ensureMemslotAvailable( 1 );
				data.signalMemslotTaken();

				assert( data.registered_slots.find( static_cast< const void* >( buf ) ) == data.registered_slots.end() );

				lpf_rc = lpf_rc ? lpf_rc : lpf_register_global(
					data.context,
					static_cast< void* >( buf ),
					size, &memslot
				);
				lpf_rc = lpf_rc ? lpf_rc : lpf_sync( data.context, LPF_SYNC_DEFAULT );

				data.registered_slots.insert({ static_cast< const void* >( buf ),
						std::make_pair( size, memslot ) });
				data.global_memslots.insert({ memslot, static_cast< const void* >( buf ) });

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Deregisters a local buffer for RDMA
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			template< typename T >
			RC deregister( const T &buf ) {
#ifdef _DEBUG
				std::cout << "deregister: memslot " << memslot << std::endl;
#endif
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_err_t lpf_rc = LPF_SUCCESS;
				lpf_memslot_t memslot = LPF_INVALID_MEMSLOT;


				const auto it0 = data.registered_slots.find( buf )
				assert( it0 != data.registered_slots.end() );

				memslot = it0->second->second;
				data.registered_slots.erase( it0 );

				// TODO: delete registered memory from map
				const auto it1 = data.global_memslots.find( memslot );
				assert( it1 != data.global_memslots.end() );
				data.global_memslots.erase( it1 );

				lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, &memslot );

				data.signalMemslotReleased( 1 );

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Writes a message to another process' registered memory
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			template< typename T1, typename T2 >
			RC put( T1* src, const size_t dst_pid, T2* dst, const size_t &size ) {
#ifdef _DEBUG
				std::cout << "rdma::put( " << src << ", " << size << ", " << dst_pid  << ", " << dst << ") called" << std::endl;
#endif

				const auto lpf_attr = LPF_MSG_DEFAULT;
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_err_t lpf_rc = LPF_SUCCESS;
				lpf_memslot_t src_memslot = LPF_INVALID_MEMSLOT;
				lpf_memslot_t dst_memslot = LPF_INVALID_MEMSLOT;

				// dynamic checks
				if( dst_pid >= data.P ) {
					return grb::ILLEGAL;
				}

				// check trivial dispatch
				if( size == 0 ) {
					return grb::SUCCESS;
				}
				if( data.P == 1 ) {
					return grb::SUCCESS;
				}

				data.ensureMemslotAvailable( 1 );
				{
					const auto it = data.registered_slots.find( dst );
					assert( it != data.registered_slots.end() );
					assert( it->second.first >= size );
					dst_memslot = it->second.second;
				}

				const auto it = data.registered_slots.find( src );
				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, src, size, &src_memslot );
				} else {
					// there must be a better check...
					assert( it->second.first >= size ); // is there enough space?
					src_memslot = it->second.second;
				}

				lpf_rc = lpf_rc ? lpf_rc : lpf_put( data.context, src_memslot, 0, dst_pid, dst_memslot, 0, size, lpf_attr  );
				data.put_requests.emplace_back( src, dst_pid, dst_memslot, 0, size );

				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, src_memslot );
				}

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Reads a message from another process' registered memory
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			template< typename T >
			RC get( const size_t &src_pid, T* src, T* dst, const size_t size ) {
#ifdef _DEBUG
				std::cout << "rdma::get( " << src << ", " << size << ", " << src_pid  << ", " << src_memslot << ") called" << std::endl;
#endif
				const auto lpf_attr = LPF_MSG_DEFAULT;
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_memslot_t src_memslot = LPF_INVALID_MEMSLOT;
				lpf_memslot_t dst_memslot = LPF_INVALID_MEMSLOT;
				lpf_err_t lpf_rc = LPF_SUCCESS;

				// dynamic checks
				if( src_pid >= data.P ) {
					return ILLEGAL;
				}

				// check trivial dispatch
				if( size == 0 ) {
					return SUCCESS;
				}
				if( data.P == 1 ) {
					return SUCCESS;
				}

				data.ensureMemslotAvailable( 1 );
				{
					const auto it = data.registered_slots.find( src );
					assert( it != data.registered_slots.end() );
					assert( it->second.first >= size );
					src_memslot = it->second.second;
				}

				const auto it = data.registered_slots.find( dst );
				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, dst, size, &dst_memslot );
				} else {
					// there must be a better check...
					assert( it->second.first >= size );
					dst_memslot = it->second.second;
				}

				lpf_rc = lpf_rc ? lpf_rc : lpf_get( data.context, src_pid, src_memslot , 0, dst_memslot, 0, size, lpf_attr );
				data.get_requests.emplace_back( src_pid, src_memslot, 0, src, size );

				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, dst_memslot );
				}

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

		} // namespace internal

		/*
		 * RDMA wrappers for internal functions
		 */
		template< typename T >
		inline RC register_global( T &buf) {
			return internal::register_global( &buf, sizeof(T) );
		}


		template< typename T >
		inline RC register_global( grb::Vector< T, grb::reference > &inout ) {
			const size_t size = grb::internal::getCoordinates( inout ).size();
			const size_t bsize = size * sizeof( T );
			T* raw_ptr = grb::internal::getRaw( inout );

			lpf_memslot_t slot = LPF_INVALID_MEMSLOT;
			lpf_err_t lpf_rc = LPF_SUCCESS;

			return internal::register_global( raw_ptr, bsize );
		}


		template< typename T >
		inline RC get( const size_t src_pid, T &src, T &dst ) {
			return internal::get(  src_pid, &src, &dst, sizeof(T) );
		}

		template<
			grb::Descriptor descr = descriptors::no_operation,
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		inline RC get( const size_t src_pid, const grb::Vector< T, backend, Coords > &src, grb::Vector< T, backend, Coords > &dst ) {

 			// we only support grb::reference for now
			static_assert( grb::reference ==  backend );

			const size_t size = grb::internal::getCoordinates( dst ).size();
			const size_t bsize = size * sizeof( T );

			return internal::get( src_pid, grb::internal::getRaw( src ), grb::internal::getRaw( dst ), bsize );
		}

		template< typename T >
		inline RC put( T &src, const size_t dst_pid, T &dst ) {
			return internal::put( &src, dst_pid, &dst, sizeof(T) );
		}

		template<
			grb::Descriptor descr = descriptors::no_operation,
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		inline RC put( const grb::Vector< T, backend, Coords > &src, const size_t dst_pid, grb::Vector< T, backend, Coords > &dst) {
 			// we only support grb::reference for now
			static_assert( grb::reference ==  backend );
			const size_t size = grb::internal::getCoordinates( src ).size();
			const size_t bsize = size * sizeof( T );

			return internal::put( grb::internal::getRaw( src ), dst_pid, grb::internal::getRaw( dst ), bsize );
		}

	}; // end class ``rdma'' generic LPF implementation

} // namespace grb

#endif // end _H_GRB_LPF_RDMA
