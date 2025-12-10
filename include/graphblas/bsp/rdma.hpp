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
	namespace rdma {

		/**
		 * Deregisters a local buffer for RDMA
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename IOType >
		RC deregister( const lpf_memslot_t memslot ) {
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();
			lpf_err_t lpf_rc = LPF_SUCCESS;

			lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, &memslot );

			// TODO: delete registered memory from map
			const auto it = data.global_memslots.find( memslot );
			assert( it != data.global_memslots.end() );
			data.registered_slots.erase( data.registered_slots.find( it->second ) );
			data.global_memslots.erase( it );

			data.signalMemslotReleased( 1 );
			
			if( lpf_rc == LPF_SUCCESS ) {
				return grb::SUCCESS;
			} else {
				return grb::PANIC;
			}
		}

		/**
		 * Registers a global buffer for RDMA
		 *
		 * This is a collective operation!
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename IOType >
		RC register_global( IOType* buf, const size_t size, lpf_memslot_t &memslot ) {
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();
			lpf_err_t lpf_rc = LPF_SUCCESS;
			
			data.ensureMemslotAvailable( 1 );
			data.signalMemslotTaken();

			assert( data.registered_slots.find( buf ) == data.registered_slots.end() );

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

		template< typename IOType >
		inline RC register_global( IOType &buf, lpf_memslot_t &memslot ) {
			return register_global( &buf, sizeof(IOType), memslot );
		}

		/**
		 * Reads a message from another process' registered memory
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename IOType >
		RC get( IOType* msg, const size_t size, const size_t &src_pid, const lpf_memslot_t &src_memslot ) {
			const auto lpf_attr = LPF_MSG_DEFAULT;
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();
			lpf_memslot_t dst_memslot; 
			lpf_err_t lpf_rc = LPF_SUCCESS;

			data.ensureMemslotAvailable( 1 );

			lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, msg, size, &dst_memslot );

			lpf_rc = lpf_rc ? lpf_rc : lpf_get( data.context, src_pid, src_memslot , 0, dst_memslot, 0, size, lpf_attr  );

			lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, dst_memslot );

			if( lpf_rc == LPF_SUCCESS ) {
				return grb::SUCCESS;
			} else {
				return grb::PANIC;
			}
		}
		template< typename IOType >
		inline RC get( IOType &msg, const size_t src_pid, const lpf_memslot_t &src_memslot ) {
			return get( &msg, sizeof(IOType), src_pid, src_memslot );
		}

		/**
		 * Writes a message to another process' registered memory
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename IOType >
		RC put( IOType* msg, const size_t &size, const size_t dst_pid, const lpf_memslot_t &dst_memslot ) {
			const auto lpf_attr = LPF_MSG_DEFAULT;
			internal::BSP1D_Data & data = internal::grb_BSP1D.load();
			lpf_err_t lpf_rc = LPF_SUCCESS;
			lpf_memslot_t src_memslot;

			data.ensureMemslotAvailable( 1 );
			assert( data.global_memslots.find( dst_memslot ) != data.global_memslots.end() );

			lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, msg, size, &src_memslot );

			lpf_rc = lpf_rc ? lpf_rc : lpf_put( data.context, src_memslot, 0, dst_pid, dst_memslot, 0, size, lpf_attr  );

			lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, src_memslot );

			if( lpf_rc == LPF_SUCCESS ) {
				return grb::SUCCESS;
			} else {
				return grb::PANIC;
			}
		}

		template< typename IOType >
		inline RC put( IOType&msg, const size_t dst_pid, const lpf_memslot_t&dst_memslot ) {
			return put( &msg, sizeof(IOType), dst_pid, dst_memslot );
		}


	}; // end class ``rdma'' generic LPF implementation

} // namespace grb

#endif // end _H_GRB_LPF_RDMA
