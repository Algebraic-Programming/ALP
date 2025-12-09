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
	class rdma {
		private:
		std::map< const void* , std::pair< size_t, const lpf_memslot_t > > registered_slots;
		std::map< const lpf_memslot_t , const void* > memslots;

		size_t memory_register_size = 0;

		/*
		 * This is a collective function. It must be called by all the processes with
		 * the same parameters!!
		 */
		RC resize_memory_register( const size_t new_register_size ) {
			if( new_register_size < memory_register_size ) return grb::SUCCESS;

			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_err_t lpf_rc = LPF_SUCCESS;

			lpf_rc = lpf_rc ? lpf_rc : lpf_resize_memory_register( data.context, new_register_size );
			lpf_rc = lpf_rc ? lpf_rc : lpf_sync( data.context, LPF_SYNC_DEFAULT );
			if( lpf_rc == LPF_SUCCESS ) {
				memory_register_size = new_register_size;
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
		template< typename IOType >
		RC put( IOType* msg, const size_t &size, const size_t dst_pid, const lpf_memslot_t &dst_memslot ) {
			const auto lpf_attr = LPF_MSG_DEFAULT;
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_err_t lpf_rc = LPF_SUCCESS;
			lpf_memslot_t src_memslot; 

			assert( memslots.find( dst_memslot ) != memslots.end() );

			// TODO: checks for local registration! Already registered? Register full?
			lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, msg, size, &src_memslot );

			lpf_rc = lpf_rc ? lpf_rc : lpf_put( data.context, src_memslot, 0, dst_pid, dst_memslot, 0, size, lpf_attr  );

			lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, src_memslot );

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
		template< typename IOType >
		RC deregister( const lpf_memslot_t memslot ) {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_err_t lpf_rc = LPF_SUCCESS;

			assert( registered_slots.size() < memory_register_size );

			lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, &memslot );

			// TODO: delete registered memory from map
			auto it = memslots.find( memslot );
			registered_slots.erase( registered_slots.find( it->second ) );
			memslots.erase( it );
			
			if( lpf_rc == LPF_SUCCESS ) {
				return grb::SUCCESS;
			} else {
				return grb::PANIC;
			}
		}

		public:

		/*
		 * This is a collective function: it must be called by all the processes with
		 * the same parameters!!
		 */
		rdma( const size_t register_size = 2 ) {

			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_rc = lpf_rc ? lpf_rc : lpf_resize_message_queue( data.context, 42 );

			registered_slots.clear();
			memslots.clear();
			memory_register_size = 0;

			grb::rdma::resize_memory_register( register_size );
		}

		/*
		 * This is a collective function: it must be called by all the processes with
		 * the same parameters!!
		 */
		~rdma() {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_err_t lpf_rc = LPF_SUCCESS;

#ifdef _DEBUG
			std::cerr << "RDMA finalize called." << std::endl;
			std::cerr << "\t" << registered_slots.size() << std::endl;
#endif

			for(auto it = registered_slots.begin(); lpf_rc == LPF_SUCCESS && it != registered_slots.end() ; it++){
				const lpf_memslot_t memslot = ((it->second).second);
				const auto memslot_it = memslots.find( memslot );
				assert( memslot_it != memslots.end() );
				memslots.erase( memslot_it );
				lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, memslot );
			}
			lpf_rc = lpf_rc ? lpf_rc : lpf_sync( data.context, LPF_SYNC_DEFAULT );

			if( lpf_rc == LPF_SUCCESS ) {
				registered_slots.clear();
				memory_register_size = 0;
			}
		}
		/**
		 * Registers a buffer for RDMA
		 *
		 * This is a collective operation!
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename IOType >
		RC register_global( IOType* buf, const size_t size, lpf_memslot_t &memslot ) {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_err_t lpf_rc = LPF_SUCCESS;
			
			while( memory_register_size <= registered_slots.size() + 1 ){
				grb::rdma::resize_memory_register( 2*memory_register_size );
			}

			assert( registered_slots.find( buf ) == registered_slots.end() );

			lpf_rc = lpf_rc ? lpf_rc : lpf_register_global(
				data.context,
				static_cast< void* >( buf ),
				size, &memslot
			);
			lpf_rc = lpf_rc ? lpf_rc : lpf_sync( data.context, LPF_SYNC_DEFAULT );

			registered_slots.insert({ static_cast< const void* >( buf ),
					std::make_pair( size, memslot ) });
			memslots.insert({ memslot, static_cast< const void* >( buf ) });

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
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			lpf_memslot_t dst_memslot; 
			lpf_err_t lpf_rc = LPF_SUCCESS;

			// TODO: checks for local registration! Already registered? Register full?
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

		template< typename IOType >
		inline RC put( IOType&msg, const size_t dst_pid, const lpf_memslot_t&dst_memslot ) {
			return put( &msg, sizeof(IOType), dst_pid, dst_memslot );
		}


	}; // end class ``rdma'' generic LPF implementation

} // namespace grb

#endif // end _H_GRB_LPF_RDMA
