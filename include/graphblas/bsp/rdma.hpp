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
	template<>
	class rdma< GENERIC_BSP > {
		private:

			/*
			 * Registers a global buffer for RDMA
			 *
			 * \warning This is a collective operation, therefore it must be called by all the processes!
			 *
			 * @param[in] buf	Pointer to the start of the buffer of memory to be registered.
			 * @param[in] size	Size of the buffer in bytes.
			 *
			 * @return grb::SUCCESS When registration is completed successfully
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			static grb::RC register_global( const void *buf, const size_t size ) {
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_err_t lpf_rc = LPF_SUCCESS;
				grb::RC rc = grb::SUCCESS;
				lpf_memslot_t memslot = LPF_INVALID_MEMSLOT;

				const bool buffer_already_registered = (data.registered_slots.find( buf ) != data.registered_slots.end()); 
				if( buffer_already_registered ){
#ifdef _DEBUG
					std::cerr << "Buffer already registered. Nothing to do." << std::endl;
#endif
					return grb::SUCCESS;
				}

				rc = rc ? rc : data.ensureMemslotAvailable( 1 );

				lpf_rc = lpf_rc ? lpf_rc : lpf_register_global(
					data.context, const_cast< void* >( buf ),
					size, &memslot
				);
				lpf_rc = lpf_rc ? lpf_rc : lpf_sync( data.context, LPF_SYNC_DEFAULT );

				data.signalMemslotTaken();
				data.registered_slots.insert({ buf, grb::internal::registered_slot(buf, memslot, size, true ) });
				data.global_memslots.insert({ memslot, buf });

				if( lpf_rc == LPF_SUCCESS ) {
					return rc;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Deregisters a buffer for RDMA. If the buffer is global, this is a collective
			 * function
			 *
			 * \warning This is a collective operation, therefore it must be called by all the processes!
			 * @param[in] buf	Pointer to the start of the buffer of memory to be deregistered.
			 *
			 * @return grb::SUCCESS When deregistration is completed successfully
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			static grb::RC deregister( const void *buf ) {
#ifdef _DEBUG
				std::cout << "deregister: memslot " << memslot << std::endl;
#endif
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();
				lpf_err_t lpf_rc = LPF_SUCCESS;
				lpf_memslot_t memslot = LPF_INVALID_MEMSLOT;

				const auto it0 = data.registered_slots.find( buf );
				assert( it0 != data.registered_slots.end() );

				memslot = it0->second.slot;
				data.registered_slots.erase( it0 );

				const auto it1 = data.global_memslots.find( memslot );
				assert( it1 != data.global_memslots.end() );
				data.global_memslots.erase( it1 );

				lpf_rc = lpf_rc ? lpf_rc : lpf_deregister( data.context, memslot );

				data.signalMemslotReleased( 1 );

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Reads a message from another process' registered memory
			 *
			 * Before calling this function, the user is responsible for ensuring that
			 * enough registers are free, using the function localRegisterSize. Each call
			 * to function will use one register if the source pointer has not been
			 * already registered (locally or globally), and zero otherwise.
			 *
			 * @param[in] src_pid	PID of process whose memory to access.
			 * @param[in] src		Local pointer to the buffer whose associated pointer in the source process where to read.
			 * @param[out] dst		Pointer to the begin of the output buffer.
			 * @param[in] size		Number of bytes to copy.
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			static grb::RC get( const size_t &src_pid, const void *src, void *dst, const size_t size ) {
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
					return grb::ILLEGAL;
				}

				// check trivial dispatch
				if( size == 0 ) {
					return grb::SUCCESS;
				}

				{
					const auto it = data.registered_slots.find( src );
					assert( it != data.registered_slots.end() );
					assert( it->second.size >= size );
					assert( it->second.global );
					src_memslot = it->second.slot;
				}

				const auto it = data.registered_slots.find( dst );
				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, const_cast< void* >( dst ), size, &dst_memslot );
					data.signalMemslotTaken();
					data.registered_slots.insert({dst, grb::internal::registered_slot( dst, dst_memslot, size, false )});
				} else {
					assert( it->second.buf == dst );
					assert( it->second.size >= size );
					dst_memslot = it->second.slot;
				}

				lpf_rc = lpf_rc ? lpf_rc : lpf_get( data.context, src_pid, src_memslot , 0, dst_memslot, 0, size, lpf_attr );
				data.get_requests.emplace_back( src_pid, src_memslot, 0, src, size );

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

			/**
			 * Writes a message to another process' registered memory.
			 *
			 * Before calling this function, the user is responsible for ensuring that
			 * enough registers are free, using the function localRegisterSize. Each call
			 * to function will use one register if the destination pointer has not been
			 * already registered (locally or globally), and zero otherwise.
			 *
			 * @param[in] src		Pointer to the begin of the input buffer.
			 * @param[in] dst_pid	PID of process whose memory to access.
			 * @param[out] dst		Local pointer to the buffer whose associated pointer in the source process where to write.
			 * @param[in] size		Number of bytes to copy.
			 *
			 * @return grb::SUCCESS When all queued communication is executed succesfully.
			 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
			 *                      returned, the library enters an undefined state.
			 */
			static grb::RC put( const void *src, const size_t dst_pid, void *dst, const size_t &size ) {
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

				// rc = rc ? rc : data.ensureMemslotAvailable( 1 ); // this function calls lpf_sync
				{
					const auto it = data.registered_slots.find( dst );
					assert( it != data.registered_slots.end() );
					assert( it->second.buf == dst );
					assert( it->second.size >= size );
					assert( it->second.global );
					dst_memslot = it->second.slot;
				}

				const auto it = data.registered_slots.find( src );
				if( it == data.registered_slots.end() ){
					lpf_rc = lpf_rc ? lpf_rc : lpf_register_local( data.context, const_cast< void* >( src ), size, &src_memslot );
					data.signalMemslotTaken();
					data.registered_slots.insert({src, grb::internal::registered_slot( src, src_memslot, size, false )});
				} else {
					assert( it->second.buf == src );
					assert( it->second.size >= size ); // is there enough space?
					src_memslot = it->second.slot;
				}

				lpf_rc = lpf_rc ? lpf_rc : lpf_put( data.context, src_memslot, 0, dst_pid, dst_memslot, 0, size, lpf_attr  );
				data.put_requests.emplace_back( src, dst_pid, dst_memslot, 0, size );

				if( lpf_rc == LPF_SUCCESS ) {
					return grb::SUCCESS;
				} else {
					return grb::PANIC;
				}
			}

		public:

		/*
		 * Registers a global buffer for RDMA on a POD variable.
		 *
		 * \warning This is a collective operation, therefore it must be called by all the processes!
		 *
		 * @param[in] buf	Scalar to be registered
		 *
		 * @return grb::SUCCESS When registration is completed successfully
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename T >
		static inline grb::RC register_global( const T &buf ) {
			return register_global( reinterpret_cast< const void* >( &buf ), sizeof(T) );
		}

		/*
		 * Registers a global buffer for RDMA
		 *
		 * \warning This is a collective operation, therefore it must be called by all the processes!
		 *
		 * @param[in] buf	Pointer to the start of the buffer of memory to be reserved.
		 * @param[in] size	Size of the buffer in bytes.
		 *
		 * @return grb::SUCCESS When registration is completed successfully
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template<
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static inline grb::RC register_global( const grb::Vector< T, backend, Coords > &buf ) {
			const size_t size = grb::internal::getCoordinates( buf ).size();
			const size_t bsize = size * sizeof( T );
			const void *raw_ptr = reinterpret_cast< const void* >( grb::internal::getRaw( buf ) );

			return register_global( raw_ptr, bsize );
		}

		/*
		 * Deregisters a global buffer on a POD variable.
		 *
		 * \warning This is a collective operation, therefore it must be called by all the processes!
		 *
		 * @param[in] buf	Scalar to be deregistered
		 *
		 * @return grb::SUCCESS When registration is completed successfully
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template< typename T >
		static inline grb::RC deregister( const T &buf ) {
			return deregister( reinterpret_cast< const void* >( &buf ) );
		}

		/*
		 * Deregisters a global buffer.
		 *
		 * \warning This is a collective operation, therefore it must be called by all the processes!
		 *
		 * @param[in] buf	Pointer to the start of the buffer of memory to be deregistered.
		 *
		 * @return grb::SUCCESS When registration is completed successfully
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		template<
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static inline grb::RC deregister( const grb::Vector< T, backend, Coords > &buf ) {
			const void *raw_ptr = reinterpret_cast< const void* >( grb::internal::getRaw( buf ) );

			return deregister( raw_ptr );
		}

		/*
		 * Reserve space for size additional registers. This function should be used
		 * before calling put and/or get. These RDMA functions automatically register
		 * local unregistered buffers, so need space in LPF register to do so.
		 * One register is needed for each put/get between successive a syncs.
		 *
		 * \warning This is a collective operation. A sync could be called internally, therefore it must be called by all the processes!
		 *
		 * @param[in] size	The number of registers to reserve space for.
		 *
		 * @return grb::SUCCESS	The register space was successfully ensured.
		 * @return PANIC   Could not ensure a large enough buffer space. The state
		 *                 of the library has become undefined.
		 */
		static inline grb::RC localRegisterSize( const size_t size ) {
				grb::internal::BSP1D_Data & data = grb::internal::grb_BSP1D.load();

				grb::RC rc = data.ensureMemslotAvailable( size );
				return rc;
		}

		/*
		 * Copy a variable from a remote process.
		 * Source variable must have been globally registered.
		 *
		 * See private get function for more details.
		 *
		 * @param[in] src_pid	Source Process ID.
		 * @param[in] src	Object associated with the remote object to read from.
		 * @param[out] dst	Destination object to write to.
		 *
		 * @return grb::SUCCESS	The register space was successfully ensured.
		 * @return PANIC   Could not ensure a large enough buffer space. The state
		 *                 of the library has become undefined.
		 */
		template< typename T >
		static inline grb::RC get( const size_t src_pid, const T &src, T &dst ) {

			const void *src_ptr = reinterpret_cast< const void* >( &src );
			void *dst_ptr = reinterpret_cast< void* >( &dst );
			constexpr size_t size = sizeof(T);

			return get( src_pid, src_ptr, dst_ptr, size );
		}

		/*
		 *
		 * Copy a grb::Vector from a remote process.
		 * Source grb::Vector must have been globally registered.
		 *
		 * See private get function for more details.
		 *
		 * @param[in] src_pid	Source Process ID.
		 * @param[in] src	Object associated with the remote Vector to read from.
		 * @param[out] dst	Destination Vector to write to.
		 *
		 * @return grb::SUCCESS	The register space was successfully ensured.
		 * @return PANIC   Could not ensure a large enough buffer space. The state
		 *                 of the library has become undefined.
		 */
		template<
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static inline grb::RC get( const size_t src_pid, const grb::Vector< T, backend, Coords > &src, grb::Vector< T, backend, Coords > &dst ) {

 			// we only support grb::reference for now
			static_assert( grb::reference ==  backend );

			const size_t size = grb::internal::getCoordinates( dst ).size();
			const size_t bsize = size * sizeof( T );
			const void *src_ptr = reinterpret_cast< const void* >( grb::internal::getRaw( src ) );
			void *dst_ptr = reinterpret_cast< void* >( grb::internal::getRaw( dst ) );

			return get( src_pid, src_ptr, dst_ptr, bsize );
		}

		/*
		 * Write a scalar variable to another process.
		 *
		 * Vectors must be of the same size (in the respective processes)!
		 *
		 * See private put function for more details.
		 *
		 * @param[in] src	Variable to read from.
		 * @param[in] dst_pid	Destination Process ID.
		 * @param[out] dst	Variable associated with the scalar in the destination process to write to.
		 *
		 * @return grb::SUCCESS	The register space was successfully ensured.
		 * @return PANIC   Could not ensure a large enough buffer space. The state
		 *                 of the library has become undefined.
		 */
		template< typename T >
		static inline grb::RC put( const T &src, const size_t dst_pid, T &dst ) {

			const void *src_ptr = reinterpret_cast< const void* >( &src );
			void *dst_ptr = reinterpret_cast< void* >( &dst );
			constexpr size_t size = sizeof(T);

			return put( src_ptr, dst_pid, dst_ptr, size );
		}

		/*
		 * Write a grb::Vector to another process.
		 *
		 * @param[in] src	grb::Vector to read from.
		 * @param[in] dst_pid	Destination Process ID.
		 * @param[out] dst	grb::Vector whose associated object in the destination process to write to.
		 *
		 * @return grb::SUCCESS	The register space was successfully ensured.
		 * @return PANIC   Could not ensure a large enough buffer space. The state
		 *                 of the library has become undefined.
		 */
		template<
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static inline grb::RC put( const grb::Vector< T, backend, Coords > &src, const size_t dst_pid, grb::Vector< T, backend, Coords > &dst) {
 			// we only support grb::reference for now
			static_assert( grb::reference ==  backend );

			const size_t size = grb::internal::getCoordinates( src ).size();
			const size_t bsize = size * sizeof( T );
			const void *src_ptr = reinterpret_cast< const void* >( grb::internal::getRaw( src ) );
			void *dst_ptr = reinterpret_cast< void* >( grb::internal::getRaw( dst ) );

			return put( src_ptr, dst_pid, dst_ptr, bsize );
		}
	}; // end class ``rdma'' generic LPF implementation

} // namespace grb

#endif // end _H_GRB_LPF_RDMA
