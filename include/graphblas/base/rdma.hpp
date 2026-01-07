
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
 * Exposes facilities for Remote Direct Memory Access
 *
 * @author G. Gaio
 * @date 16th of December, 2025
 */

#ifndef _H_GRB_BASE_RDMA
#define _H_GRB_BASE_RDMA

#include <cstddef> //size_t

#include <stdint.h> // SIZE_MAX

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>

#include "config.hpp"


namespace grb {

	/**
	 * For backends that support multiple user processes this class defines some
	 * basic primitives to support RDMA programming.
	 *
	 * All backends must implement this interface, including backends that do not
	 * support multiple user processes. The interface herein defined hence ensures
	 * to allow for trivial implementations for single user process backends.
	 */
	template< Backend implementation >
	class rdma {
		public:
		template< typename T >
		static inline grb::RC register_global( const T &buf );

		template< typename T >
		static inline grb::RC register_global( const grb::Vector< T, grb::reference > &buf );

		template< typename T >
		static inline grb::RC deregister( const T &buf );

		template< typename T >
		static inline grb::RC deregister( const grb::Vector< T, grb::reference > &buf );

		static inline grb::RC localRegisterSize( const size_t size );

		template< typename T >
		static inline grb::RC get( const size_t src_pid, T &src, T &dst );

		template<
			grb::Descriptor descr,
			grb::Backend backend,
			typename T,
			typename Coords
			>
		static inline grb::RC get( const size_t src_pid, const grb::Vector< T, backend, Coords > &src, grb::Vector< T, backend, Coords > &dst );

		template< typename T >
		static inline grb::RC put( const T &src, const size_t dst_pid, T &dst );

		template<
			grb::Descriptor descr,
			grb::Backend backend,
			typename T,
			typename Coords
			>
		static inline grb::RC put( const grb::Vector< T, backend, Coords > &src, const size_t dst_pid, grb::Vector< T, backend, Coords > &dst );
	}; // end class ``rdma''

} // namespace grb

#endif // end _H_GRB_BASE_SPMD

