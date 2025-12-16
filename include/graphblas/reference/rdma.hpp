
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
 * @author A. N. Yzelman
 * @date 28th of April, 2017
 */

#if ! defined _H_GRB_REFERENCE_RDMA || defined _H_GRB_REFERENCE_OMP_RDMA
#define _H_GRB_REFERENCE_RDMA

#include <cstddef> //size_t

#include <graphblas/base/rdma.hpp>

namespace grb {

	/** No implementation notes. */
	template<>
	class rdma< reference > {
	public:
		template< typename T >
		static grb::RC register_global( T &buf) {
			return grb::SUCCESS;
		}

		template< typename T >
		static grb::RC register_global( grb::Vector< T, grb::reference > &buf ) {
			return grb::SUCCESS;
		}

		template< typename T >
		static grb::RC get( const size_t src_pid, T &src, T &dst ) {
			assert( src_pid == 0 );
			dst = src;
			return grb::SUCCESS;
		}

		template<
			grb::Descriptor descr = descriptors::no_operation,
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static grb::RC get( const size_t src_pid, const grb::Vector< T, backend, Coords > &src, grb::Vector< T, backend, Coords > &dst ) {
			assert( src_pid == 0 );
			return grb::set< descr >( dst, src );
		}

		template< typename T >
		static grb::RC put( const T &src, const size_t dst_pid, T &dst ) {
			assert( dst_pid == 0 );
			dst = src;
			return grb::SUCCESS;
		}

		template<
			grb::Descriptor descr = descriptors::no_operation,
			grb::Backend backend = grb::reference,
			typename T,
			typename Coords
			>
		static grb::RC put( const grb::Vector< T, backend, Coords > &src, const size_t dst_pid, grb::Vector< T, backend, Coords > &dst) {
			assert( dst_pid == 0 );
			return grb::set< descr >( dst, src );
		}
	}; // end class ``rdma'' reference implementation

} // namespace grb

// parse again for reference_omp backend
#ifdef _GRB_WITH_OMP
#ifndef _H_GRB_REFERENCE_OMP_RDMA
#define _H_GRB_REFERENCE_OMP_RDMA
#define reference reference_omp
#include "graphblas/reference/rdma.hpp"
#undef reference
#undef _H_GRB_REFERENCE_OMP_RDMA
#endif
#endif

#endif // end _H_GRB_REFERENCE_RDMA
