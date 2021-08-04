
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
 * Contains the BSP1D implementation for the PinnedVector class.
 *
 * @author A. N. Yzelman
 * @date 20180725
 */

#ifndef _H_GRB_BSP1D_PINNEDVECTOR
#define _H_GRB_BSP1D_PINNEDVECTOR

#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/base/vector.hpp>
#include <graphblas/reference/coordinates.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "distribution.hpp"
#include "init.hpp"

namespace grb {

	/** No implementation notes. */
	template< typename IOType >
	class PinnedVector< IOType, BSP1D > {

	private:
		/**
		 * Tell the system to delete \a _buffered_values only when we had its last
		 * reference.
		 */
		utils::AutoDeleter< IOType > _raw_deleter;

		/**
		 * Tell the system to delete \a _buffered_mask only when we had its last
		 * reference.
		 */
		utils::AutoDeleter< char > _assigned_deleter;

		/** A buffer of the local vector. */
		IOType * _buffered_values;

		/** A buffer of the sparsity pattern of \a _buffered_values. */
		internal::Coordinates< _GRB_BSP1D_BACKEND > _buffered_mask;

		/** The user process ID of the information in \a _buffered_values. */
		size_t _s;

		/**
		 * The total number of user process IDs in the context of
		 * \a _buffered_values.
		 */
		size_t _P;

	public:
		/** No implementation notes. */
		PinnedVector() : _buffered_values( NULL ), _s( 0 ), _P( 0 ) {}

		/** No implementation notes. */
		template< typename Coords >
		PinnedVector( const Vector< IOType, BSP1D, Coords > & x, const IOMode mode ) :
			_raw_deleter( x._raw_deleter ), _assigned_deleter( x._assigned_deleter ), _buffered_values( mode == PARALLEL ? x._raw + x._offset : x._raw ),
			_buffered_mask( mode == PARALLEL ? x._local._coordinates : x._global._coordinates ) {
			const auto data = internal::grb_BSP1D.cload();
			_s = data.s;
			_P = data.P;
			if( mode != PARALLEL ) {
				assert( mode == SEQUENTIAL );
				x.synchronize();
			}
		}

		/** No implementation notes. */
		inline IOType & operator[]( const size_t i ) noexcept {
			return _buffered_values[ i ];
		}

		/** No implementation notes. */
		inline const IOType & operator[]( const size_t i ) const noexcept {
			return _buffered_values[ i ];
		}

		/** No implementation notes. */
		bool mask( const size_t i ) const noexcept {
			return _buffered_mask.assigned( i );
		}

		/** No implementation notes. */
		size_t length() const noexcept {
			return _buffered_mask.size();
		}

		/** No implementation notes. */
		size_t index( size_t i ) const noexcept {
			const size_t length = _buffered_mask.size();
			size_t s = 0;
			size_t local_length = internal::Distribution< BSP1D >::global_length_to_local( length, s, _P );
			while( s < _P && i > local_length ) {
				i -= local_length;
				++s;
				local_length = internal::Distribution< BSP1D >::global_length_to_local( length, s, _P );
			}
			assert( s < _P );
			const size_t ret = internal::Distribution< BSP1D >::local_index_to_global( i, length, s, _P );
			return ret;
		}

		/**
		 * Frees the underlying raw memory area iff the underlying vector was
		 * destroyed. Otherwise set the underlying vector to unpinned state.
		 */
		void free() noexcept {
			_raw_deleter.clear();
			_assigned_deleter.clear();
		}
	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_PINNEDVECTOR''
