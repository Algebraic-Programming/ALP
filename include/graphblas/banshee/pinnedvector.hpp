
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
 * Contains the banshee and banshee_omp implementations for the
 * PinnedVector class.
 *
 * @author A. N. Yzelman
 */

#if ! defined _H_GRB_BANSHEE_PINNEDVECTOR
#define _H_GRB_BANSHEE_PINNEDVECTOR

#include <graphblas/banshee/coordinates.hpp>
#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/utils/autodeleter.hpp>

namespace grb {

	/** No implementation notes. */
	template< typename IOType >
	class PinnedVector< IOType, banshee > {

	private:
		/**
		 * Tell the system to delete \a _buffered_values only when we had its last
		 * banshee.
		 */
		utils::AutoDeleter< IOType > _raw_deleter;

		/**
		 * Tell the system to delete \a _buffered_mask only when we had its last
		 * banshee.
		 */
		utils::AutoDeleter< char > _assigned_deleter;

		/** A buffer of the local vector. */
		IOType * _buffered_values;

		/** A buffer of the sparsity pattern of \a _buffered_values. */
		internal::Coordinates< banshee > _buffered_mask;

	public:
		/** No implementation notes. */
		PinnedVector() : _buffered_values( NULL ) {}

		/** No implementation notes. */
		PinnedVector( const Vector< IOType, banshee > & x, IOMode mode ) :
			_raw_deleter( x._raw_deleter ), _assigned_deleter( x._assigned_deleter ), _buffered_values( x._raw ), _buffered_mask( x._coordinates ) {
			(void)mode; // sequential and parallel IO mode are equivalent for this implementation.
		}

		/** No implementation notes. */
		IOType & operator[]( const size_t i ) noexcept {
			return _buffered_values[ i ];
		}

		/** No implementation notes. */
		const IOType & operator[]( const size_t i ) const noexcept {
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
		size_t index( const size_t index ) const noexcept {
			return index;
		}

		/**
		 * Frees the underlying raw memory area iff the underlying vector was
		 * destroyed. Otherwise set the underlying vector to unpinned state.
		 */
		void free() {
			_raw_deleter.clear();
			_assigned_deleter.clear();
		}
	};

} // namespace grb

#endif // end ``_H_GRB_BANSHEE_PINNEDVECTOR
